"""The Phase 3 deliverable: a clean `retrieve(query, k)` interface, layering
optional hybrid search and cross-encoder re-ranking on top of plain top-k
dense vector search.

The project plan asks for a bare `retrieve(query, k) -> List[Chunk]`
function. `Retriever` implements that as a *bound method* instead of a
free function on purpose: a real call needs an embedder (possibly a loaded
sentence-transformers model), a cross-encoder (another loaded model), and a
BM25 index (built from a full strategy's chunk corpus) — all expensive to
construct and completely reusable across calls. A free function with that
signature would either have to reload all of them on every single query,
or reach for hidden global state to cache them. `Retriever(...)` is the
one-time setup; `.retrieve(query, k)` is the cheap, repeatable call — the
interface Phase 7's evaluation loop will actually call in a loop over 20-30
questions.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

from embeddings.base import Embedder
from ingest.chunkers import Chunk
from ingest.storage import load_chunks_jsonl

from .hybrid_search import BM25Index, reciprocal_rank_fusion
from .reranker import CrossEncoderReranker
from .vector_store import ChromaVectorStore, RetrievedChunk

_CHUNK_TOP_LEVEL_FIELDS = {f.name for f in fields(Chunk)} - {"metadata"}


@dataclass
class Retrieval:
    chunk: Chunk
    score: float  # higher = more relevant; NOT comparable across `method`s
    method: str  # "dense" | "hybrid_rrf" | "reranked"


def _chunk_from_stored(stored: RetrievedChunk, fallback_strategy: str) -> Chunk:
    """Rebuild a full `Chunk` from what Chroma actually stored (document
    text + flat metadata) — the inverse of `vector_store.chunk_to_metadata`.
    """
    m = stored.metadata
    extra_metadata = {k: v for k, v in m.items() if k not in _CHUNK_TOP_LEVEL_FIELDS}
    return Chunk(
        chunk_id=stored.chunk_id,
        doc_id=m.get("doc_id", ""),
        source_path=m.get("source_path", ""),
        filetype=m.get("filetype", ""),
        strategy=m.get("strategy", fallback_strategy),
        chunk_index=m.get("chunk_index", 0),
        text=stored.text,
        char_start=m.get("char_start", 0),
        char_end=m.get("char_end", 0),
        page_start=m.get("page_start"),
        page_end=m.get("page_end"),
        metadata=extra_metadata,
    )


class Retriever:
    def __init__(
        self,
        strategy: str,
        embedder: Embedder,
        store: ChromaVectorStore | None = None,
        processed_dir: str | Path = "data/processed",
        use_hybrid: bool = False,
        use_reranker: bool = False,
        pool_size: int = 20,
        reranker: CrossEncoderReranker | None = None,
    ):
        self.strategy = strategy
        self.embedder = embedder
        self.store = store or ChromaVectorStore()
        self.collection_name = f"{strategy}__{embedder.name}"
        self.pool_size = pool_size
        self.use_hybrid = use_hybrid
        self.use_reranker = use_reranker

        self._bm25 = BM25Index.from_jsonl(Path(processed_dir) / f"{strategy}.jsonl") if use_hybrid else None
        self._reranker = (reranker or CrossEncoderReranker()) if use_reranker else None

    def _dense_search(self, query: str, k: int) -> list[Retrieval]:
        query_vector = self.embedder.embed_query(query)
        stored = self.store.query(self.collection_name, query_vector, k=k)
        # Chroma's distance is smaller-is-better; invert it to a bounded
        # (0, 1] similarity-like score so every method's score means
        # "higher is more relevant," matching `Retrieval.score`'s contract.
        return [Retrieval(chunk=_chunk_from_stored(s, self.strategy), score=1.0 / (1.0 + s.distance), method="dense") for s in stored]

    def _hybrid_fuse(self, query: str, dense: list[Retrieval]) -> list[Retrieval]:
        bm25_hits = self._bm25.search(query, self.pool_size)
        chunk_by_id = {r.chunk.chunk_id: r.chunk for r in dense}
        chunk_by_id.update({chunk.chunk_id: chunk for chunk, _ in bm25_hits})

        dense_ranking = [r.chunk.chunk_id for r in dense]
        bm25_ranking = [chunk.chunk_id for chunk, _ in bm25_hits]
        fused_scores = reciprocal_rank_fusion([dense_ranking, bm25_ranking])

        ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
        return [Retrieval(chunk=chunk_by_id[cid], score=fused_scores[cid], method="hybrid_rrf") for cid in ranked_ids]

    def _rerank(self, query: str, candidates: list[Retrieval]) -> list[Retrieval]:
        scores = self._reranker.score(query, [c.chunk.text for c in candidates])
        reranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [Retrieval(chunk=c.chunk, score=float(s), method="reranked") for c, s in reranked]

    def retrieve_with_scores(self, query: str, k: int = 5) -> list[Retrieval]:
        """Full pipeline, exposing each candidate's score and which stage
        produced it — what Phase 7's evaluation and `retrieve.py`'s CLI
        output both actually want. `retrieve()` below is the trimmed-down
        `List[Chunk]` view for callers that just want the text."""
        needs_pool = self.use_hybrid or self.use_reranker
        candidates = self._dense_search(query, self.pool_size if needs_pool else k)

        if self.use_hybrid:
            candidates = self._hybrid_fuse(query, candidates)
        if self.use_reranker:
            candidates = self._rerank(query, candidates)

        return candidates[:k]

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        return [r.chunk for r in self.retrieve_with_scores(query, k)]
