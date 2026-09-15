import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingest.chunkers import Chunk  # noqa: E402
from retrieval.retriever import Retriever  # noqa: E402
from retrieval.vector_store import ChromaVectorStore  # noqa: E402


class FakeEmbedder:
    """A 2D embedder: hand-place chunks/queries wherever a test needs them,
    with no real model involved, so `Retriever` can be tested without
    downloading or running sentence-transformers or Ollama."""

    name = "fake_embedder"
    dimension = 2

    def __init__(self, query_vector):
        self._query_vector = query_vector

    def embed_query(self, text):
        return np.asarray(self._query_vector)


class FakeReranker:
    """Deterministically reverses whatever order it's given (ascending
    scores mean the *last* candidate it's handed scores highest, so sorting
    by score descending flips the input order), so a test can assert
    re-ranking actually changed the result order."""

    def score(self, query, texts):
        return np.arange(len(texts), dtype=float)


def make_chunk(chunk_id, text):
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        source_path="data/raw/doc.txt",
        filetype="text",
        strategy="fixed_size",
        chunk_index=int(chunk_id[-1]),
        text=text,
        char_start=0,
        char_end=len(text),
    )


CHUNKS = [
    make_chunk("c0", "cats are small domestic animals"),
    make_chunk("c1", "rockets fly to space using thrust"),
    make_chunk("c2", "dogs are loyal domestic animals"),
]
# Placed on a circle so "closest to (1, 0)" is unambiguous: c0 exact match,
# c2 near, c1 far.
EMBEDDINGS = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]


def setup_store(tmp_path):
    store = ChromaVectorStore(persist_dir=tmp_path / "chroma")
    store.upsert_chunks("fixed_size__fake_embedder", "fake_embedder", CHUNKS, EMBEDDINGS)
    return store


def write_processed_jsonl(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    with (processed_dir / "fixed_size.jsonl").open("w") as f:
        for c in CHUNKS:
            f.write(json.dumps(asdict(c)) + "\n")
    return processed_dir


def test_dense_retrieve_returns_chunks_in_similarity_order(tmp_path):
    store = setup_store(tmp_path)
    retriever = Retriever(strategy="fixed_size", embedder=FakeEmbedder([1.0, 0.0]), store=store)
    results = retriever.retrieve("query", k=3)
    assert [c.chunk_id for c in results] == ["c0", "c2", "c1"]


def test_retrieve_with_scores_reports_dense_method(tmp_path):
    store = setup_store(tmp_path)
    retriever = Retriever(strategy="fixed_size", embedder=FakeEmbedder([1.0, 0.0]), store=store)
    results = retriever.retrieve_with_scores("query", k=1)
    assert results[0].method == "dense"
    assert results[0].chunk.chunk_id == "c0"


def test_reranker_can_change_order(tmp_path):
    store = setup_store(tmp_path)
    retriever = Retriever(
        strategy="fixed_size",
        embedder=FakeEmbedder([1.0, 0.0]),
        store=store,
        use_reranker=True,
        reranker=FakeReranker(),
        pool_size=3,
    )
    results = retriever.retrieve_with_scores("query", k=3)
    # FakeReranker reverses dense order (c0, c2, c1) -> (c1, c2, c0)
    assert [r.chunk.chunk_id for r in results] == ["c1", "c2", "c0"]
    assert all(r.method == "reranked" for r in results)


def test_hybrid_promotes_a_keyword_match_dense_search_ranks_last(tmp_path):
    store = setup_store(tmp_path)
    processed_dir = write_processed_jsonl(tmp_path)
    # Query embedding [1, 0] is an exact dense match for c0, near c2, and far
    # from c1 -- so dense-only search ranks c1 dead last. But "rockets
    # thrust" is a keyword match for c1's text and nothing else. Hybrid
    # search should promote c1 above c2 even though dense search alone
    # ranked it lowest.
    dense_only = Retriever(strategy="fixed_size", embedder=FakeEmbedder([1.0, 0.0]), store=store)
    dense_order = [c.chunk_id for c in dense_only.retrieve("rockets thrust", k=3)]
    assert dense_order == ["c0", "c2", "c1"]  # c1 ranked last by dense search alone

    hybrid = Retriever(
        strategy="fixed_size",
        embedder=FakeEmbedder([1.0, 0.0]),
        store=store,
        processed_dir=processed_dir,
        use_hybrid=True,
        pool_size=3,
    )
    hybrid_results = hybrid.retrieve_with_scores("rockets thrust", k=3)
    hybrid_order = [r.chunk.chunk_id for r in hybrid_results]
    assert hybrid_order.index("c1") < dense_order.index("c1")  # BM25 pulled it up
    assert all(r.method == "hybrid_rrf" for r in hybrid_results)


def test_retrieve_reconstructs_pdf_page_metadata(tmp_path):
    pdf_chunk = Chunk(
        chunk_id="p0",
        doc_id="doc-2",
        source_path="data/raw/doc.pdf",
        filetype="pdf",
        strategy="fixed_size",
        chunk_index=0,
        text="a pdf chunk",
        char_start=0,
        char_end=11,
        page_start=3,
        page_end=3,
    )
    store = ChromaVectorStore(persist_dir=tmp_path / "chroma2")
    store.upsert_chunks("fixed_size__fake_embedder", "fake_embedder", [pdf_chunk], [[1.0, 0.0]])
    retriever = Retriever(strategy="fixed_size", embedder=FakeEmbedder([1.0, 0.0]), store=store)
    result = retriever.retrieve("query", k=1)[0]
    assert result.page_start == 3
    assert result.page_end == 3
