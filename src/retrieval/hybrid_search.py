"""BM25 keyword search, and reciprocal rank fusion (RRF) to combine its
ranking with dense vector search's ranking.

Dense search and BM25 fail in complementary ways. Dense search can miss an
exact, rare token (a product SKU, an error code, a proper noun) if that
token's presence doesn't shift the sentence's overall meaning much — the
embedding is still "close enough" to plausible-but-wrong passages. BM25
finds exact token matches perfectly but has no notion of synonymy or
paraphrase at all. Combining the two rankings covers each one's blind
spot; see explaination/phase3_theory.md for the fuller argument and why
RRF specifically (rather than, say, averaging raw scores) is the standard
way to combine them.
"""

from __future__ import annotations

import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from ingest.chunkers import Chunk
from ingest.storage import load_chunks_jsonl

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """An in-memory BM25 index over one chunking strategy's chunks. Built
    once per (strategy, retriever), not per query — BM25Okapi computes
    corpus-wide statistics (document frequencies, average doc length) up
    front from the tokenized corpus.
    """

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "BM25Index":
        return cls(load_chunks_jsonl(Path(path)))

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in ranked_indices]


def reciprocal_rank_fusion(rankings: list[list[str]], k_constant: int = 60) -> dict[str, float]:
    """Combine several ranked lists of chunk IDs (best first) into one
    fused score per chunk ID: sum(1 / (k_constant + rank)) across every
    ranking the chunk appears in (rank is 0-indexed).

    RRF only looks at *rank position*, never at the raw scores, which is
    exactly why it works to combine two methods (BM25 scores, cosine
    distances) whose raw magnitudes aren't comparable to each other at all
    — see the module docstring and phase3_theory.md. `k_constant=60` is
    the value used in the original RRF paper and most implementations
    since; it just dampens how much the very top rank dominates.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k_constant + rank + 1)
    return fused
