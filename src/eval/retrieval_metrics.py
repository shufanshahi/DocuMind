"""Retrieval quality metrics: Precision@k and Recall@k, computed against
ground truth at the *source document* level.

Ground truth is deliberately not "the exact right chunk" — the whole point
of Phase 7 is comparing all 4 chunking strategies against each other, and
each strategy carves the same document into different chunk boundaries.
There is no single "correct chunk ID" that means the same thing across
fixed_size, recursive, semantic, and sentence_window output. What *is*
comparable across all four is "which source file did the answer need to
come from" — so that's the ground truth unit these metrics use. See
explaination/phase7_theory.md §1 for the fuller argument.
"""

from __future__ import annotations


def precision_at_k(retrieved_sources: list[str], expected_sources: set[str], k: int) -> float:
    """Of the top-k retrieved chunks, what fraction came from an expected
    source document? Requires `expected_sources` to be non-empty (callers
    should skip adversarial cases, where nothing is truly relevant and
    this metric isn't meaningful — see `evaluate_retrieval` below)."""
    if k == 0:
        return 0.0
    top_k = retrieved_sources[:k]
    hits = sum(1 for source in top_k if source in expected_sources)
    return hits / k


def recall_at_k(retrieved_sources: list[str], expected_sources: set[str], k: int) -> float:
    """Of the expected source documents, what fraction showed up somewhere
    in the top-k retrieved chunks? For a multi-hop question with 2 expected
    sources, finding only 1 of them gives 0.5, not a pass/fail."""
    if not expected_sources:
        raise ValueError("recall_at_k is undefined for an empty expected_sources set")
    top_k = set(retrieved_sources[:k])
    hits = len(expected_sources & top_k)
    return hits / len(expected_sources)
