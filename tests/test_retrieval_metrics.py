import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval.retrieval_metrics import precision_at_k, recall_at_k  # noqa: E402


def test_precision_at_k_all_relevant():
    retrieved = ["a.md", "a.md", "a.md"]
    assert precision_at_k(retrieved, {"a.md"}, k=3) == 1.0


def test_precision_at_k_none_relevant():
    retrieved = ["b.md", "c.md"]
    assert precision_at_k(retrieved, {"a.md"}, k=2) == 0.0


def test_precision_at_k_partial():
    retrieved = ["a.md", "b.md", "c.md", "a.md"]
    assert precision_at_k(retrieved, {"a.md"}, k=4) == 0.5


def test_precision_at_k_only_considers_top_k():
    # Two hits total, but only the top-2 slice should count.
    retrieved = ["b.md", "a.md", "a.md"]
    assert precision_at_k(retrieved, {"a.md"}, k=2) == 0.5


def test_precision_at_k_zero_k():
    assert precision_at_k(["a.md"], {"a.md"}, k=0) == 0.0


def test_recall_at_k_finds_all_expected_sources():
    retrieved = ["a.md", "b.md", "c.md"]
    assert recall_at_k(retrieved, {"a.md", "b.md"}, k=3) == 1.0


def test_recall_at_k_finds_half_of_multi_hop_sources():
    retrieved = ["a.md", "c.md"]
    assert recall_at_k(retrieved, {"a.md", "b.md"}, k=2) == 0.5


def test_recall_at_k_only_considers_top_k():
    retrieved = ["c.md", "a.md"]  # a.md is ranked 2nd
    assert recall_at_k(retrieved, {"a.md"}, k=1) == 0.0
    assert recall_at_k(retrieved, {"a.md"}, k=2) == 1.0


def test_recall_at_k_raises_on_empty_expected_sources():
    # Adversarial questions have no expected source -- recall isn't a
    # meaningful metric for them, so this should fail loudly rather than
    # silently returning 0.0 or 1.0.
    with pytest.raises(ValueError):
        recall_at_k(["a.md"], set(), k=3)
