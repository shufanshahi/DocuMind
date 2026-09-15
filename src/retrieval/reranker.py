"""Cross-encoder re-ranking.

A dense vector search (`vector_store.py`) scores a query against every
chunk *independently* — the query and the chunk are each embedded once, on
their own, and never see each other. A cross-encoder instead takes the
(query, chunk) *pair* as joint input to one model and outputs a single
relevance score, which lets it model interactions a bi-encoder can't (e.g.
"does this chunk answer the specific thing being asked," not just "is this
chunk topically similar"). That's more accurate but far more expensive —
it requires one full model forward pass per (query, chunk) pair, so it's
only run over a small candidate pool the dense search already narrowed
down, never over the whole collection.
"""

from __future__ import annotations

import numpy as np


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder  # lazy: heavy import (torch)

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        """Higher score = more relevant. Not a probability or a distance —
        only meaningful as a ranking within this one call's `texts`."""
        pairs = [(query, text) for text in texts]
        return np.asarray(self._model.predict(pairs))
