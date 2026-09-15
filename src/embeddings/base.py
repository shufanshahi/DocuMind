"""Common interface every embedding backend implements, so `embed.py` and
(later) `retrieve.py` can treat sentence-transformers and nomic-embed-text
interchangeably.

Documents and queries are embedded through separate methods on purpose:
some models (nomic-embed-text is the example used here) are *asymmetric* —
trained with different instruction prefixes for the text being indexed
versus the text being searched for — and produce measurably worse
similarity scores if you embed both the same way. A model that doesn't
need the distinction (sentence-transformers here) just implements both
methods identically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    name: str
    dimension: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed chunk text for storage. Shape: (len(texts), self.dimension)."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single user query for similarity search. Shape: (self.dimension,)."""
