"""Minimal embedding interface used only to drive *semantic chunking*
(Phase 1). Real retrieval embeddings (sentence-transformers, nomic-embed-text)
are a Phase 2 concern — this module deliberately stays dependency-light so
`ingest.py` runs without pulling in torch/sentence-transformers just to
split documents.

Any embedder here is just: list[str] -> np.ndarray of shape (n, dim).
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np

EmbedFn = Callable[[list[str]], np.ndarray]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingBagOfWordsEmbedder:
    """A dependency-free stand-in embedder.

    Hashes each lowercased word into one of `n_dims` buckets and builds an
    L2-normalized term-frequency vector. It's a crude proxy for meaning (no
    synonyms, no word order) but it's enough to detect topic shifts between
    adjacent sentences for semantic chunking, and it needs no model download.
    Swap in a real sentence-transformers embedder (see `get_embedder`) for
    better quality.
    """

    def __init__(self, n_dims: int = 512):
        self.n_dims = n_dims

    def __call__(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.n_dims), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                bucket = hash(token) % self.n_dims
                vectors[i, bucket] += 1.0
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        return vectors


def get_embedder(prefer_sentence_transformers: bool = True) -> EmbedFn:
    """Best available embedder: sentence-transformers if it's installed,
    otherwise the hashing bag-of-words fallback.
    """
    if prefer_sentence_transformers:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")
            return lambda texts: model.encode(texts, normalize_embeddings=True)
        except ImportError:
            pass
    return HashingBagOfWordsEmbedder()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
