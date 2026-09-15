"""Local embedding via sentence-transformers. Runs entirely on-device, no
network call, no per-token cost — see explaination/phase2_theory.md for the
tradeoffs against the Ollama-hosted nomic-embed-text alternative.
"""

from __future__ import annotations

import numpy as np

from .base import Embedder


class SentenceTransformerEmbedder(Embedder):
    name = "sentence_transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import (torch)

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False))

    def embed_query(self, text: str) -> np.ndarray:
        # Symmetric model: a query is embedded the same way as a document.
        return self.embed_documents([text])[0]
