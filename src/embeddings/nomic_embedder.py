"""Local embedding via nomic-embed-text, served by Ollama. Requires an
Ollama daemon running locally with the model pulled:

    ollama pull nomic-embed-text

nomic-embed-text is a *task-instruction-tuned* embedding model: its model
card specifies different literal prefixes for text being indexed versus
text being searched for (`search_document: ...` vs `search_query: ...`).
Skipping this halves retrieval quality in practice — the two prefixes
steer the model into slightly different regions of its embedding space, so
a bare, unprefixed query drifts away from how documents were encoded. See
explaination/phase2_theory.md for more on why asymmetric models exist.
"""

from __future__ import annotations

import numpy as np
import requests

from .base import Embedder

DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class NomicEmbedder(Embedder):
    name = "nomic_embed_text"

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimension = len(self._embed_raw([DOCUMENT_PREFIX + "dimension probe"])[0])

    def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        resp = requests.post(f"{self.base_url}/api/embed", json={"model": self.model, "input": texts}, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama embedding request failed ({resp.status_code}): {resp.text}\n"
                f"Is Ollama running with `{self.model}` pulled? Try: ollama pull {self.model}"
            )
        return resp.json()["embeddings"]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        prefixed = [DOCUMENT_PREFIX + t for t in texts]
        return np.asarray(self._embed_raw(prefixed))

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self._embed_raw([QUERY_PREFIX + text])[0])
