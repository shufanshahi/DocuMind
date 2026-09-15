import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embeddings.base import Embedder  # noqa: E402
from embeddings.llamaindex_adapter import LlamaIndexEmbedderAdapter  # noqa: E402


class FakeEmbedder(Embedder):
    """A proper Embedder subclass (not just duck-typed): LlamaIndexEmbedderAdapter's
    `wrapped` field is validated by pydantic against the `Embedder` type, so a
    plain object without this base class fails construction, not just at
    the call site."""

    name = "fake"
    dimension = 3

    def embed_query(self, text: str) -> np.ndarray:
        return np.array([1.0, 2.0, 3.0])

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.array([[4.0, 5.0, 6.0] for _ in texts])


def test_query_embedding_delegates_to_wrapped_embed_query():
    adapter = LlamaIndexEmbedderAdapter(FakeEmbedder())
    assert adapter.get_query_embedding("anything") == [1.0, 2.0, 3.0]


def test_text_embedding_delegates_to_wrapped_embed_documents():
    adapter = LlamaIndexEmbedderAdapter(FakeEmbedder())
    assert adapter.get_text_embedding("anything") == [4.0, 5.0, 6.0]


def test_model_name_reflects_the_wrapped_embedder():
    adapter = LlamaIndexEmbedderAdapter(FakeEmbedder())
    assert adapter.model_name == "fake"
