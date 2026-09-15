"""Wraps one of our own `Embedder` implementations (`SentenceTransformerEmbedder`,
`NomicEmbedder`) so LlamaIndex's `VectorStoreIndex` can use it to embed
queries against an already-populated Chroma collection.

Why an adapter instead of LlamaIndex's own embedding integrations
(`HuggingFaceEmbedding`, `OllamaEmbedding`): a Chroma collection built by
Phase 2's `embed.py` was populated using *our* embedder classes, and
`NomicEmbedder` applies nomic-embed-text's required asymmetric
`search_document:`/`search_query:` prefixes (see
explaination/phase2_theory.md). LlamaIndex's own `OllamaEmbedding` is a
generic wrapper with no knowledge of that model-specific convention. Using
it here would query the collection with vectors computed in a subtly
different way than the vectors already stored in it — a real correctness
bug that would silently degrade retrieval, not an error. Delegating to our
own `Embedder` guarantees exact parity with however the collection was
actually built, no matter which framework is doing the querying.
"""

from __future__ import annotations

from llama_index.core.embeddings import BaseEmbedding

from .base import Embedder


class LlamaIndexEmbedderAdapter(BaseEmbedding):
    """A `BaseEmbedding` that delegates every call to a wrapped `Embedder`.
    `arbitrary_types_allowed` is already set on `BaseEmbedding`'s pydantic
    config, so `wrapped` can be typed as our own (non-pydantic) `Embedder`
    directly, no `PrivateAttr` workaround needed.
    """

    wrapped: Embedder

    def __init__(self, wrapped: Embedder, **kwargs):
        super().__init__(wrapped=wrapped, model_name=wrapped.name, **kwargs)

    def _get_query_embedding(self, query: str) -> list[float]:
        return list(map(float, self.wrapped.embed_query(query)))

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return list(map(float, self.wrapped.embed_documents([text])[0]))
