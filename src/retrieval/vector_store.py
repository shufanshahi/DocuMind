"""Thin wrapper around a persistent ChromaDB client.

Kept deliberately small: it only knows how to upsert already-embedded
chunks into a named collection and query a collection by vector. Anything
about *how* chunks got embedded belongs in `src/embeddings/`, and anything
about re-ranking or hybrid search (Phase 3) belongs in `src/retrieval/` but
not in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb


def chunk_to_metadata(chunk) -> dict[str, Any]:
    """Flatten a `Chunk` (see src/ingest/chunkers.py) into the flat
    str/int/float/bool-only dict Chroma requires as metadata. `None` fields
    (e.g. page numbers on a non-PDF source) are dropped rather than sent as
    `None`, which Chroma rejects.
    """
    fields = {
        "doc_id": chunk.doc_id,
        "source_path": chunk.source_path,
        "filetype": chunk.filetype,
        "strategy": chunk.strategy,
        "chunk_index": chunk.chunk_index,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
    }
    fields.update(chunk.metadata or {})
    return {k: v for k, v in fields.items() if v is not None}


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float


class ChromaVectorStore:
    def __init__(self, persist_dir: str | Path = "data/chroma_db"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))

    def get_or_create_collection(self, name: str, embedder_name: str, dimension: int):
        # Stamping the embedder's name/dimension into the collection's own
        # metadata makes an accidental cross-embedder mix-up (e.g. querying
        # a nomic-embed-text collection with a sentence-transformers vector)
        # visible immediately instead of silently returning garbage results.
        return self._client.get_or_create_collection(
            name=name,
            metadata={"embedder": embedder_name, "dimension": dimension},
        )

    def upsert_chunks(self, collection_name: str, embedder_name: str, chunks: list, embeddings) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"got {len(chunks)} chunks but {len(embeddings)} embeddings")
        if not chunks:
            return
        collection = self.get_or_create_collection(collection_name, embedder_name, len(embeddings[0]))
        collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=[list(map(float, e)) for e in embeddings],
            documents=[c.text for c in chunks],
            metadatas=[chunk_to_metadata(c) for c in chunks],
        )

    def count(self, collection_name: str) -> int:
        return self._client.get_collection(collection_name).count()

    def query(self, collection_name: str, query_embedding, k: int = 5) -> list[RetrievedChunk]:
        collection = self._client.get_collection(collection_name)
        result = collection.query(query_embeddings=[list(map(float, query_embedding))], n_results=k)
        return [
            RetrievedChunk(chunk_id=chunk_id, text=text, metadata=metadata, distance=distance)
            for chunk_id, text, metadata, distance in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def list_collections(self) -> list[str]:
        return [c.name for c in self._client.list_collections()]
