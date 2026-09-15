"""Read the .jsonl chunk files `ingest.py` writes back into `Chunk` objects.
Shared by `embed.py` (Phase 2) and `src/retrieval/hybrid_search.py` (Phase 3),
so both stay in sync with the on-disk format without duplicating the parsing.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from .chunkers import Chunk

_CHUNK_FIELDS = {f.name for f in fields(Chunk)}


def load_chunks_jsonl(path: Path) -> list[Chunk]:
    chunks = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            chunks.append(Chunk(**{k: v for k, v in data.items() if k in _CHUNK_FIELDS}))
    return chunks
