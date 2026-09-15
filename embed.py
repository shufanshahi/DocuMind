#!/usr/bin/env python3
"""Phase 2 deliverable: embed the chunked output of `ingest.py` with two
different embedding backends (sentence-transformers, local; nomic-embed-text,
via Ollama) and store each (strategy, embedder) combination in its own
ChromaDB collection, so Phase 3+ can A/B test retrieval quality across both
embedding models and all 4 chunking strategies.

Usage:
    python embed.py                                   # every strategy x both embedders
    python embed.py --strategies recursive --embedders sentence_transformers
    python embed.py --input-dir data/processed --persist-dir data/chroma_db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from embeddings.base import Embedder  # noqa: E402
from embeddings.nomic_embedder import NomicEmbedder  # noqa: E402
from embeddings.st_embedder import SentenceTransformerEmbedder  # noqa: E402
from ingest.chunkers import STRATEGIES  # noqa: E402
from ingest.storage import load_chunks_jsonl  # noqa: E402
from retrieval.vector_store import ChromaVectorStore  # noqa: E402

EMBEDDER_FACTORIES = {
    "sentence_transformers": SentenceTransformerEmbedder,
    "nomic_embed_text": NomicEmbedder,
}


def embed_in_batches(embedder: Embedder, texts: list[str], batch_size: int = 32):
    vectors = []
    for i in range(0, len(texts), batch_size):
        vectors.extend(embedder.embed_documents(texts[i : i + batch_size]))
    return vectors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--persist-dir", type=Path, default=Path("data/chroma_db"))
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--embedders", nargs="+", choices=list(EMBEDDER_FACTORIES), default=list(EMBEDDER_FACTORIES))
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = ChromaVectorStore(persist_dir=args.persist_dir)

    embedders = {}
    for name in args.embedders:
        print(f"Loading embedder: {name} ...")
        embedders[name] = EMBEDDER_FACTORIES[name]()

    summary = []
    for strategy in args.strategies:
        jsonl_path = args.input_dir / f"{strategy}.jsonl"
        if not jsonl_path.exists():
            print(f"Skipping {strategy!r}: {jsonl_path} not found (run ingest.py first)", file=sys.stderr)
            continue
        chunks = load_chunks_jsonl(jsonl_path)
        if not chunks:
            continue
        texts = [c.text for c in chunks]

        for embedder_name, embedder in embedders.items():
            collection_name = f"{strategy}__{embedder_name}"
            print(f"Embedding {len(chunks)} chunks from {strategy!r} with {embedder_name!r} ...")
            vectors = embed_in_batches(embedder, texts, batch_size=args.batch_size)
            store.upsert_chunks(collection_name, embedder_name, chunks, vectors)
            count = store.count(collection_name)
            summary.append((collection_name, count, embedder.dimension))
            print(f"  -> collection {collection_name!r}: {count} vectors, dim={embedder.dimension}")

    print("\nSummary")
    print(f"{'collection':<40}{'vectors':>10}{'dim':>8}")
    for name, count, dim in summary:
        print(f"{name:<40}{count:>10}{dim:>8}")
    print(f"\nPersisted to {args.persist_dir}/")


if __name__ == "__main__":
    main()
