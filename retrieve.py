#!/usr/bin/env python3
"""Phase 3 deliverable: query one of Phase 2's embedded collections through
`Retriever.retrieve(query, k) -> List[Chunk]`, with optional BM25+dense
hybrid search and cross-encoder re-ranking layered on top of plain top-k
vector similarity search.

Usage:
    python retrieve.py "how does chunk size affect retrieval quality"
    python retrieve.py "..." --strategy semantic --embedder sentence_transformers
    python retrieve.py "..." --hybrid --rerank --k 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from embeddings.nomic_embedder import NomicEmbedder  # noqa: E402
from embeddings.st_embedder import SentenceTransformerEmbedder  # noqa: E402
from ingest.chunkers import STRATEGIES  # noqa: E402
from retrieval.retriever import Retriever  # noqa: E402

EMBEDDER_FACTORIES = {
    "sentence_transformers": SentenceTransformerEmbedder,
    "nomic_embed_text": NomicEmbedder,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query")
    parser.add_argument("--strategy", choices=STRATEGIES, default="recursive")
    parser.add_argument("--embedder", choices=list(EMBEDDER_FACTORIES), default="nomic_embed_text")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=20, help="candidates pulled from dense search before hybrid/rerank narrows to k")
    parser.add_argument("--hybrid", action="store_true", help="fuse BM25 keyword search with dense search via reciprocal rank fusion")
    parser.add_argument("--rerank", action="store_true", help="re-score the candidate pool with a cross-encoder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedder = EMBEDDER_FACTORIES[args.embedder]()
    retriever = Retriever(
        strategy=args.strategy,
        embedder=embedder,
        use_hybrid=args.hybrid,
        use_reranker=args.rerank,
        pool_size=args.pool_size,
    )

    results = retriever.retrieve_with_scores(args.query, k=args.k)

    print(f"Query: {args.query!r}")
    print(f"Collection: {retriever.collection_name}  hybrid={args.hybrid}  rerank={args.rerank}\n")
    for i, r in enumerate(results, 1):
        page = f", p.{r.chunk.page_start}" if r.chunk.page_start else ""
        preview = " ".join(r.chunk.text.split())[:160]
        print(f"{i}. [{r.method} score={r.score:.4f}] {r.chunk.source_path}{page}")
        print(f"   {preview}...\n")


if __name__ == "__main__":
    main()
