#!/usr/bin/env python3
"""Phase 1 deliverable: read a folder of documents (PDF/Markdown/plain text),
run each of the 4 chunking strategies over them, and write the resulting
metadata-tagged chunks out separately per strategy so they can be compared
later (Phase 7 evaluation).

Usage:
    python ingest.py --input-dir data/raw --output-dir data/processed
    python ingest.py --strategies fixed_size recursive --chunk-size 300
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingest.chunkers import STRATEGIES, chunk_document  # noqa: E402
from ingest.embedders import get_embedder  # noqa: E402
from ingest.loaders import load_documents_from_folder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--chunk-size", type=int, default=500, help="fixed_size / recursive (chars)")
    parser.add_argument("--overlap", type=int, default=50, help="fixed_size / recursive (chars)")
    parser.add_argument("--semantic-threshold", type=float, default=0.2, help="cosine similarity cutoff for a topic shift")
    parser.add_argument("--window-size", type=int, default=2, help="sentence_window: sentences of context per side")
    parser.add_argument(
        "--sentence-transformers",
        action="store_true",
        help="use a real sentence-transformers model for semantic chunking instead of the dependency-free fallback",
    )
    return parser.parse_args()


def strategy_kwargs(args: argparse.Namespace, strategy: str) -> dict:
    if strategy in ("fixed_size", "recursive"):
        return {"chunk_size": args.chunk_size, "overlap": args.overlap}
    if strategy == "semantic":
        return {"similarity_threshold": args.semantic_threshold}
    if strategy == "sentence_window":
        return {"window_size": args.window_size}
    return {}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    documents = load_documents_from_folder(args.input_dir)
    if not documents:
        print(f"No supported documents found under {args.input_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(documents)} document(s) from {args.input_dir}")

    embed_fn = get_embedder(prefer_sentence_transformers=args.sentence_transformers) if "semantic" in args.strategies else None

    summary_rows = []
    for strategy in args.strategies:
        out_path = args.output_dir / f"{strategy}.jsonl"
        kwargs = strategy_kwargs(args, strategy)
        all_chunks = []
        with out_path.open("w", encoding="utf-8") as f:
            for doc in documents:
                fn_kwargs = dict(kwargs)
                if strategy == "semantic":
                    fn_kwargs["embed_fn"] = embed_fn
                chunks = chunk_document(doc, strategy, **fn_kwargs)
                all_chunks.extend(chunks)
                for chunk in chunks:
                    f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

        chunk_count = len(all_chunks)
        avg_len = sum(len(c.text) for c in all_chunks) / chunk_count if chunk_count else 0
        summary_rows.append((strategy, chunk_count, avg_len))
        print(f"  [{strategy}] {chunk_count} chunks -> {out_path} (avg {avg_len:.0f} chars/chunk)")

    print("\nSummary")
    print(f"{'strategy':<18}{'chunks':>10}{'avg chars/chunk':>20}")
    for strategy, count, avg_len in summary_rows:
        print(f"{strategy:<18}{count:>10}{avg_len:>20.1f}")


if __name__ == "__main__":
    main()
