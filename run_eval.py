#!/usr/bin/env python3
"""Phase 7 deliverable: evaluate the RAG pipeline against the labeled eval
set (data/eval/eval_set.jsonl) instead of eyeballing outputs.

Two passes, deliberately different in cost:

1. Retrieval-only metrics (Precision@k, Recall@k) across all 8
   (chunking strategy x embedder) combinations. No LLM calls -- just
   embedding the query and searching Chroma -- so running all 8 is cheap.
2. Generation metrics (faithfulness, relevancy via a custom LLM-judge
   rubric, a context-precision proxy, and refusal accuracy on adversarial
   questions) across 4 "headline" configs: fixed_size and semantic
   chunking, crossed with both embedders -- the same 4-config shape the
   project plan's own example results table (§9) uses. Every generation
   metric needs at least one LLM call (a generation, sometimes two more
   for judging), so this pass is intentionally scoped to 4 configs rather
   than all 8 -- see explaination/phase7_theory.md §4 for the full
   reasoning.

Usage:
    python run_eval.py                    # both passes, writes eval_results.md
    python run_eval.py --retrieval-only    # skip the LLM-heavy generation pass
    python run_eval.py --k 3
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from langchain_ollama import ChatOllama  # noqa: E402

from embeddings.nomic_embedder import NomicEmbedder  # noqa: E402
from embeddings.st_embedder import SentenceTransformerEmbedder  # noqa: E402
from eval.eval_set import EvalCase, load_eval_set  # noqa: E402
from eval.llm_judge import judge_answer  # noqa: E402
from eval.retrieval_metrics import precision_at_k, recall_at_k  # noqa: E402
from guardrail_experiments import is_refusal  # noqa: E402
from ingest.chunkers import STRATEGIES  # noqa: E402
from prompts.templates import format_chunk_for_citation  # noqa: E402
from rag.context_budget import TokenCounter  # noqa: E402
from rag_langchain import ask, build_chain  # noqa: E402
from retrieval.retriever import Retriever  # noqa: E402

EMBEDDER_FACTORIES = {
    "sentence_transformers": SentenceTransformerEmbedder,
    "nomic_embed_text": NomicEmbedder,
}
GENERATION_STRATEGIES = ("fixed_size", "semantic")  # the plan's own §9 example table's two strategies


@dataclass
class RetrievalResult:
    strategy: str
    embedder: str
    precision: float
    recall: float
    n_cases: int


@dataclass
class GenerationResult:
    strategy: str
    embedder: str
    faithfulness: float | None
    relevancy: float | None
    context_precision: float
    refusal_accuracy: float
    n_answerable: int
    n_adversarial: int
    n_judge_parse_errors: int
    transcript: list[dict] = field(default_factory=list)


def evaluate_retrieval(strategy: str, embedder, cases: list[EvalCase], k: int) -> RetrievalResult:
    retriever = Retriever(strategy=strategy, embedder=embedder)
    precisions, recalls = [], []
    for case in cases:
        if not case.expected_sources:
            continue  # adversarial cases have no ground truth to score precision/recall against
        chunks = retriever.retrieve(case.question, k=k)
        retrieved_sources = [c.source_path for c in chunks]
        expected = set(case.expected_sources)
        precisions.append(precision_at_k(retrieved_sources, expected, k))
        recalls.append(recall_at_k(retrieved_sources, expected, k))
    return RetrievalResult(
        strategy=strategy,
        embedder=embedder.name,
        precision=sum(precisions) / len(precisions),
        recall=sum(recalls) / len(recalls),
        n_cases=len(precisions),
    )


def evaluate_generation(strategy: str, embedder, cases: list[EvalCase], k: int, llm, judge_llm, token_budget: int) -> GenerationResult:
    retriever = Retriever(strategy=strategy, embedder=embedder)
    chain = build_chain(retriever, llm, k=k, token_budget=token_budget, counter=TokenCounter())

    faithfulness_scores, relevancy_scores, context_precisions = [], [], []
    refusal_correct, n_adversarial = 0, 0
    parse_errors = 0
    transcript = []

    for case in cases:
        try:
            result = ask(chain, case.question)
        except Exception as e:  # a single flaky Ollama call shouldn't sink a multi-minute run
            transcript.append({"id": case.id, "question": case.question, "error": str(e)})
            continue

        row = {"id": case.id, "category": case.category, "question": case.question, "answer": result.answer}

        if case.category == "adversarial":
            n_adversarial += 1
            correct = is_refusal(result.answer)
            refusal_correct += int(correct)
            row["refused"] = correct
        else:
            used_sources = [c.source_path for c in result.used_chunks]
            expected = set(case.expected_sources)
            hits = sum(1 for s in used_sources if s in expected)
            context_precisions.append(hits / len(used_sources) if used_sources else 0.0)

            # Reconstruct the exact citation-tagged context the generator
            # saw (assemble_context's numbering), not just the raw chunk
            # text -- the judge should see precisely what the answer was
            # allowed to draw from, [source: N] tags included.
            judged_context = "\n\n".join(format_chunk_for_citation(i, c.text) for i, c in enumerate(result.used_chunks, 1))
            score = judge_answer(judge_llm, case.question, judged_context, result.answer)
            if score.parse_error:
                parse_errors += 1
            if score.faithfulness is not None:
                faithfulness_scores.append(score.faithfulness)
            if score.relevancy is not None:
                relevancy_scores.append(score.relevancy)
            row.update(faithfulness=score.faithfulness, relevancy=score.relevancy)

        transcript.append(row)

    return GenerationResult(
        strategy=strategy,
        embedder=embedder.name,
        faithfulness=(sum(faithfulness_scores) / len(faithfulness_scores)) if faithfulness_scores else None,
        relevancy=(sum(relevancy_scores) / len(relevancy_scores)) if relevancy_scores else None,
        context_precision=(sum(context_precisions) / len(context_precisions)) if context_precisions else 0.0,
        refusal_accuracy=(refusal_correct / n_adversarial) if n_adversarial else 0.0,
        n_answerable=len(context_precisions),
        n_adversarial=n_adversarial,
        n_judge_parse_errors=parse_errors,
        transcript=transcript,
    )


def format_retrieval_table(results: list[RetrievalResult], k: int) -> str:
    lines = [f"| Chunking strategy | Embedder | Precision@{k} | Recall@{k} | n |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r.strategy} | {r.embedder} | {r.precision:.2f} | {r.recall:.2f} | {r.n_cases} |")
    return "\n".join(lines)


def format_generation_table(results: list[GenerationResult]) -> str:
    lines = [
        "| Chunking strategy | Embedder | Faithfulness (1-5) | Relevancy (1-5) | Context precision | Refusal accuracy |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        faith = f"{r.faithfulness:.2f}" if r.faithfulness is not None else "n/a"
        rel = f"{r.relevancy:.2f}" if r.relevancy is not None else "n/a"
        lines.append(f"| {r.strategy} | {r.embedder} | {faith} | {rel} | {r.context_precision:.2f} | {r.refusal_accuracy:.2f} |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--token-budget", type=int, default=2000)
    parser.add_argument("--eval-set", type=Path, default=Path("data/eval/eval_set.jsonl"))
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM-heavy generation pass")
    parser.add_argument("--output", type=Path, default=Path("eval_results.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_eval_set(args.eval_set)
    print(f"Loaded {len(cases)} eval cases from {args.eval_set}")

    embedders = {name: factory() for name, factory in EMBEDDER_FACTORIES.items()}

    print("\n=== Pass 1/2: retrieval-only metrics (all 4 strategies x both embedders) ===")
    retrieval_results = []
    for strategy, embedder_name in product(STRATEGIES, EMBEDDER_FACTORIES):
        t0 = time.time()
        result = evaluate_retrieval(strategy, embedders[embedder_name], cases, args.k)
        print(f"  {strategy:<16}{embedder_name:<22} P@{args.k}={result.precision:.2f}  R@{args.k}={result.recall:.2f}  ({time.time() - t0:.1f}s)")
        retrieval_results.append(result)

    generation_results = []
    if not args.retrieval_only:
        print(f"\n=== Pass 2/2: generation metrics ({len(GENERATION_STRATEGIES)} strategies x both embedders) ===")
        llm = ChatOllama(model="llama3.1:8b", temperature=0)
        for strategy, embedder_name in product(GENERATION_STRATEGIES, EMBEDDER_FACTORIES):
            t0 = time.time()
            result = evaluate_generation(strategy, embedders[embedder_name], cases, args.k, llm, llm, args.token_budget)
            print(
                f"  {strategy:<16}{embedder_name:<22} faithfulness={result.faithfulness} relevancy={result.relevancy} "
                f"ctx_precision={result.context_precision:.2f} refusal_acc={result.refusal_accuracy:.2f} ({time.time() - t0:.0f}s)"
            )
            generation_results.append(result)

    report = [
        "# Phase 7 — Evaluation Results\n",
        f"Generated by `run_eval.py` against `{args.eval_set}` ({len(cases)} cases), k={args.k}, token_budget={args.token_budget}.\n",
        "## Retrieval quality (all 4 chunking strategies x both embedders)\n",
        format_retrieval_table(retrieval_results, args.k),
        "",
    ]
    if generation_results:
        report += [
            "\n## Generation quality (fixed_size & semantic chunking x both embedders)\n",
            format_generation_table(generation_results),
            "",
        ]
    args.output.write_text("\n".join(report), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
