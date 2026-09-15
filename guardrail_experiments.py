#!/usr/bin/env python3
"""Phase 6 deliverable: run the three experiments the project plan calls
out explicitly (§8) and report the results.

1. Refuse-to-answer calibration: a small set of answerable, multi-hop,
   ambiguous, and adversarial questions, run through both the baseline
   prompt (plan §8, unmodified) and Phase 6's guarded+few-shot prompt, to
   measure whether the guardrail changes actually reduce hallucination
   rather than just reading better.
2. Context ordering: does putting the most relevant chunk first or last
   in the assembled context change the answer, given a token budget tight
   enough that not everything fits?
3. Chunk budget: what actually happens to the assembled context (and the
   final answer) as the token budget shrinks from generous to zero?

Uses the LangChain pipeline (rag_langchain.py) as the test harness for all
three -- the guardrail logic and prompt are shared with the LlamaIndex
pipeline (Phase 4/5's whole point), so results here apply to both; running
every case through both frameworks would double the LLM calls for no
additional signal about prompt behavior specifically.

Usage:
    python guardrail_experiments.py                    # all three experiments
    python guardrail_experiments.py --experiment refusal
    python guardrail_experiments.py --experiment ordering
    python guardrail_experiments.py --experiment budget
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from langchain_ollama import ChatOllama  # noqa: E402

from embeddings.nomic_embedder import NomicEmbedder  # noqa: E402
from prompts.templates import SYSTEM_PROMPT, SYSTEM_PROMPT_BASELINE, format_chunk_for_citation  # noqa: E402
from rag.context_budget import TokenCounter, assemble_context  # noqa: E402
from rag_langchain import ask, build_chain  # noqa: E402
from retrieval.retriever import Retriever  # noqa: E402

REFUSAL_PHRASE = "don't have enough information"


def is_refusal(answer: str) -> bool:
    return REFUSAL_PHRASE in answer.lower()


def has_citation(answer: str) -> bool:
    return "[source:" in answer.lower()


@dataclass
class TestCase:
    question: str
    category: str  # "answerable" | "multi_hop" | "ambiguous" | "adversarial"
    expects: str  # "answer" | "refuse" | "manual" (ambiguous cases are judged by eye, not string-matched)


# Deliberately small and hand-picked rather than the 20-30 question eval
# set Phase 7 builds: this is a targeted guardrail smoke test, not a
# statistically powered evaluation. "answer" cases are only auto-scored on
# two objective signals (did it refuse when it shouldn't have, did it
# cite a source) -- whether the *content* is actually correct is exactly
# the kind of judgment Phase 7's LLM-as-judge rubric exists for, not
# something this script tries to fake with string matching.
TEST_CASES = [
    TestCase("why does chunk size matter for retrieval quality", "answerable", "answer"),
    TestCase("which local embedding option is served through Ollama, and what is it optimized for", "multi_hop", "answer"),
    TestCase("what is the boiling point of mercury", "adversarial", "refuse"),
    TestCase("does the sky ever turn green during a solar eclipse", "adversarial", "refuse"),
    TestCase("who won the 2018 world cup", "adversarial", "refuse"),
    TestCase("what programming language is documind written in", "adversarial", "refuse"),
    TestCase("what chunk size should I use for my own project", "ambiguous", "manual"),
]

PROMPT_VARIANTS = {
    "baseline": {"system_prompt": SYSTEM_PROMPT_BASELINE, "use_few_shot": False},
    "guarded": {"system_prompt": SYSTEM_PROMPT, "use_few_shot": True},
}


def run_refusal_experiment(strategy: str = "recursive", k: int = 5, token_budget: int = 2000) -> None:
    print("=" * 100)
    print("EXPERIMENT 1 — Refuse-to-answer calibration: baseline prompt vs. guarded+few-shot prompt")
    print("=" * 100)

    embedder = NomicEmbedder()
    retriever = Retriever(strategy=strategy, embedder=embedder)
    llm = ChatOllama(model="llama3.1:8b", temperature=0)

    scoreboard: dict[str, tuple[int, int]] = {}
    for variant_name, variant_kwargs in PROMPT_VARIANTS.items():
        print(f"\n--- prompt variant: {variant_name} ---")
        chain = build_chain(retriever, llm, k=k, token_budget=token_budget, **variant_kwargs)
        correct = scored = 0
        for case in TEST_CASES:
            result = ask(chain, case.question)
            refused, cited = is_refusal(result.answer), has_citation(result.answer)
            if case.expects == "refuse":
                ok, verdict = refused, "PASS" if refused else "FAIL (hallucinated instead of refusing)"
                scored += 1
                correct += int(ok)
            elif case.expects == "answer":
                ok = (not refused) and cited
                verdict = "PASS" if ok else "FAIL (refused, or answered without a citation)"
                scored += 1
                correct += int(ok)
            else:
                verdict = "(manual review — not auto-scored)"
            print(f"\n[{case.category:<11}] {verdict}")
            print(f"  Q: {case.question}")
            print(f"  A: {result.answer}")
        scoreboard[variant_name] = (correct, scored)

    print("\nSummary")
    for variant_name, (correct, scored) in scoreboard.items():
        print(f"  {variant_name:<10}: {correct}/{scored} objectively-scored cases correct")


def run_ordering_experiment(strategy: str = "recursive", k: int = 5, token_budget: int = 100) -> None:
    print("\n" + "=" * 100)
    print("EXPERIMENT 2 — Context ordering: most-relevant-first vs. most-relevant-last")
    print("=" * 100)

    question = "why does chunk size matter for retrieval quality"
    embedder = NomicEmbedder()
    retriever = Retriever(strategy=strategy, embedder=embedder)
    llm = ChatOllama(model="llama3.1:8b", temperature=0)
    counter = TokenCounter()

    chunks = retriever.retrieve(question, k=k)
    print(f"\nToken budget: {token_budget} (deliberately tight -- not all {len(chunks)} retrieved chunks fit)")
    print(f"{'rank':<6}{'tokens':<8}source")
    for i, c in enumerate(chunks, 1):
        print(f"{i:<6}{counter.count(format_chunk_for_citation(i, c.text)):<8}{c.source_path}")

    for label, ordered_chunks in [("most-relevant-first (retriever's own order)", chunks), ("most-relevant-last (reversed)", list(reversed(chunks)))]:
        assembled = assemble_context(ordered_chunks, token_budget, counter)
        used_sources = [c.source_path for c in assembled.used_chunks]
        print(f"\n--- {label} ---")
        print(f"  used: {used_sources}")
        chain = build_chain(retriever, llm, k=k, token_budget=token_budget, counter=counter)

        # Bypass the retriever inside the chain so both orderings are
        # measured against the *exact* same retrieved chunks, in only the
        # order under test -- otherwise a second retrieval call could
        # return a different top-k due to non-determinism upstream.
        original_retrieve = retriever.retrieve
        retriever.retrieve = lambda q, k=k, _oc=ordered_chunks: _oc  # noqa: E731
        try:
            result = ask(chain, question)
        finally:
            retriever.retrieve = original_retrieve
        print(f"  answer: {result.answer}")


def run_budget_experiment(strategy: str = "recursive", k: int = 5) -> None:
    print("\n" + "=" * 100)
    print("EXPERIMENT 3 — Chunk budget: sweeping the token budget from generous to zero")
    print("=" * 100)

    question = "why does chunk size matter for retrieval quality"
    embedder = NomicEmbedder()
    retriever = Retriever(strategy=strategy, embedder=embedder)
    counter = TokenCounter()
    chunks = retriever.retrieve(question, k=k)

    print(f"\n{'budget':<10}{'used':<8}{'dropped':<10}{'tokens used':<14}")
    for budget in (2000, 300, 150, 100, 50, 0):
        assembled = assemble_context(chunks, budget, counter)
        print(f"{budget:<10}{len(assembled.used_chunks):<8}{len(assembled.dropped_chunks):<10}{assembled.total_tokens:<14}")

    llm = ChatOllama(model="llama3.1:8b", temperature=0)
    for budget in (2000, 50):
        chain = build_chain(retriever, llm, k=k, token_budget=budget, counter=counter)
        result = ask(chain, question)
        print(f"\n--- budget={budget} ---")
        print(f"  {len(result.used_chunks)} used, {len(result.dropped_chunks)} dropped")
        print(f"  answer: {result.answer}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", choices=["refusal", "ordering", "budget", "all"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.experiment in ("refusal", "all"):
        run_refusal_experiment()
    if args.experiment in ("ordering", "all"):
        run_ordering_experiment()
    if args.experiment in ("budget", "all"):
        run_budget_experiment()


if __name__ == "__main__":
    main()
