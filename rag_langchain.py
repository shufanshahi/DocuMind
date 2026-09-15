#!/usr/bin/env python3
"""Phase 4 deliverable: the retrieve -> prompt -> generate RAG pipeline,
built with LangChain's LCEL (LangChain Expression Language), on top of
Phase 3's `Retriever` and a token-budget-aware context assembly step.

Usage:
    python rag_langchain.py "why does chunk size matter for retrieval quality"
    python rag_langchain.py "..." --strategy semantic --embedder sentence_transformers
    python rag_langchain.py "..." --hybrid --rerank --k 3 --token-budget 1000
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_core.runnables import RunnableLambda, RunnablePassthrough  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402

from embeddings.nomic_embedder import NomicEmbedder  # noqa: E402
from embeddings.st_embedder import SentenceTransformerEmbedder  # noqa: E402
from ingest.chunkers import STRATEGIES, Chunk  # noqa: E402
from prompts.templates import HUMAN_TEMPLATE, SYSTEM_PROMPT, SYSTEM_PROMPT_BASELINE, build_few_shot_messages  # noqa: E402
from rag.context_budget import TokenCounter, assemble_context  # noqa: E402
from retrieval.retriever import Retriever  # noqa: E402

EMBEDDER_FACTORIES = {
    "sentence_transformers": SentenceTransformerEmbedder,
    "nomic_embed_text": NomicEmbedder,
}


def build_prompt(system_prompt: str = SYSTEM_PROMPT, use_few_shot: bool = True) -> ChatPromptTemplate:
    """Phase 6: the prompt is no longer a fixed module-level constant --
    `guardrail_experiments.py` needs to build both the baseline and the
    guarded+few-shot variant side by side to measure whether the guardrail
    changes actually help (see explaination/phase6_theory.md), so prompt
    construction became a function of which variant is being tested.
    """
    messages = [("system", system_prompt)]
    if use_few_shot:
        messages.extend(build_few_shot_messages())
    messages.append(("human", HUMAN_TEMPLATE))
    return ChatPromptTemplate.from_messages(messages)


@dataclass
class RagAnswer:
    question: str
    answer: str
    used_chunks: list[Chunk]
    dropped_chunks: list[Chunk]
    context_tokens: int


def build_chain(
    retriever: Retriever,
    llm,
    k: int = 5,
    token_budget: int = 2000,
    counter: TokenCounter | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    use_few_shot: bool = True,
):
    """Assemble the LCEL chain: retrieve -> budget context -> prompt -> generate.

    Structured as two composed Runnables rather than one big function so
    each step is independently swappable/inspectable — the same reason the
    project plan calls out LCEL specifically rather than one hand-rolled
    Python function that happens to call an LLM. `RunnablePassthrough.assign`
    keeps the retrieval bookkeeping (which chunks were used/dropped, token
    count) flowing alongside the generated answer instead of throwing it
    away, so a caller can still show citations after the chain runs.
    """
    counter = counter or TokenCounter()
    prompt = build_prompt(system_prompt, use_few_shot)
    generate = prompt | llm | StrOutputParser()

    def retrieve_and_assemble(question: str) -> dict:
        chunks = retriever.retrieve(question, k=k)
        assembled = assemble_context(chunks, token_budget, counter)
        return {
            "question": question,
            "context": assembled.text,
            "_used_chunks": assembled.used_chunks,
            "_dropped_chunks": assembled.dropped_chunks,
            "_context_tokens": assembled.total_tokens,
        }

    def generate_or_refuse(input_dict: dict) -> str:
        # An empty context is the most extreme case of "insufficient
        # information" the system prompt is supposed to guard against --
        # but an instruction is just a suggestion the model can ignore, and
        # it does: tested against a token budget too small to fit any
        # chunk, the model confidently hallucinated a fake bibliography
        # instead of refusing (see explaination/phase4_theory.md). Refusing
        # here in code, before the LLM call even happens, is a real
        # guarantee instead of a hopeful instruction, and it saves a wasted
        # generation call.
        if not input_dict["_used_chunks"]:
            return "I don't have enough information to answer that. (No retrieved context fit within the token budget.)"
        return generate.invoke(input_dict)

    return RunnableLambda(retrieve_and_assemble) | RunnablePassthrough.assign(answer=RunnableLambda(generate_or_refuse))


def ask(chain, question: str) -> RagAnswer:
    result = chain.invoke(question)
    return RagAnswer(
        question=question,
        answer=result["answer"],
        used_chunks=result["_used_chunks"],
        dropped_chunks=result["_dropped_chunks"],
        context_tokens=result["_context_tokens"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--strategy", choices=STRATEGIES, default="recursive")
    parser.add_argument("--embedder", choices=list(EMBEDDER_FACTORIES), default="nomic_embed_text")
    parser.add_argument("--llm-model", default="llama3.1:8b")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--token-budget", type=int, default=2000)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument(
        "--prompt-variant",
        choices=["guarded", "baseline"],
        default="guarded",
        help="'guarded' (default): Phase 6's refined system prompt + few-shot examples. 'baseline': the unmodified plan §8 prompt, for comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedder = EMBEDDER_FACTORIES[args.embedder]()
    retriever = Retriever(strategy=args.strategy, embedder=embedder, use_hybrid=args.hybrid, use_reranker=args.rerank)
    llm = ChatOllama(model=args.llm_model, temperature=0)
    system_prompt = SYSTEM_PROMPT if args.prompt_variant == "guarded" else SYSTEM_PROMPT_BASELINE
    chain = build_chain(
        retriever, llm, k=args.k, token_budget=args.token_budget, system_prompt=system_prompt, use_few_shot=(args.prompt_variant == "guarded")
    )

    result = ask(chain, args.question)

    print(f"Question: {result.question}\n")
    print(f"Answer:\n{result.answer}\n")
    print(f"Context: {len(result.used_chunks)} chunk(s) used ({result.context_tokens} tokens), {len(result.dropped_chunks)} dropped by budget")
    for i, chunk in enumerate(result.used_chunks, 1):
        page = f", p.{chunk.page_start}" if chunk.page_start else ""
        print(f"  [source: {i}] {chunk.source_path}{page}")


if __name__ == "__main__":
    main()
