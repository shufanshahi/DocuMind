#!/usr/bin/env python3
"""Phase 5 deliverable: the *same* retrieve -> prompt -> generate pipeline
as Phase 4, rebuilt on LlamaIndex's abstractions instead of LangChain's --
a `VectorStoreIndex` wrapping one of Phase 2's existing Chroma collections,
a retriever from `index.as_retriever()`, and a custom `QueryEngine`
(LlamaIndex's own extension point for "I want the QueryEngine interface,
full control over what happens inside it") reusing the same shared prompt
template and token-budget assembly Phase 4 uses.

See explaination/phase5_comparison.md for how this compared to Phase 4's
LangChain implementation to actually build.

Usage:
    python rag_llamaindex.py "why does chunk size matter for retrieval quality"
    python rag_llamaindex.py "..." --strategy semantic --embedder sentence_transformers
    python rag_llamaindex.py --compare-chunking     # LlamaIndex's SentenceSplitter vs. our recursive chunker
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex  # noqa: E402
from llama_index.core.base.llms.types import ChatMessage, MessageRole  # noqa: E402
from llama_index.core.base.response.schema import Response  # noqa: E402
from llama_index.core.node_parser import SentenceSplitter  # noqa: E402
from llama_index.core.query_engine import CustomQueryEngine  # noqa: E402
from llama_index.core.retrievers import BaseRetriever  # noqa: E402
from llama_index.core.schema import NodeWithScore  # noqa: E402
from llama_index.llms.ollama import Ollama  # noqa: E402
from llama_index.vector_stores.chroma import ChromaVectorStore as LlamaChromaVectorStore  # noqa: E402

from embeddings.base import Embedder  # noqa: E402
from embeddings.llamaindex_adapter import LlamaIndexEmbedderAdapter  # noqa: E402
from embeddings.nomic_embedder import NomicEmbedder  # noqa: E402
from embeddings.st_embedder import SentenceTransformerEmbedder  # noqa: E402
from ingest.chunkers import STRATEGIES, Chunk  # noqa: E402
from prompts.templates import SYSTEM_PROMPT, SYSTEM_PROMPT_BASELINE, build_few_shot_messages  # noqa: E402
from rag.context_budget import TokenCounter, assemble_context  # noqa: E402
from retrieval.vector_store import ChromaVectorStore  # noqa: E402

EMBEDDER_FACTORIES = {
    "sentence_transformers": SentenceTransformerEmbedder,
    "nomic_embed_text": NomicEmbedder,
}

_CHUNK_TOP_LEVEL_FIELDS = {f.name for f in fields(Chunk)} - {"metadata"}


def _chunk_from_node(node_with_score: NodeWithScore, fallback_strategy: str) -> Chunk:
    """The LlamaIndex-side equivalent of Phase 3's `_chunk_from_stored`:
    rebuild a full `Chunk` from a `TextNode`'s text + metadata, so the rest
    of the pipeline (citation formatting, token budgeting) is completely
    framework-agnostic and shared with Phase 4."""
    node = node_with_score.node
    m = node.metadata
    extra_metadata = {k: v for k, v in m.items() if k not in _CHUNK_TOP_LEVEL_FIELDS}
    return Chunk(
        chunk_id=node.node_id,
        doc_id=m.get("doc_id", ""),
        source_path=m.get("source_path", ""),
        filetype=m.get("filetype", ""),
        strategy=m.get("strategy", fallback_strategy),
        chunk_index=m.get("chunk_index", 0),
        text=node.get_content(),
        char_start=m.get("char_start", 0),
        char_end=m.get("char_end", 0),
        page_start=m.get("page_start"),
        page_end=m.get("page_end"),
        metadata=extra_metadata,
    )


class DocuMindQueryEngine(CustomQueryEngine):
    """A LlamaIndex `QueryEngine` -- the abstraction the project plan names
    explicitly -- implemented via `CustomQueryEngine`, the documented
    extension point for wanting the `.query(str) -> Response` contract
    while controlling exactly what happens between retrieval and
    generation. Reuses `assemble_context` (Phase 4) and `SYSTEM_PROMPT`
    (shared with LangChain) rather than LlamaIndex's own response
    synthesizer, so token budgeting and citation formatting are identical
    across both framework implementations -- the two are compared fairly,
    on the same prompt and the same truncation behavior, differing only in
    how retrieval and generation are orchestrated.
    """

    retriever: BaseRetriever
    llm: object  # LlamaIndex's LLM base class; `object` keeps this test-double-friendly
    strategy: str
    k: int = 5
    token_budget: int = 2000
    counter: object = None
    system_prompt: str = SYSTEM_PROMPT
    use_few_shot: bool = True

    _ROLE_MAP = {"user": MessageRole.USER, "assistant": MessageRole.ASSISTANT}

    def custom_query(self, query_str: str) -> Response:
        nodes = self.retriever.retrieve(query_str)[: self.k]
        chunks = [_chunk_from_node(n, self.strategy) for n in nodes]
        assembled = assemble_context(chunks, self.token_budget, self.counter or TokenCounter())

        if not assembled.used_chunks:
            return Response(
                response="I don't have enough information to answer that. (No retrieved context fit within the token budget.)",
                source_nodes=nodes,
                metadata={"used_chunks": [], "dropped_chunks": assembled.dropped_chunks, "context_tokens": 0},
            )

        messages = [ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt)]
        if self.use_few_shot:
            # build_few_shot_messages() returns generic ("user"/"assistant", text)
            # tuples so this same data drives both frameworks' message lists --
            # see rag_langchain.py's build_prompt() for the LangChain side.
            messages.extend(ChatMessage(role=self._ROLE_MAP[role], content=content) for role, content in build_few_shot_messages())
        messages.append(
            ChatMessage(role=MessageRole.USER, content=f"CONTEXT:\n{assembled.text}\n\nQUESTION:\n{query_str}\n\nANSWER (with citations):")
        )
        answer = self.llm.chat(messages).message.content

        return Response(
            response=answer,
            source_nodes=nodes,
            metadata={
                "used_chunks": assembled.used_chunks,
                "dropped_chunks": assembled.dropped_chunks,
                "context_tokens": assembled.total_tokens,
            },
        )


def build_query_engine(
    strategy: str,
    embedder: Embedder,
    k: int = 5,
    token_budget: int = 2000,
    llm_model: str = "llama3.1:8b",
    persist_dir: str | Path = "data/chroma_db",
    system_prompt: str = SYSTEM_PROMPT,
    use_few_shot: bool = True,
) -> DocuMindQueryEngine:
    """Point LlamaIndex's `VectorStoreIndex` at a Chroma collection Phase
    2's `embed.py` already populated, rather than re-ingesting documents
    through LlamaIndex's own pipeline -- proving the vector store built in
    Phase 2 is genuinely reusable across frameworks, per the project
    plan's §4 "build swappable modules" design decision."""
    store = ChromaVectorStore(persist_dir=persist_dir)
    collection_name = f"{strategy}__{embedder.name}"
    llama_vector_store = LlamaChromaVectorStore(chroma_collection=store.raw_collection(collection_name))
    embed_model = LlamaIndexEmbedderAdapter(embedder)
    index = VectorStoreIndex.from_vector_store(vector_store=llama_vector_store, embed_model=embed_model)

    retriever = index.as_retriever(similarity_top_k=k)
    # context_window must be pinned explicitly. LlamaIndex's Ollama wrapper
    # defaults to -1, which it resolves to the *model's* advertised max
    # context (131072 for llama3.1) and sends that as `num_ctx` on every
    # request -- but if the local `ollama serve` process was started with a
    # smaller hard cap (this machine's is `-c 4096`), asking for a KV cache
    # far beyond that crashes the request mid-flight (the client sees
    # "server disconnected without sending a response", not a clean error).
    # A generous request_timeout and explicit keep_alive also matter: Ollama
    # unloads an idle model after a few minutes, and reloading a multi-GB
    # model from disk on this machine has been observed to occasionally take
    # well past LlamaIndex's 30s default timeout -- a slow cold start, not a
    # hung server.
    llm = Ollama(model=llm_model, request_timeout=180.0, keep_alive="30m", context_window=4096)
    return DocuMindQueryEngine(
        retriever=retriever,
        llm=llm,
        strategy=strategy,
        k=k,
        token_budget=token_budget,
        counter=TokenCounter(),
        system_prompt=system_prompt,
        use_few_shot=use_few_shot,
    )


@dataclass
class RagAnswer:
    question: str
    answer: str
    used_chunks: list[Chunk]
    dropped_chunks: list[Chunk]
    context_tokens: int


def ask(query_engine: DocuMindQueryEngine, question: str) -> RagAnswer:
    response = query_engine.query(question)
    return RagAnswer(
        question=question,
        answer=response.response,
        used_chunks=response.metadata["used_chunks"],
        dropped_chunks=response.metadata["dropped_chunks"],
        context_tokens=response.metadata["context_tokens"],
    )


def compare_chunking(input_dir: Path = Path("data/raw"), processed_dir: Path = Path("data/processed")) -> None:
    """The plan's other Phase 5 ask: 'try LlamaIndex's built-in node
    parsers and compare to your hand-rolled chunkers.' Runs LlamaIndex's
    `SentenceSplitter` (its recursive-ish default splitter) over the same
    raw corpus with the same size/overlap Phase 1's `recursive` strategy
    used, and prints both next to each other."""
    documents = SimpleDirectoryReader(input_dir=str(input_dir)).load_data()
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    li_count = len(nodes)
    li_avg = sum(len(n.get_content()) for n in nodes) / li_count if li_count else 0

    recursive_path = processed_dir / "recursive.jsonl"
    our_count = our_avg = None
    if recursive_path.exists():
        our_chunks = [json.loads(line) for line in recursive_path.open(encoding="utf-8")]
        our_count = len(our_chunks)
        our_avg = sum(len(c["text"]) for c in our_chunks) / our_count if our_count else 0

    print(f"{'':<45}{'chunks':>10}{'avg chars/chunk':>20}")
    print(f"{'LlamaIndex SentenceSplitter':<45}{li_count:>10}{li_avg:>20.1f}")
    if our_count is not None:
        print(f"{'our recursive_chunk (Phase 1)':<45}{our_count:>10}{our_avg:>20.1f}")
    else:
        print(f"{'our recursive_chunk (Phase 1)':<45}{'n/a':>10}  (run ingest.py first)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--strategy", choices=STRATEGIES, default="recursive")
    parser.add_argument("--embedder", choices=list(EMBEDDER_FACTORIES), default="nomic_embed_text")
    parser.add_argument("--llm-model", default="llama3.1:8b")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--token-budget", type=int, default=2000)
    parser.add_argument("--compare-chunking", action="store_true", help="compare LlamaIndex's SentenceSplitter to our recursive chunker and exit")
    parser.add_argument(
        "--prompt-variant",
        choices=["guarded", "baseline"],
        default="guarded",
        help="'guarded' (default): Phase 6's refined system prompt + few-shot examples. 'baseline': the unmodified plan §8 prompt, for comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.compare_chunking:
        compare_chunking()
        return

    if not args.question:
        print("error: a question is required unless --compare-chunking is given", file=sys.stderr)
        sys.exit(1)

    embedder = EMBEDDER_FACTORIES[args.embedder]()
    system_prompt = SYSTEM_PROMPT if args.prompt_variant == "guarded" else SYSTEM_PROMPT_BASELINE
    query_engine = build_query_engine(
        strategy=args.strategy,
        embedder=embedder,
        k=args.k,
        token_budget=args.token_budget,
        llm_model=args.llm_model,
        system_prompt=system_prompt,
        use_few_shot=(args.prompt_variant == "guarded"),
    )
    result = ask(query_engine, args.question)

    print(f"Question: {result.question}\n")
    print(f"Answer:\n{result.answer}\n")
    print(f"Context: {len(result.used_chunks)} chunk(s) used ({result.context_tokens} tokens), {len(result.dropped_chunks)} dropped by budget")
    for i, chunk in enumerate(result.used_chunks, 1):
        page = f", p.{chunk.page_start}" if chunk.page_start else ""
        print(f"  [source: {i}] {chunk.source_path}{page}")


if __name__ == "__main__":
    main()
