"""Phase 8 deliverable: wrap the RAG pipeline in a FastAPI endpoint.

Every previous phase's CLI (retrieve.py, rag_langchain.py,
guardrail_experiments.py, run_eval.py) exposed the same configurability
knobs -- strategy, embedder, k, token budget, hybrid, rerank, prompt
variant -- as command-line flags. This wraps the exact same pipeline
(`Retriever` + `rag_langchain.build_chain`/`ask`) behind one HTTP
endpoint with the same knobs as request fields, so a frontend (Phase 8's
Streamlit app) or any other client can drive it without a Python import.

Run with:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # for `import rag_langchain`
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from embeddings.base import Embedder  # noqa: E402
from embeddings.nomic_embedder import NomicEmbedder  # noqa: E402
from embeddings.st_embedder import SentenceTransformerEmbedder  # noqa: E402
from ingest.chunkers import STRATEGIES  # noqa: E402
from prompts.templates import SYSTEM_PROMPT, SYSTEM_PROMPT_BASELINE  # noqa: E402
from rag_langchain import ask, build_chain  # noqa: E402
from retrieval.retriever import Retriever  # noqa: E402

EMBEDDER_FACTORIES = {
    "sentence_transformers": SentenceTransformerEmbedder,
    "nomic_embed_text": NomicEmbedder,
}

# Defaults: the (chunking strategy, embedder) combination Phase 7's
# evaluation (see ../eval_results.md) found to perform best on retrieval
# precision/recall and generation faithfulness/relevancy. A caller can
# still override either per request for experimentation.
DEFAULT_STRATEGY = "recursive"
DEFAULT_EMBEDDER = "nomic_embed_text"

app = FastAPI(
    title="DocuMind",
    description="RAG-based document Q&A over a local corpus -- retrieval, guardrails, and citations, all running locally.",
)

# Built once, reused across requests: constructing an embedder can mean
# loading a model (sentence-transformers) or probing Ollama (nomic), and a
# Retriever with hybrid search on tokenizes an entire chunking strategy's
# corpus for BM25 -- none of that should happen on every single request.
_llm = ChatOllama(model="llama3.1:8b", temperature=0)
_embedder_cache: dict[str, Embedder] = {}
_retriever_cache: dict[tuple, Retriever] = {}


def get_embedder(name: str) -> Embedder:
    if name not in EMBEDDER_FACTORIES:
        raise HTTPException(status_code=400, detail=f"unknown embedder {name!r}; choose from {list(EMBEDDER_FACTORIES)}")
    if name not in _embedder_cache:
        _embedder_cache[name] = EMBEDDER_FACTORIES[name]()
    return _embedder_cache[name]


def get_retriever(strategy: str, embedder_name: str, hybrid: bool, rerank: bool) -> Retriever:
    if strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail=f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    key = (strategy, embedder_name, hybrid, rerank)
    if key not in _retriever_cache:
        embedder = get_embedder(embedder_name)
        _retriever_cache[key] = Retriever(strategy=strategy, embedder=embedder, use_hybrid=hybrid, use_reranker=rerank)
    return _retriever_cache[key]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    strategy: str = DEFAULT_STRATEGY
    embedder: str = DEFAULT_EMBEDDER
    k: int = Field(5, ge=1, le=20)
    token_budget: int = Field(2000, ge=0)
    hybrid: bool = False
    rerank: bool = False
    prompt_variant: str = Field("guarded", pattern="^(guarded|baseline)$")


class SourceChunk(BaseModel):
    citation_index: int
    source_path: str
    page_start: int | None = None
    page_end: int | None = None
    text_preview: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    context_tokens: int
    chunks_used: int
    chunks_dropped: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    retriever = get_retriever(request.strategy, request.embedder, request.hybrid, request.rerank)
    system_prompt = SYSTEM_PROMPT if request.prompt_variant == "guarded" else SYSTEM_PROMPT_BASELINE
    chain = build_chain(
        retriever,
        _llm,
        k=request.k,
        token_budget=request.token_budget,
        system_prompt=system_prompt,
        use_few_shot=(request.prompt_variant == "guarded"),
    )
    result = ask(chain, request.question)

    sources = [
        SourceChunk(
            citation_index=i,
            source_path=chunk.source_path,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            text_preview=" ".join(chunk.text.split())[:200],
        )
        for i, chunk in enumerate(result.used_chunks, 1)
    ]
    return AskResponse(
        question=result.question,
        answer=result.answer,
        sources=sources,
        context_tokens=result.context_tokens,
        chunks_used=len(result.used_chunks),
        chunks_dropped=len(result.dropped_chunks),
    )
