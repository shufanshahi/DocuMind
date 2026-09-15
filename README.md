# DocuMind — An AI-Powered Knowledge Assistant

A retrieval-augmented generation (RAG) system for asking natural-language
questions over a local document corpus and getting back accurate, cited
answers grounded in the source material — instead of an LLM hallucinating
from parametric memory alone.

Built as a hands-on, phase-by-phase project covering embedding models,
chunking strategies, dense/hybrid retrieval, two orchestration frameworks
(LangChain and LlamaIndex), prompt-level guardrails, quantitative
evaluation, and a FastAPI + Streamlit demo. Every model runs **locally** —
Ollama for generation and one embedding option, `sentence-transformers`
in-process for the other. No API key is required to run anything in this
repo as built.

See [RAG_Project_Plan.md](RAG_Project_Plan.md) for the original project
plan this was built against, and [explaination/](explaination) for a
theory + implementation writeup of every phase.

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │            INGESTION PIPELINE             │
  Raw Documents ───▶│  Loader → Chunker → Embedder             │───▶ Vector Store
  (PDF/MD/TXT)       │                                            │      (Chroma)
                    └─────────────────────────────────────────┘

                    ┌─────────────────────────────────────────┐
                    │              QUERY PIPELINE                │
  User Question ───▶│ Query Embed → Vector Search → Re-rank    │
                    │      → Context Assembly (token budget)    │
                    │      → Prompt Template → LLM Generation   │───▶ Answer +
                    └─────────────────────────────────────────┘      Citations
```

Ingestion and querying are separate, swappable modules throughout —
chunking strategy, embedding model, orchestration framework (LangChain vs.
LlamaIndex), and prompt variant can each be swapped independently, and
every stage is covered by its own CLI script for standalone testing.

## What's built

| Phase | What it does | Entry point |
|---|---|---|
| 1. Ingestion & chunking | Loaders for PDF/Markdown/text; 4 chunking strategies (fixed-size, recursive, semantic, sentence-window) | `ingest.py` |
| 2. Embeddings & vector store | Embeds chunks with `sentence-transformers` and `nomic-embed-text`; stores both in Chroma | `embed.py` |
| 3. Retrieval | Dense top-k search, BM25 hybrid search, cross-encoder re-ranking | `retrieve.py` |
| 4. RAG pipeline (LangChain) | Retrieve → budget → prompt → generate, built with LCEL | `rag_langchain.py` |
| 5. RAG pipeline (LlamaIndex) | The same pipeline rebuilt on a `CustomQueryEngine` over the same Chroma collections | `rag_llamaindex.py` |
| 6. Prompt engineering & guardrails | A refined system prompt + few-shot examples; experiments on refusal calibration, context ordering, and token budgets | `guardrail_experiments.py` |
| 7. Evaluation | A 20-question labeled eval set; retrieval precision/recall and an LLM-judge rubric for faithfulness/relevancy | `run_eval.py` → [eval_results.md](eval_results.md) |
| 8. API & UI | A FastAPI `/ask` endpoint and a Streamlit chat frontend showing citations | `api/main.py`, `app/streamlit_app.py` |

Each phase has a matching theory + code writeup under
[explaination/](explaination) (`phaseN_theory.md` / `phaseN_code.md`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install [Ollama](https://ollama.com/) and pull the two local models this
project uses:

```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

`sentence-transformers` (the other embedding option) downloads its model
automatically on first use — no separate pull needed.

Copy `.env.example` to `.env` if you want to point at a non-default Ollama
host/port, or swap in one of the hosted API alternatives named in the
project plan's tech stack table (not required for the default local setup).

## Running the pipeline end to end

```bash
# 1. Chunk the sample corpus in data/raw/ (all 4 strategies)
python ingest.py

# 2. Embed every chunking strategy's output with both embedders,
#    populating 8 Chroma collections (data/chroma_db/)
python embed.py

# 3. Try retrieval directly
python retrieve.py "why does chunk size matter for retrieval quality"
python retrieve.py "..." --hybrid --rerank --strategy semantic

# 4. Ask a question through either RAG implementation
python rag_langchain.py "why does chunk size matter for retrieval quality"
python rag_llamaindex.py "..." --strategy recursive --embedder nomic_embed_text

# 5. Run the guardrail experiments (refusal calibration, context ordering, token budget)
python guardrail_experiments.py

# 6. Run the full evaluation (writes eval_results.md)
python run_eval.py
python run_eval.py --retrieval-only   # skip the slower LLM-judge pass

# 7. Serve the API, then the chat UI in a second terminal
uvicorn api.main:app --reload
streamlit run app/streamlit_app.py
```

Swap your own documents into `data/raw/` (PDF, Markdown, or plain text)
and re-run steps 1-2 to use this on a different corpus.

## Project structure

```
DocuMind/
├── data/
│   ├── raw/                       # source documents
│   ├── processed/                 # chunked output, one .jsonl per strategy
│   ├── eval/eval_set.jsonl        # the Phase 7 labeled eval set
│   └── chroma_db/                 # persisted vector store (generated, gitignored)
├── src/
│   ├── ingest/                    # loaders, 4 chunking strategies
│   ├── embeddings/                # sentence-transformers, nomic-embed-text, LlamaIndex adapter
│   ├── retrieval/                 # Chroma wrapper, hybrid search, re-ranker, Retriever
│   ├── rag/                       # token-budget-aware context assembly
│   ├── prompts/                   # shared system prompt + few-shot examples
│   └── eval/                      # eval set loader, retrieval metrics, LLM-judge rubric
├── api/main.py                    # FastAPI app
├── app/streamlit_app.py           # Streamlit chat frontend
├── ingest.py / embed.py / retrieve.py
├── rag_langchain.py / rag_llamaindex.py
├── guardrail_experiments.py / run_eval.py
├── eval_results.md                # Phase 7 deliverable: results + conclusions
├── explaination/                  # theory + code writeup per phase
├── tests/                         # pytest suite
├── requirements.txt
└── .env.example
```

## Testing

```bash
pytest tests/
```

Everything that doesn't require a live model or network call is tested
with fakes (a fake embedder, retriever, LLM, or cross-encoder standing in
for the real thing) — the suite runs in a few seconds without Ollama
running. Tests that exercise real models or a real Chroma collection are
called out explicitly in each phase's `phaseN_code.md`.

## Documentation

- [RAG_Project_Plan.md](RAG_Project_Plan.md) — the original project brief.
- [explaination/](explaination) — a `phaseN_theory.md` (why) and
  `phaseN_code.md` (how) per phase, plus [phase5_comparison.md](explaination/phase5_comparison.md)
  (LangChain vs. LlamaIndex, built firsthand) and
  [phase6_results.md](explaination/phase6_results.md) (guardrail experiment
  transcripts and conclusions).
- [eval_results.md](eval_results.md) — Phase 7's measured results across
  chunking strategies and embedding models.

## Tech stack

| Layer | Choice |
|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) + `nomic-embed-text` via Ollama |
| Orchestration | LangChain (LCEL) and LlamaIndex (`CustomQueryEngine`) |
| Vector store | ChromaDB |
| Generation | `llama3.1:8b` via Ollama |
| Retrieval extras | BM25 hybrid search, `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranking |
| Evaluation | Custom retrieval metrics + a custom LLM-as-judge rubric |
| API / UI | FastAPI + Streamlit |
