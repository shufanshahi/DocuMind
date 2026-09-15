"""The starting-point prompt from the project plan (§8). Guardrail
refinement and few-shot examples for tricky cases are Phase 6's job, not
this one — Phase 4/5 just need *a* working, citation-forcing prompt to
build the retrieve-then-generate chain around.
"""

SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer the user's question using ONLY "
    "the context provided below. If the context does not contain enough "
    "information to answer confidently, say \"I don't have enough information "
    "to answer that.\" Do not use outside knowledge. When you use a piece of "
    "context, cite it using its [source: N] tag."
)

HUMAN_TEMPLATE = "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER (with citations):"


def format_chunk_for_citation(index: int, text: str) -> str:
    """Tag one chunk's text with the numbered `[source: N]` marker the
    system prompt instructs the model to cite by. `index` is 1-based and
    only meaningful within *this one prompt's* context block — see
    `rag_langchain.py` for how a citation number gets resolved back to a
    real `Chunk` (and its source file) after generation.
    """
    return f"[source: {index}] {text}"
