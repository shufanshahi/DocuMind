"""Prompt templates shared by both RAG implementations (LangChain,
LlamaIndex). `SYSTEM_PROMPT_BASELINE` is the starting-point prompt from the
project plan (§8), used unmodified through Phase 4/5. `SYSTEM_PROMPT` is
Phase 6's refined version, informed by real failure modes observed while
testing the baseline (see explaination/phase6_theory.md) — it's now the
default both `rag_langchain.py` and `rag_llamaindex.py` build with, with
the baseline kept importable for the head-to-head comparison
`guardrail_experiments.py` runs.
"""

SYSTEM_PROMPT_BASELINE = (
    "You are a careful research assistant. Answer the user's question using ONLY "
    "the context provided below. If the context does not contain enough "
    "information to answer confidently, say \"I don't have enough information "
    "to answer that.\" Do not use outside knowledge. When you use a piece of "
    "context, cite it using its [source: N] tag."
)

SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer the user's question using ONLY "
    "the context provided below.\n\n"
    "Rules:\n"
    "- If the context does not contain enough information to answer confidently, "
    "say exactly: \"I don't have enough information to answer that.\" Do not guess, "
    "and do not fill gaps with outside knowledge.\n"
    "- Never state a specific fact, number, name, or date unless it appears in the "
    "context. If the context only partially answers the question, answer the part "
    "it supports and say what is missing.\n"
    "- Every claim must cite the [source: N] tag it came from. If you combine facts "
    "from more than one source, cite each one used.\n"
    "- Do not invent a [source: N] tag that was not given to you in the context.\n\n"
    "Some examples of how to handle tricky questions follow."
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


# Three tricky cases named explicitly in the project plan's Phase 6 task
# list: a multi-hop question (the answer requires combining two source
# chunks), an ambiguous/partially-answerable question (the context is
# related but doesn't state what's asked, the case most likely to tempt a
# model into guessing a plausible-sounding specific), and a fully
# out-of-corpus adversarial question. Each is written as a genuine
# CONTEXT/QUESTION/ANSWER turn using the same HUMAN_TEMPLATE shape real
# turns use, so the model sees one consistent format throughout the prompt.
FEW_SHOT_EXAMPLES = [
    {
        "context_chunks": [
            "sentence-transformers runs entirely on your own machine, has no "
            "per-call cost, and supports many pretrained models of different "
            "sizes and quality.",
            "Nomic-embed-text is available for local use through Ollama and "
            "was trained specifically to perform well on retrieval tasks.",
        ],
        "question": "Which local embedding option is Ollama-based, and what is it optimized for?",
        "answer": (
            "Nomic-embed-text is the Ollama-based option [source: 2], and it was "
            "trained specifically to perform well on retrieval tasks [source: 2]. "
            "sentence-transformers, by contrast, runs directly on your own machine "
            "rather than through Ollama [source: 1]."
        ),
    },
    {
        "context_chunks": [
            "Fixed-size splitting divides a document into pieces of a constant "
            "character or token length, usually with a small overlap between "
            "consecutive pieces.",
        ],
        "question": "What chunk size should I use for my documents?",
        "answer": (
            "I don't have enough information to answer that. The context explains "
            "what fixed-size splitting is [source: 1], but it does not state a "
            "recommended or optimal chunk size."
        ),
    },
    {
        "context_chunks": [
            "Retrieval-Augmented Generation combines a retrieval system with a "
            "generative language model.",
        ],
        "question": "What is the boiling point of mercury?",
        "answer": "I don't have enough information to answer that.",
    },
]


def build_few_shot_messages() -> list[tuple[str, str]]:
    """Flatten `FEW_SHOT_EXAMPLES` into (role, content) turns ready to
    splice between the system message and the real question — `"user"` /
    `"assistant"` role labels work directly in LangChain's
    `ChatPromptTemplate.from_messages` and map onto LlamaIndex's
    `MessageRole.USER` / `MessageRole.ASSISTANT` in `rag_llamaindex.py`.
    """
    messages: list[tuple[str, str]] = []
    for example in FEW_SHOT_EXAMPLES:
        context = "\n\n".join(
            format_chunk_for_citation(i, text) for i, text in enumerate(example["context_chunks"], start=1)
        )
        human = HUMAN_TEMPLATE.format(context=context, question=example["question"])
        messages.append(("user", human))
        messages.append(("assistant", example["answer"]))
    return messages
