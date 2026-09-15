"""Token-budget-aware context assembly: given a list of already-ranked
`Chunk`s (most relevant first, straight out of Phase 3's
`Retriever.retrieve()`), pack as many as fit under a token budget before
they ever reach the LLM.

Why this matters: an LLM's context window is finite, and "just concatenate
every retrieved chunk" works fine right up until retrieval returns enough
chunks (or big enough ones) to exceed it — then a call either errors out
or, on providers that silently truncate, drops content from one end
without telling you. Budgeting explicitly, and stopping deliberately once
the budget is hit, makes truncation a designed behavior instead of an
accident. See explaination/phase4_theory.md for the fuller argument, and
§8 of the project plan for what to experiment with here in Phase 6
(context ordering, what happens right at the budget edge).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ingest.chunkers import Chunk
from prompts.templates import format_chunk_for_citation

_CHARS_PER_TOKEN_ESTIMATE = 4  # rough, model-agnostic English-prose heuristic


class TokenCounter:
    """Counts tokens with tiktoken's `cl100k_base` encoding if available,
    falling back to a cheap chars/4 estimate otherwise. No model actually
    used in this project (llama3.1, nomic-embed-text) uses an OpenAI
    tokenizer, so this was never going to be an *exact* count for them —
    it's a consistent, good-enough, dependency-optional proxy for
    *budgeting*, not a claim about any specific model's real token count.
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        self._encoding = None
        try:
            import tiktoken

            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            self._encoding = None  # offline, or tiktoken not installed

    def count(self, text: str) -> int:
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        return max(1, len(text) // _CHARS_PER_TOKEN_ESTIMATE)


@dataclass
class AssembledContext:
    text: str
    used_chunks: list[Chunk] = field(default_factory=list)
    dropped_chunks: list[Chunk] = field(default_factory=list)
    total_tokens: int = 0


def assemble_context(chunks: list[Chunk], token_budget: int, counter: TokenCounter | None = None) -> AssembledContext:
    """Pack `chunks` (assumed already ordered by relevance, most relevant
    first) into `token_budget` tokens.

    Stops adding new chunks at the *first* one that would overflow the
    budget, rather than skipping it to bin-pack a smaller chunk from
    further down the ranking into its place: the retriever's ranking is
    trusted as the priority order, so once the budget runs out, everything
    remaining is dropped rather than reordered. Every chunk after that
    first overflow is still recorded in `dropped_chunks` (not just the one
    that triggered it), so a caller can report exactly how much of the
    retrieved evidence never made it into the prompt.
    """
    counter = counter or TokenCounter()
    used: list[Chunk] = []
    dropped: list[Chunk] = []
    parts: list[str] = []
    total = 0
    overflowed = False

    for i, chunk in enumerate(chunks, start=1):
        tagged = format_chunk_for_citation(i, chunk.text)
        tokens = counter.count(tagged)
        if overflowed or total + tokens > token_budget:
            overflowed = True
            dropped.append(chunk)
            continue
        used.append(chunk)
        parts.append(tagged)
        total += tokens

    return AssembledContext(text="\n\n".join(parts), used_chunks=used, dropped_chunks=dropped, total_tokens=total)
