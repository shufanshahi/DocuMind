"""Four chunking strategies for splitting a document's text into retrievable
units, plus the `Chunk` container that tags every unit with enough metadata
to trace it back to its source (file, char offsets, PDF page, strategy).

Strategies (see explaination/phase1_theory.md for the reasoning behind each):
  1. fixed_size        - split every N characters, with overlap
  2. recursive          - split on paragraph/sentence/word boundaries, recursively
  3. semantic           - split where meaning shifts between adjacent sentences
  4. sentence_window    - index single sentences, store the surrounding window
                          as context to expand into at retrieval time
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from .embedders import EmbedFn, cosine_similarity, get_embedder
from .loaders import Document

Span = tuple[str, int, int]  # (text, char_start, char_end)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_path: str
    filetype: str
    strategy: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Sentence splitting (shared by the semantic and sentence-window strategies)
# --------------------------------------------------------------------------

# Split after a sentence-ending punctuation mark that's followed by whitespace.
# Known limitation: doesn't special-case abbreviations ("Dr.", "e.g.") or
# decimal numbers, so those occasionally produce an extra split. Good enough
# for chunking purposes, where an over-eager sentence boundary just means a
# slightly smaller unit, not incorrect retrieval.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[Span]:
    sentences: list[Span] = []
    start = 0
    for m in _SENTENCE_SPLIT_RE.finditer(text):
        end = m.start()
        if text[start:end].strip():
            sentences.append((text[start:end], start, end))
        start = m.end()
    if start < len(text) and text[start:].strip():
        sentences.append((text[start:], start, len(text)))
    return sentences


# --------------------------------------------------------------------------
# 1. Fixed-size splitting
# --------------------------------------------------------------------------


def fixed_size_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> list[Span]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: list[Span] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append((text[start:end], start, end))
        if end == n:
            break
        start = end - overlap
    return chunks


# --------------------------------------------------------------------------
# 2. Recursive character splitting
# --------------------------------------------------------------------------

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _find_all(text: str, sep: str) -> list[Span]:
    """Split `text` on `sep`, keeping exact character offsets of each piece
    (the separator itself is dropped, same as str.split, but we remember
    where it was so the caller can slice the *original* text later)."""
    if sep == "":
        return [(ch, i, i + 1) for i, ch in enumerate(text)]
    pieces: list[Span] = []
    start = 0
    while True:
        idx = text.find(sep, start)
        if idx == -1:
            pieces.append((text[start:], start, len(text)))
            break
        pieces.append((text[start:idx], start, idx))
        start = idx + len(sep)
    return pieces


def _split_to_leaves(text: str, offset: int, separators: list[str], chunk_size: int) -> list[Span]:
    """Recursively split `text` until every piece is <= chunk_size, trying
    separators in order (paragraph, then line, then sentence, then word,
    then character as a last resort so we always terminate)."""
    if len(text) <= chunk_size or not separators:
        return [(text, offset, offset + len(text))]
    sep, *rest = separators
    leaves: list[Span] = []
    for piece, s, e in _find_all(text, sep):
        if not piece.strip() and sep != "":
            continue  # drop blank pieces produced by consecutive separators
        abs_s, abs_e = offset + s, offset + e
        if len(piece) > chunk_size:
            leaves.extend(_split_to_leaves(piece, abs_s, rest, chunk_size))
        else:
            leaves.append((piece, abs_s, abs_e))
    return leaves or [(text, offset, offset + len(text))]


def _merge_leaves(text: str, leaves: list[Span], chunk_size: int, overlap: int) -> list[Span]:
    """Greedily pack adjacent leaves back together up to chunk_size, carrying
    the last `overlap` characters' worth of leaves into the next chunk. Slicing
    the *original* text by offset (rather than joining piece strings) means
    separators that fell between leaves are naturally restored."""
    if not leaves:
        return []
    chunks: list[Span] = []
    n = len(leaves)
    start_idx = 0
    while start_idx < n:
        end_idx = start_idx
        while end_idx + 1 < n and (leaves[end_idx + 1][2] - leaves[start_idx][1]) <= chunk_size:
            end_idx += 1
        chunk_start, chunk_end = leaves[start_idx][1], leaves[end_idx][2]
        chunks.append((text[chunk_start:chunk_end], chunk_start, chunk_end))
        if end_idx == n - 1:
            break
        j = end_idx
        while j > start_idx and (chunk_end - leaves[j][1]) < overlap:
            j -= 1
        start_idx = max(j, start_idx + 1)  # always make forward progress
    return chunks


def recursive_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separators: list[str] | None = None,
) -> list[Span]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    leaves = _split_to_leaves(text, 0, separators or DEFAULT_SEPARATORS, chunk_size)
    return _merge_leaves(text, leaves, chunk_size, overlap)


# --------------------------------------------------------------------------
# 3. Semantic chunking
# --------------------------------------------------------------------------


def semantic_chunk(
    text: str,
    embed_fn: EmbedFn | None = None,
    similarity_threshold: float = 0.2,
    max_chunk_chars: int = 1500,
) -> list[Span]:
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [(text, 0, len(text))] if text.strip() else []

    embed_fn = embed_fn or get_embedder()
    vectors = embed_fn([s for s, _, _ in sentences])

    chunks: list[Span] = []
    group_start = 0  # index into `sentences` where the current chunk began
    for i in range(1, len(sentences)):
        sim = cosine_similarity(vectors[i - 1], vectors[i])
        current_len = sentences[i][2] - sentences[group_start][1]
        topic_shift = sim < similarity_threshold
        too_long = current_len >= max_chunk_chars
        if topic_shift or too_long:
            chunks.append((text[sentences[group_start][1] : sentences[i - 1][2]], sentences[group_start][1], sentences[i - 1][2]))
            group_start = i
    chunks.append((text[sentences[group_start][1] : sentences[-1][2]], sentences[group_start][1], sentences[-1][2]))
    return chunks


# --------------------------------------------------------------------------
# 4. Sentence-window / parent-document chunking
# --------------------------------------------------------------------------


def sentence_window_chunk(text: str, window_size: int = 2) -> list[tuple[str, int, int, str]]:
    """Each returned unit is a single sentence (precise enough to match a
    query closely), paired with the surrounding `window_size`-sentence
    context on each side (enough context for the LLM to actually answer
    from). Returns (sentence_text, char_start, char_end, window_text)."""
    sentences = split_sentences(text)
    results = []
    for i, (sentence, s, e) in enumerate(sentences):
        lo = max(0, i - window_size)
        hi = min(len(sentences) - 1, i + window_size)
        window_text = text[sentences[lo][1] : sentences[hi][2]]
        results.append((sentence, s, e, window_text))
    return results


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

STRATEGIES = ("fixed_size", "recursive", "semantic", "sentence_window")


def chunk_document(document: Document, strategy: str, embed_fn: EmbedFn | None = None, **kwargs) -> list[Chunk]:
    if strategy == "fixed_size":
        spans = fixed_size_chunk(document.text, **kwargs)
    elif strategy == "recursive":
        spans = recursive_chunk(document.text, **kwargs)
    elif strategy == "semantic":
        spans = semantic_chunk(document.text, embed_fn=embed_fn, **kwargs)
    elif strategy == "sentence_window":
        spans = sentence_window_chunk(document.text, **kwargs)
    else:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {STRATEGIES}")

    chunks: list[Chunk] = []
    for idx, span in enumerate(spans):
        text, start, end = span[0], span[1], span[2]
        metadata = {"window_text": span[3]} if strategy == "sentence_window" else {}
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}-{strategy}-{idx}-{uuid.uuid4().hex[:6]}",
                doc_id=document.doc_id,
                source_path=document.source_path,
                filetype=document.filetype,
                strategy=strategy,
                chunk_index=idx,
                text=text,
                char_start=start,
                char_end=end,
                page_start=document.page_for_offset(start),
                page_end=document.page_for_offset(max(end - 1, start)),
                metadata=metadata,
            )
        )
    return chunks
