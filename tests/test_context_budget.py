import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingest.chunkers import Chunk  # noqa: E402
from rag.context_budget import TokenCounter, assemble_context  # noqa: E402


def make_chunk(chunk_id, text):
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        source_path="data/raw/doc.txt",
        filetype="text",
        strategy="fixed_size",
        chunk_index=0,
        text=text,
        char_start=0,
        char_end=len(text),
    )


class FixedCounter:
    """A TokenCounter stand-in that reports an exact, caller-chosen token
    count per call regardless of text length, so budget-edge tests don't
    depend on tiktoken's (or the chars/4 fallback's) actual counts."""

    def __init__(self, tokens_per_chunk: int):
        self.tokens_per_chunk = tokens_per_chunk

    def count(self, text: str) -> int:
        return self.tokens_per_chunk


def test_all_chunks_fit_under_a_generous_budget():
    chunks = [make_chunk("a", "short"), make_chunk("b", "also short")]
    result = assemble_context(chunks, token_budget=1000, counter=FixedCounter(10))
    assert result.used_chunks == chunks
    assert result.dropped_chunks == []
    assert result.total_tokens == 20


def test_stops_at_first_overflow_and_drops_the_rest():
    chunks = [make_chunk("a", "one"), make_chunk("b", "two"), make_chunk("c", "three")]
    # Each chunk costs 10 tokens; budget only fits one.
    result = assemble_context(chunks, token_budget=15, counter=FixedCounter(10))
    assert [c.chunk_id for c in result.used_chunks] == ["a"]
    assert [c.chunk_id for c in result.dropped_chunks] == ["b", "c"]
    assert result.total_tokens == 10


def test_does_not_bin_pack_a_smaller_chunk_past_a_dropped_larger_one():
    # "a" costs 20 (won't fit in a budget of 15) but "b" costs 5 (would
    # fit). assemble_context must NOT skip "a" and pack "b" in its place --
    # the retriever's rank order is priority, not just a size constraint.
    chunks = [make_chunk("a", "big"), make_chunk("b", "small")]

    class VariableCounter:
        def count(self, text):
            return 20 if "[source: 1]" in text else 5

    result = assemble_context(chunks, token_budget=15, counter=VariableCounter())
    assert result.used_chunks == []
    assert [c.chunk_id for c in result.dropped_chunks] == ["a", "b"]


def test_citation_tags_are_numbered_from_one_in_rank_order():
    chunks = [make_chunk("a", "first"), make_chunk("b", "second")]
    result = assemble_context(chunks, token_budget=1000, counter=FixedCounter(1))
    assert "[source: 1] first" in result.text
    assert "[source: 2] second" in result.text


def test_empty_chunk_list_produces_empty_context():
    result = assemble_context([], token_budget=1000, counter=FixedCounter(1))
    assert result.text == ""
    assert result.used_chunks == []
    assert result.total_tokens == 0


def test_token_counter_falls_back_without_crashing():
    # Whether or not tiktoken/network is available, count() must return a
    # positive int for non-empty text.
    counter = TokenCounter()
    assert counter.count("hello world") > 0
