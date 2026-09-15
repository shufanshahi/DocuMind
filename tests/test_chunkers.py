import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingest.chunkers import (  # noqa: E402
    fixed_size_chunk,
    recursive_chunk,
    semantic_chunk,
    sentence_window_chunk,
    split_sentences,
)
from ingest.embedders import HashingBagOfWordsEmbedder  # noqa: E402

# Always pin the dependency-free fallback embedder in tests: now that
# sentence-transformers is installed (Phase 2), semantic_chunk()'s default
# get_embedder() would otherwise load a real model, making these tests slow
# and dependent on model-cache/network availability.
FAKE_EMBED_FN = HashingBagOfWordsEmbedder()

LOREM = (
    "The quick brown fox jumps over the lazy dog. "
    "It was a bright cold day in April, and the clocks were striking thirteen. "
    "Call me Ishmael. Some years ago, never mind how long precisely, having "
    "little or no money in my purse, I thought I would sail about a little."
)


def _assert_offsets_reconstruct(text, spans):
    """Every (span_text, start, end) must satisfy text[start:end] == span_text."""
    for item in spans:
        span_text, start, end = item[0], item[1], item[2]
        assert text[start:end] == span_text


def test_fixed_size_covers_whole_text_with_correct_offsets():
    spans = fixed_size_chunk(LOREM, chunk_size=50, overlap=10)
    _assert_offsets_reconstruct(LOREM, spans)
    assert spans[0][1] == 0
    assert spans[-1][2] == len(LOREM)


def test_fixed_size_rejects_overlap_ge_chunk_size():
    try:
        fixed_size_chunk(LOREM, chunk_size=10, overlap=10)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_recursive_respects_chunk_size_and_offsets():
    spans = recursive_chunk(LOREM, chunk_size=60, overlap=10)
    _assert_offsets_reconstruct(LOREM, spans)
    assert all(len(s[0]) <= 60 for s in spans)
    assert spans[0][1] == 0
    assert spans[-1][2] == len(LOREM)


def test_recursive_handles_text_shorter_than_chunk_size():
    short = "Just one short sentence."
    spans = recursive_chunk(short, chunk_size=500, overlap=50)
    assert len(spans) == 1
    assert spans[0][0] == short


def test_split_sentences_offsets_are_exact():
    sentences = split_sentences(LOREM)
    assert len(sentences) == 4
    for text, start, end in sentences:
        assert LOREM[start:end] == text


def test_semantic_chunk_covers_whole_text():
    spans = semantic_chunk(LOREM, embed_fn=FAKE_EMBED_FN, similarity_threshold=0.9)  # force many splits
    _assert_offsets_reconstruct(LOREM, spans)
    assert spans[0][1] == 0
    assert spans[-1][2] == len(LOREM)


def test_semantic_chunk_empty_text():
    assert semantic_chunk("") == []


def test_sentence_window_context_includes_target_sentence():
    windows = sentence_window_chunk(LOREM, window_size=1)
    for sentence, start, end, window_text in windows:
        assert sentence in window_text
        assert LOREM[start:end] == sentence


def test_sentence_window_first_and_last_dont_crash_on_boundary():
    windows = sentence_window_chunk(LOREM, window_size=5)
    assert len(windows) == 4  # window_size larger than doc just clamps to full text
