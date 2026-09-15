import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingest.chunkers import Chunk  # noqa: E402
from retrieval.hybrid_search import BM25Index, reciprocal_rank_fusion, tokenize  # noqa: E402


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


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Nomic-Embed-Text, and RAG!") == ["nomic", "embed", "text", "and", "rag"]


def test_bm25_ranks_exact_keyword_match_first():
    chunks = [
        make_chunk("a", "the quick brown fox jumps over the lazy dog"),
        make_chunk("b", "nomic-embed-text is an embedding model served by ollama"),
        make_chunk("c", "sentence transformers run entirely on your own machine"),
    ]
    index = BM25Index(chunks)
    hits = index.search("nomic embed text ollama", k=2)
    assert hits[0][0].chunk_id == "b"
    assert hits[0][1] > 0


def test_bm25_from_jsonl_roundtrip(tmp_path):
    import json
    from dataclasses import asdict

    jsonl_path = tmp_path / "fixed_size.jsonl"
    # 3+ documents: with only 2 docs, a term appearing in exactly half of
    # them hits BM25's classic IDF formula at exactly zero (log(1) == 0),
    # which masks the ranking effect this test wants to demonstrate.
    chunks = [
        make_chunk("a", "cats are animals"),
        make_chunk("b", "rockets fly to space"),
        make_chunk("c", "dogs are animals too"),
    ]
    with jsonl_path.open("w") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")

    index = BM25Index.from_jsonl(jsonl_path)
    hits = index.search("rockets", k=1)
    assert hits[0][0].chunk_id == "b"


def test_rrf_rewards_agreement_between_rankings():
    # "shared" is ranked #1 by one method and #2 by the other; "only_a" is
    # #1 by the other method but absent from the first. RRF should place
    # the item both methods agree is relevant above an item only one method saw.
    ranking_a = ["shared", "only_a"]
    ranking_b = ["only_b", "shared"]
    fused = reciprocal_rank_fusion([ranking_a, ranking_b])
    assert fused["shared"] > fused["only_a"]
    assert fused["shared"] > fused["only_b"]


def test_rrf_handles_disjoint_rankings():
    fused = reciprocal_rank_fusion([["x", "y"], ["z"]])
    assert set(fused) == {"x", "y", "z"}
    assert fused["x"] > fused["y"]  # x ranked higher within its own list
