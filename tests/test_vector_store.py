import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingest.chunkers import Chunk  # noqa: E402
from retrieval.vector_store import ChromaVectorStore, chunk_to_metadata  # noqa: E402


def make_chunk(chunk_id, text, page_start=None, page_end=None, metadata=None):
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
        page_start=page_start,
        page_end=page_end,
        metadata=metadata or {},
    )


def test_chunk_to_metadata_drops_none_fields():
    chunk = make_chunk("c1", "hello")
    meta = chunk_to_metadata(chunk)
    assert "page_start" not in meta
    assert "page_end" not in meta
    assert meta["doc_id"] == "doc-1"


def test_chunk_to_metadata_keeps_pdf_pages_and_extra_metadata():
    chunk = make_chunk("c1", "hello", page_start=1, page_end=2, metadata={"window_text": "wider context"})
    meta = chunk_to_metadata(chunk)
    assert meta["page_start"] == 1
    assert meta["page_end"] == 2
    assert meta["window_text"] == "wider context"


def test_upsert_and_query_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        store = ChromaVectorStore(persist_dir=tmp)
        chunks = [make_chunk("a", "cats are animals"), make_chunk("b", "rockets fly to space")]
        # Hand-built 2D vectors so we know exactly which one should be nearest.
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        store.upsert_chunks("test_collection", "fake_embedder", chunks, embeddings)

        assert store.count("test_collection") == 2
        results = store.query("test_collection", [0.9, 0.1], k=1)
        assert len(results) == 1
        assert results[0].chunk_id == "a"


def test_upsert_rejects_mismatched_lengths():
    with tempfile.TemporaryDirectory() as tmp:
        store = ChromaVectorStore(persist_dir=tmp)
        chunks = [make_chunk("a", "one"), make_chunk("b", "two")]
        try:
            store.upsert_chunks("c", "fake", chunks, [[1.0, 0.0]])
            assert False, "expected ValueError"
        except ValueError:
            pass
