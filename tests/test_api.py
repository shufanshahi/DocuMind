import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from ingest.chunkers import Chunk  # noqa: E402
from rag_langchain import RagAnswer  # noqa: E402

import api.main as api_main  # noqa: E402

client = TestClient(api_main.app)


def make_chunk(chunk_id, text, source_path="data/raw/doc.txt", page_start=None):
    return Chunk(
        chunk_id=chunk_id,
        doc_id="d",
        source_path=source_path,
        filetype="text",
        strategy="fixed_size",
        chunk_index=0,
        text=text,
        char_start=0,
        char_end=len(text),
        page_start=page_start,
        page_end=page_start,
    )


class FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, query, k=5):
        return self._chunks[:k]


@pytest.fixture(autouse=True)
def clear_caches():
    # Every test gets a clean slate rather than reusing whatever a
    # previous test cached under the same (strategy, embedder, ...) key.
    api_main._retriever_cache.clear()
    api_main._embedder_cache.clear()
    yield
    api_main._retriever_cache.clear()
    api_main._embedder_cache.clear()


def install_fake_retriever(monkeypatch, chunks, strategy="recursive", embedder="nomic_embed_text", hybrid=False, rerank=False):
    key = (strategy, embedder, hybrid, rerank)
    api_main._retriever_cache[key] = FakeRetriever(chunks)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_grounded_answer_with_sources(monkeypatch):
    install_fake_retriever(monkeypatch, [make_chunk("a", "relevant fact one", page_start=3)])
    monkeypatch.setattr(
        api_main,
        "_llm",
        lambda prompt_value: "FAKE ANSWER",
    )

    response = client.post("/ask", json={"question": "some question"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "FAKE ANSWER"
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source_path"] == "data/raw/doc.txt"
    assert data["sources"][0]["page_start"] == 3
    assert data["sources"][0]["citation_index"] == 1
    assert data["chunks_used"] == 1
    assert data["chunks_dropped"] == 0


def test_ask_refuses_without_calling_llm_when_budget_too_small(monkeypatch):
    install_fake_retriever(monkeypatch, [make_chunk("a", "relevant fact one")])
    calls = []
    monkeypatch.setattr(api_main, "_llm", lambda prompt_value: calls.append(prompt_value) or "SHOULD NOT BE CALLED")

    response = client.post("/ask", json={"question": "some question", "token_budget": 1})
    assert response.status_code == 200
    data = response.json()
    assert "don't have enough information" in data["answer"].lower()
    assert calls == []
    assert data["chunks_used"] == 0
    assert data["chunks_dropped"] == 1


def test_ask_rejects_unknown_strategy():
    response = client.post("/ask", json={"question": "q", "strategy": "not_a_real_strategy"})
    assert response.status_code == 400


def test_ask_rejects_unknown_embedder():
    response = client.post("/ask", json={"question": "q", "embedder": "not_a_real_embedder"})
    assert response.status_code == 400


def test_ask_rejects_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422  # pydantic validation: min_length=1


def test_ask_rejects_invalid_prompt_variant():
    response = client.post("/ask", json={"question": "q", "prompt_variant": "not_a_real_variant"})
    assert response.status_code == 422


def test_retriever_cache_reuses_the_same_instance(monkeypatch):
    install_fake_retriever(monkeypatch, [make_chunk("a", "fact")])
    cached = api_main.get_retriever("recursive", "nomic_embed_text", False, False)
    assert cached is api_main._retriever_cache[("recursive", "nomic_embed_text", False, False)]
