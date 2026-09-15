import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embeddings.nomic_embedder import DOCUMENT_PREFIX, QUERY_PREFIX, NomicEmbedder  # noqa: E402


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload


def make_embedder(monkeypatch, dim=4):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse({"embeddings": [[0.1] * dim for _ in json["input"]]})

    monkeypatch.setattr("embeddings.nomic_embedder.requests.post", fake_post)
    embedder = NomicEmbedder()
    return embedder, captured


def test_embed_documents_applies_document_prefix(monkeypatch):
    embedder, captured = make_embedder(monkeypatch)
    embedder.embed_documents(["hello", "world"])
    assert captured["json"]["input"] == [DOCUMENT_PREFIX + "hello", DOCUMENT_PREFIX + "world"]


def test_embed_query_applies_query_prefix(monkeypatch):
    embedder, captured = make_embedder(monkeypatch)
    embedder.embed_query("what is RAG")
    assert captured["json"]["input"] == [QUERY_PREFIX + "what is RAG"]


def test_dimension_probed_on_init(monkeypatch):
    embedder, _ = make_embedder(monkeypatch, dim=768)
    assert embedder.dimension == 768


def test_non_200_response_raises_with_helpful_message(monkeypatch):
    def fake_post(url, json, timeout):
        return FakeResponse({}, status_code=500)

    monkeypatch.setattr("embeddings.nomic_embedder.requests.post", fake_post)
    try:
        NomicEmbedder()
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "ollama pull" in str(e).lower()
