import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # for `import rag_llamaindex`
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole  # noqa: E402
from llama_index.core.retrievers import BaseRetriever  # noqa: E402
from llama_index.core.schema import NodeWithScore, TextNode  # noqa: E402

from prompts.templates import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT_BASELINE  # noqa: E402
from rag_llamaindex import DocuMindQueryEngine, _chunk_from_node  # noqa: E402


def make_node(node_id, text, metadata=None, score=1.0) -> NodeWithScore:
    return NodeWithScore(node=TextNode(id_=node_id, text=text, metadata=metadata or {}), score=score)


class FakeRetriever(BaseRetriever):
    """A real BaseRetriever subclass -- DocuMindQueryEngine.retriever is a
    pydantic field typed `BaseRetriever`, so a plain duck-typed object fails
    construction validation, not just the call site (same reasoning as
    Phase 3's FakeEmbedder needing to subclass the `Embedder` ABC)."""

    def __init__(self, nodes: list[NodeWithScore]):
        super().__init__()
        self._nodes = nodes

    def _retrieve(self, query_bundle) -> list[NodeWithScore]:
        return self._nodes


class FakeLLM:
    """DocuMindQueryEngine.llm is typed as plain `object` specifically so a
    minimal fake like this -- no LlamaIndex base class needed -- can stand
    in for `llama_index.llms.ollama.Ollama` in tests."""

    def __init__(self, content="FAKE ANSWER"):
        self.content = content
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=self.content))


def test_chunk_from_node_reconstructs_pdf_page_metadata():
    node = make_node("n1", "a pdf chunk", metadata={"doc_id": "d1", "source_path": "x.pdf", "filetype": "pdf", "page_start": 3, "page_end": 3})
    chunk = _chunk_from_node(node, fallback_strategy="fixed_size")
    assert chunk.chunk_id == "n1"
    assert chunk.source_path == "x.pdf"
    assert chunk.page_start == 3
    assert chunk.page_end == 3
    assert chunk.text == "a pdf chunk"


def test_chunk_from_node_falls_back_to_given_strategy_when_missing():
    node = make_node("n1", "text", metadata={})
    chunk = _chunk_from_node(node, fallback_strategy="semantic")
    assert chunk.strategy == "semantic"


def test_chunk_from_node_keeps_extra_metadata_like_window_text():
    node = make_node("n1", "sentence", metadata={"window_text": "wider context"})
    chunk = _chunk_from_node(node, fallback_strategy="sentence_window")
    assert chunk.metadata["window_text"] == "wider context"


def test_custom_query_generates_when_context_fits():
    retriever = FakeRetriever([make_node("a", "relevant fact one"), make_node("b", "relevant fact two")])
    llm = FakeLLM()
    engine = DocuMindQueryEngine(retriever=retriever, llm=llm, strategy="fixed_size", k=2, token_budget=1000)
    response = engine.query("some question")
    assert response.response == "FAKE ANSWER"
    assert len(llm.calls) == 1
    assert [c.chunk_id for c in response.metadata["used_chunks"]] == ["a", "b"]


def test_custom_query_refuses_without_calling_llm_when_budget_too_small():
    retriever = FakeRetriever([make_node("a", "relevant fact one")])
    llm = FakeLLM()
    engine = DocuMindQueryEngine(retriever=retriever, llm=llm, strategy="fixed_size", k=1, token_budget=1)
    response = engine.query("some question")
    assert "don't have enough information" in response.response.lower()
    assert llm.calls == []  # the LLM must never be invoked with empty context
    assert response.metadata["used_chunks"] == []


def test_custom_query_refuses_when_retriever_finds_nothing():
    retriever = FakeRetriever([])
    llm = FakeLLM()
    engine = DocuMindQueryEngine(retriever=retriever, llm=llm, strategy="fixed_size", k=5, token_budget=1000)
    response = engine.query("some question")
    assert "don't have enough information" in response.response.lower()


def test_custom_query_includes_few_shot_messages_by_default():
    retriever = FakeRetriever([make_node("a", "relevant fact one")])
    llm = FakeLLM()
    engine = DocuMindQueryEngine(retriever=retriever, llm=llm, strategy="fixed_size", k=1, token_budget=1000)
    engine.query("some question")
    assert len(llm.calls) == 1
    messages = llm.calls[0]
    assert len(messages) == 1 + len(FEW_SHOT_EXAMPLES) * 2 + 1  # system + examples + final user turn


def test_custom_query_excludes_few_shot_messages_when_disabled():
    retriever = FakeRetriever([make_node("a", "relevant fact one")])
    llm = FakeLLM()
    engine = DocuMindQueryEngine(retriever=retriever, llm=llm, strategy="fixed_size", k=1, token_budget=1000, use_few_shot=False)
    engine.query("some question")
    messages = llm.calls[0]
    assert len(messages) == 2  # just system + final user turn


def test_custom_query_uses_the_given_system_prompt():
    retriever = FakeRetriever([make_node("a", "relevant fact one")])
    llm = FakeLLM()
    engine = DocuMindQueryEngine(
        retriever=retriever, llm=llm, strategy="fixed_size", k=1, token_budget=1000, system_prompt=SYSTEM_PROMPT_BASELINE, use_few_shot=False
    )
    engine.query("some question")
    assert llm.calls[0][0].content == SYSTEM_PROMPT_BASELINE
