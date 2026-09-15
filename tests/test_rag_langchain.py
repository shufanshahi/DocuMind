import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # for `import rag_langchain`
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingest.chunkers import Chunk  # noqa: E402
from prompts.templates import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT, SYSTEM_PROMPT_BASELINE  # noqa: E402
from rag_langchain import ask, build_chain, build_prompt  # noqa: E402


def make_chunk(chunk_id, text):
    return Chunk(
        chunk_id=chunk_id,
        doc_id="d",
        source_path="data/raw/doc.txt",
        filetype="text",
        strategy="fixed_size",
        chunk_index=0,
        text=text,
        char_start=0,
        char_end=len(text),
    )


class FakeRetriever:
    """Stands in for Phase 3's Retriever so chain-assembly logic can be
    tested without a real embedder, Chroma collection, or Ollama call."""

    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, query, k=5):
        return self._chunks[:k]


def make_fake_llm(response="FAKE ANSWER", calls=None):
    """A plain callable stands in for a LangChain chat model here --
    LangChain's `|` operator auto-wraps any callable into a RunnableLambda,
    so this slots into `PROMPT | llm | StrOutputParser()` exactly like a
    real `ChatOllama` would, with no network or model load required."""

    def fake_llm(prompt_value):
        if calls is not None:
            calls.append(prompt_value)
        return response

    return fake_llm


def test_chain_generates_when_context_fits():
    retriever = FakeRetriever([make_chunk("a", "relevant fact one"), make_chunk("b", "relevant fact two")])
    chain = build_chain(retriever, make_fake_llm(), k=2, token_budget=1000)
    result = ask(chain, "some question")
    assert result.answer == "FAKE ANSWER"
    assert [c.chunk_id for c in result.used_chunks] == ["a", "b"]
    assert result.dropped_chunks == []


def test_chain_refuses_without_calling_llm_when_budget_is_too_small():
    retriever = FakeRetriever([make_chunk("a", "relevant fact one")])
    calls = []
    chain = build_chain(retriever, make_fake_llm(calls=calls), k=1, token_budget=1)
    result = ask(chain, "some question")
    assert "don't have enough information" in result.answer.lower()
    assert calls == []  # the LLM must never be invoked with empty context
    assert result.used_chunks == []
    assert len(result.dropped_chunks) == 1


def test_chain_refuses_when_retriever_finds_nothing():
    retriever = FakeRetriever([])
    chain = build_chain(retriever, make_fake_llm(), k=5, token_budget=1000)
    result = ask(chain, "some question")
    assert "don't have enough information" in result.answer.lower()


def test_ask_reports_context_token_count():
    retriever = FakeRetriever([make_chunk("a", "some text")])
    chain = build_chain(retriever, make_fake_llm(), k=1, token_budget=1000)
    result = ask(chain, "some question")
    assert result.context_tokens > 0


def test_build_prompt_includes_few_shot_messages_by_default():
    prompt = build_prompt()
    contents = [m.content for m in prompt.invoke({"context": "CTX", "question": "Q"}).to_messages()]
    assert len(contents) == 1 + len(FEW_SHOT_EXAMPLES) * 2 + 1  # system + examples + final human turn
    assert contents[0] == SYSTEM_PROMPT


def test_build_prompt_excludes_few_shot_messages_when_disabled():
    prompt = build_prompt(use_few_shot=False)
    contents = [m.content for m in prompt.invoke({"context": "CTX", "question": "Q"}).to_messages()]
    assert len(contents) == 2  # just system + final human turn


def test_build_prompt_uses_the_given_system_prompt():
    prompt = build_prompt(system_prompt=SYSTEM_PROMPT_BASELINE, use_few_shot=False)
    contents = [m.content for m in prompt.invoke({"context": "CTX", "question": "Q"}).to_messages()]
    assert contents[0] == SYSTEM_PROMPT_BASELINE


def test_build_chain_plumbs_prompt_variant_through_to_the_llm_call():
    retriever = FakeRetriever([make_chunk("a", "relevant fact one")])
    captured = []
    chain = build_chain(
        retriever, make_fake_llm(calls=captured), k=1, token_budget=1000, system_prompt=SYSTEM_PROMPT_BASELINE, use_few_shot=False
    )
    ask(chain, "some question")
    assert len(captured) == 1
    contents = [m.content for m in captured[0].to_messages()]
    assert contents[0] == SYSTEM_PROMPT_BASELINE
    assert len(contents) == 2  # no few-shot examples spliced in
