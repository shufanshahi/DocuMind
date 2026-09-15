import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prompts.templates import (  # noqa: E402
    FEW_SHOT_EXAMPLES,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_BASELINE,
    build_few_shot_messages,
)


def test_few_shot_messages_alternate_user_and_assistant():
    messages = build_few_shot_messages()
    assert len(messages) == len(FEW_SHOT_EXAMPLES) * 2
    for i, (role, _content) in enumerate(messages):
        assert role == ("user" if i % 2 == 0 else "assistant")


def test_few_shot_user_turns_include_numbered_citations_and_the_question():
    messages = build_few_shot_messages()
    first_user_turn = messages[0][1]
    assert "[source: 1]" in first_user_turn
    assert FEW_SHOT_EXAMPLES[0]["question"] in first_user_turn


def test_few_shot_assistant_turns_match_the_authored_answers():
    messages = build_few_shot_messages()
    assistant_turns = [content for role, content in messages if role == "assistant"]
    assert assistant_turns == [example["answer"] for example in FEW_SHOT_EXAMPLES]


def test_few_shot_content_has_no_stray_braces():
    # Few-shot turns get spliced into a LangChain ChatPromptTemplate as
    # static (already-rendered) strings -- see rag_langchain.py's
    # build_prompt(). A stray "{" or "}" here would be misinterpreted as a
    # template variable at format time.
    for _role, content in build_few_shot_messages():
        assert "{" not in content
        assert "}" not in content


def test_guarded_prompt_reinforces_baseline_rules():
    # The guarded prompt should still cover every guardrail the baseline
    # names explicitly (refuse when insufficient, cite every claim) --
    # Phase 6 refines the wording, it doesn't drop a requirement.
    assert "i don't have enough information" in SYSTEM_PROMPT.lower()
    assert "[source: n]" in SYSTEM_PROMPT.lower()
    assert SYSTEM_PROMPT != SYSTEM_PROMPT_BASELINE
