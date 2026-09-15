import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval.llm_judge import judge_answer  # noqa: E402


class FakeLLM:
    """Returns canned responses in call order -- the first `.invoke()` call
    is the faithfulness prompt, the second is the relevancy prompt (matches
    `judge_answer`'s call order), so a test can script exactly what each
    judge call "says" without a real model."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return self._responses.pop(0)


def test_judge_answer_parses_clean_json_from_both_calls():
    llm = FakeLLM(
        [
            '{"faithfulness": 5, "reasoning": "fully supported"}',
            '{"relevancy": 4, "reasoning": "mostly on topic"}',
        ]
    )
    score = judge_answer(llm, question="Q", context="C", answer="A")
    assert score.faithfulness == 5
    assert score.relevancy == 4
    assert not score.parse_error


def test_judge_answer_extracts_json_wrapped_in_prose():
    llm = FakeLLM(
        [
            'Sure, here is my answer:\n{"faithfulness": 2, "reasoning": "invented a date"}\nHope that helps!',
            '{"relevancy": 5, "reasoning": "on topic"}',
        ]
    )
    score = judge_answer(llm, question="Q", context="C", answer="A")
    assert score.faithfulness == 2
    assert score.relevancy == 5
    assert not score.parse_error


def test_judge_answer_reports_parse_error_on_garbage_response():
    llm = FakeLLM(["not json at all", '{"relevancy": 3, "reasoning": "ok"}'])
    score = judge_answer(llm, question="Q", context="C", answer="A")
    assert score.faithfulness is None
    assert score.relevancy == 3
    assert score.parse_error


def test_judge_answer_reports_parse_error_when_expected_key_missing():
    llm = FakeLLM(['{"score": 5}', '{"relevancy": 4, "reasoning": "ok"}'])
    score = judge_answer(llm, question="Q", context="C", answer="A")
    assert score.faithfulness is None
    assert score.parse_error


def test_judge_answer_sends_faithfulness_prompt_first_then_relevancy():
    llm = FakeLLM(
        [
            '{"faithfulness": 5, "reasoning": "ok"}',
            '{"relevancy": 5, "reasoning": "ok"}',
        ]
    )
    judge_answer(llm, question="the question", context="the context", answer="the answer")
    assert "the context" in llm.prompts[0] and "FAITHFULNESS" in llm.prompts[0].upper()
    assert "the question" in llm.prompts[1] and "RELEVANCY" in llm.prompts[1].upper()
