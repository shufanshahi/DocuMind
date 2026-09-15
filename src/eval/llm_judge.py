"""Custom LLM-as-judge rubric for generation quality, used instead of
RAGAS (see explaination/phase7_theory.md §2 for why): scores an answer's
faithfulness (is every claim actually supported by the given context?) and
relevancy (does it address the question asked?), each 1-5, using the same
local llama3.1:8b model that generated the answer in the first place.

Faithfulness and relevancy are scored with two SEPARATE calls, not one
combined rubric -- this was not the original design. A combined prompt
(rate both in one response) was tested against a deliberately fabricated
answer (a fake citation, "Miller (1956) showed chunk size should be 7±2")
and rated it fully faithful (5/5) every time, even after strengthening the
combined prompt's wording. Splitting into two single-purpose calls, each
with only one rubric to focus on, correctly caught the same fabrication
(1/5) on the first try. See explaination/phase7_theory.md §3 for the full
story -- this is itself one of Phase 7's findings, not just an
implementation detail.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

FAITHFULNESS_PROMPT_TEMPLATE = """You are a strict fact-checker for a retrieval-augmented generation system. Below is a CONTEXT and an ANSWER that claims to be based on it.

Go through the ANSWER claim by claim. For every specific detail in it -- a name, a number, a date, a citation, a technical term -- check whether that exact detail appears in the CONTEXT. A plausible-sounding detail that does NOT literally appear in the CONTEXT makes the answer unfaithful, even if the general topic is correct.

Score from 1 (worst) to 5 (best): 5 means every specific detail traces back to the context with nothing invented; 1 means the answer states specific things the context never said. An answer that correctly says "I don't have enough information" when the context doesn't support an answer is fully faithful (5).

Respond with ONLY JSON: {{"faithfulness": <integer 1-5>, "reasoning": "<name the specific unsupported detail, if any>"}}

CONTEXT:
{context}

ANSWER:
{answer}
"""

RELEVANCY_PROMPT_TEMPLATE = """You are an evaluation judge. Does the ANSWER below actually address the QUESTION that was asked?

Score from 1 (worst) to 5 (best): 5 means it directly and completely addresses the question; 1 means it is off-topic or non-responsive. A correct refusal counts as relevant (5) if it accurately explains that the information is not available; a vague or evasive non-answer scores lower.

Respond with ONLY JSON: {{"relevancy": <integer 1-5>, "reasoning": "<one short sentence>"}}

QUESTION:
{question}

ANSWER:
{answer}
"""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeScore:
    faithfulness: int | None
    relevancy: int | None
    faithfulness_reasoning: str
    relevancy_reasoning: str
    parse_error: bool = False


def _extract_json(raw: str) -> dict | None:
    """Pull the JSON object out of a judge response. Models occasionally
    wrap JSON in a sentence or a code fence despite being asked for
    "ONLY" JSON, so this extracts the first `{...}` block rather than
    requiring the whole response to already be valid JSON on its own."""
    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _invoke(llm, prompt: str) -> str:
    raw = llm.invoke(prompt)
    return raw.content if hasattr(raw, "content") else str(raw)


def judge_answer(llm, question: str, context: str, answer: str) -> JudgeScore:
    """Two separate calls to `llm` (anything with `.invoke(str)` returning
    a string or a LangChain-style message with `.content` -- a `ChatOllama`
    works directly): one scores faithfulness against the context, one
    scores relevancy against the question. A response that can't be parsed
    reports `None` for that score and `parse_error=True` rather than being
    silently coerced into a fake number -- see `run_eval.py` for how these
    get excluded from averages instead of counted as a 0/5.
    """
    faith_raw = _invoke(llm, FAITHFULNESS_PROMPT_TEMPLATE.format(context=context, answer=answer))
    rel_raw = _invoke(llm, RELEVANCY_PROMPT_TEMPLATE.format(question=question, answer=answer))

    faith_data = _extract_json(faith_raw)
    rel_data = _extract_json(rel_raw)

    parse_error = faith_data is None or rel_data is None
    try:
        faithfulness = int(faith_data["faithfulness"]) if faith_data else None
    except (KeyError, TypeError, ValueError):
        faithfulness, parse_error = None, True
    try:
        relevancy = int(rel_data["relevancy"]) if rel_data else None
    except (KeyError, TypeError, ValueError):
        relevancy, parse_error = None, True

    return JudgeScore(
        faithfulness=faithfulness,
        relevancy=relevancy,
        faithfulness_reasoning=(faith_data or {}).get("reasoning", "") if faith_data else f"unparseable response: {faith_raw!r}",
        relevancy_reasoning=(rel_data or {}).get("reasoning", "") if rel_data else f"unparseable response: {rel_raw!r}",
        parse_error=parse_error,
    )
