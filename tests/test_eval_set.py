import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval.eval_set import load_eval_set  # noqa: E402


def test_load_eval_set_from_temp_file(tmp_path):
    jsonl_path = tmp_path / "eval_set.jsonl"
    rows = [
        {"id": "q1", "question": "Q1?", "ideal_answer": "A1", "expected_sources": ["a.md"], "category": "single_hop"},
        {"id": "q2", "question": "Q2?", "ideal_answer": "refuse", "expected_sources": [], "category": "adversarial"},
    ]
    jsonl_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    cases = load_eval_set(jsonl_path)
    assert len(cases) == 2
    assert cases[0].id == "q1"
    assert cases[0].expected_sources == ["a.md"]
    assert cases[1].category == "adversarial"
    assert cases[1].expected_sources == []


def test_the_real_eval_set_loads_and_has_the_expected_categories():
    # Resolved relative to this test file, not the CWD, so it works
    # regardless of where pytest is invoked from.
    real_path = Path(__file__).parent.parent / "data" / "eval" / "eval_set.jsonl"
    cases = load_eval_set(real_path)
    assert len(cases) == 20
    categories = [c.category for c in cases]
    assert categories.count("single_hop") == 13
    assert categories.count("multi_hop") == 3
    assert categories.count("adversarial") == 4
    for case in cases:
        if case.category == "adversarial":
            assert case.expected_sources == []
        else:
            assert len(case.expected_sources) >= 1
