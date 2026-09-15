"""Load the labeled eval set: 20 question/answer pairs, each tagged with
which source document(s) should be retrieved and a category
(single_hop / multi_hop / adversarial). See explaination/phase7_theory.md
§1 for why ground truth is at the *document* level rather than the exact
chunk level, and why 20 rather than the plan's full 20-30 range.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalCase:
    id: str
    question: str
    ideal_answer: str
    expected_sources: list[str]  # Chunk.source_path values; empty for adversarial cases
    category: str  # "single_hop" | "multi_hop" | "adversarial"


def load_eval_set(path: str | Path = "data/eval/eval_set.jsonl") -> list[EvalCase]:
    cases = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            cases.append(EvalCase(**data))
    return cases
