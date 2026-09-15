import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from guardrail_experiments import has_citation, is_refusal  # noqa: E402


def test_is_refusal_matches_the_canonical_phrase():
    assert is_refusal("I don't have enough information to answer that.")


def test_is_refusal_is_case_insensitive():
    assert is_refusal("I DON'T HAVE ENOUGH INFORMATION to answer that.")


def test_is_refusal_false_for_a_grounded_answer():
    assert not is_refusal("Chunk size matters because [source: 1] explains it.")


def test_has_citation_true_when_source_tag_present():
    assert has_citation("Some claim [source: 2].")


def test_has_citation_false_without_a_source_tag():
    assert not has_citation("I don't have enough information to answer that.")
