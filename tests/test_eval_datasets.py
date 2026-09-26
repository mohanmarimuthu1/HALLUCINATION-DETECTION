"""Golden-set YAML loader tests (Phase 7.1) - offline, no network."""
import pytest
from pydantic import ValidationError

from halludetect.detect.schemas import Verdict
from halludetect.eval.datasets import GoldenCategory, load_golden_set


def _write(tmp_path, text):
    path = tmp_path / "set.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_real_golden_set_a():
    from pathlib import Path

    path = Path(__file__).parent / "data" / "golden" / "eval_set_a.yaml"
    items = load_golden_set(path)
    assert len(items) >= 30
    assert all(item.evidence for item in items)
    categories = {item.category for item in items}
    assert categories == set(GoldenCategory)


def test_parses_fields(tmp_path):
    path = _write(
        tmp_path,
        """
        - id: item-1
          category: answerable_in_evidence
          question: "What is X?"
          answer: "X is Y."
          evidence:
            - "X is Y, confirmed."
          expected_verdict: GROUNDED
          notes: "a note"
        """,
    )
    items = load_golden_set(path)
    assert len(items) == 1
    item = items[0]
    assert item.id == "item-1"
    assert item.category == GoldenCategory.ANSWERABLE_IN_EVIDENCE
    assert item.expected_verdict == Verdict.GROUNDED
    assert item.notes == "a note"


def test_question_is_optional(tmp_path):
    path = _write(
        tmp_path,
        """
        - id: item-1
          category: unanswerable_in_evidence
          answer: "X is Y."
          evidence:
            - "unrelated"
          expected_verdict: NOT_VERIFIABLE
        """,
    )
    items = load_golden_set(path)
    assert items[0].question is None
    assert items[0].notes == ""


def test_duplicate_ids_raise(tmp_path):
    path = _write(
        tmp_path,
        """
        - id: dup
          category: answerable_in_evidence
          answer: "A"
          evidence: ["A confirmed"]
          expected_verdict: GROUNDED
        - id: dup
          category: injected_contradiction
          answer: "B"
          evidence: ["B contradicted"]
          expected_verdict: CONTRADICTED
        """,
    )
    with pytest.raises(ValueError, match="duplicate golden item ids"):
        load_golden_set(path)


def test_invalid_category_raises(tmp_path):
    path = _write(
        tmp_path,
        """
        - id: item-1
          category: not_a_real_category
          answer: "A"
          evidence: ["A"]
          expected_verdict: GROUNDED
        """,
    )
    with pytest.raises(ValidationError):
        load_golden_set(path)
