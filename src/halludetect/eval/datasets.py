"""Golden-set item schema + YAML loader (Phase 7.1/7.2).

`GoldenItem` mirrors both `tests/data/golden/eval_set_a.yaml` (hand-authored,
evidence-provided) and `eval_set_b.yaml` (Phase 7.2, sampled from the public
HaluEval QA benchmark) field-for-field. Pydantic, not a dataclass, so a
malformed YAML entry (missing field, bad enum value) fails loudly at load
time rather than surfacing as a confusing `AttributeError` deep inside
`eval/runner.py`.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel

from halludetect.detect.schemas import Verdict


class GoldenCategory(str, Enum):
    ANSWERABLE_IN_EVIDENCE = "answerable_in_evidence"
    UNANSWERABLE_IN_EVIDENCE = "unanswerable_in_evidence"
    INJECTED_CONTRADICTION = "injected_contradiction"
    # Phase 7.2 (golden set B): sampled from a public benchmark (HaluEval),
    # not hand-crafted by this project - kept distinct from
    # `injected_contradiction`/`answerable_in_evidence` above so an item's
    # provenance is never misdescribed by its category alone.
    OPEN_DOMAIN_GROUNDED = "open_domain_grounded"
    OPEN_DOMAIN_HALLUCINATED = "open_domain_hallucinated"


class GoldenItem(BaseModel):
    id: str
    category: GoldenCategory
    question: str | None = None
    answer: str
    evidence: list[str]
    expected_verdict: Verdict
    notes: str = ""
    # Provenance for an item sourced from an external dataset, e.g.
    # "halueval-qa:1824" (HaluEval QA row 1824). `None` means self-authored
    # (every golden set A item).
    source: str | None = None


def load_golden_set(path: str | Path) -> list[GoldenItem]:
    """Parses and validates a golden-set YAML file.

    Raises `ValueError` on a duplicate `id` - every downstream consumer
    (the replay store, the CI gate's per-item lookup) assumes ids are
    unique, so a collision here should fail loudly at load time, not
    silently shadow one item's expected result with another's.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    items = [GoldenItem.model_validate(entry) for entry in raw]

    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        dupes = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
        raise ValueError(f"duplicate golden item ids: {dupes}")

    return items
