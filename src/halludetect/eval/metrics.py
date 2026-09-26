"""Calibration & quality metrics for golden-set eval (Phase 7.3).

Dependency-free by design: no numpy/scikit-learn added to `pyproject.toml`
for what is an eval-only module - every function here is small enough to
implement directly, matching `detect/fuse.py`'s own from-scratch
`wilson_ci` style. Never reports a bare "accuracy" (plan.md's explicit
instruction) - `average_precision` is always reported alongside
`prevalence_baseline`, its trivial-baseline comparison point.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from halludetect.detect.schemas import Label, Verdict

if TYPE_CHECKING:
    from halludetect.eval.runner import EvalOutcome

_HALLUCINATED_VERDICTS = {Verdict.CONTRADICTED, Verdict.NOT_ENOUGH_INFO, Verdict.NOT_VERIFIABLE}


def is_hallucinated_ground_truth(expected: Verdict) -> bool:
    return expected in _HALLUCINATED_VERDICTS


def average_precision(scores: list[float], labels: list[bool]) -> float | None:
    """Step-interpolated average precision - the standard practical PR-AUC
    proxy (what `sklearn.metrics.average_precision_score` computes).
    `None` (not `0.0`) when there are no positive examples: the metric is
    undefined in that case, not zero.
    """
    n_pos = sum(labels)
    if n_pos == 0:
        return None

    pairs = sorted(zip(scores, labels, strict=True), key=lambda pair: -pair[0])
    tp = fp = 0
    ap = 0.0
    prev_recall = 0.0
    for _, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


def prevalence(labels: list[bool]) -> float:
    """Expected average precision of a random ranking - the baseline
    `average_precision` must be compared against (plan.md: "PR-AUC vs
    prevalence baseline").
    """
    return sum(labels) / len(labels) if labels else 0.0


def brier_score(scores: list[float], labels: list[bool]) -> float:
    if not scores:
        return 0.0
    return sum((s - (1.0 if label else 0.0)) ** 2 for s, label in zip(scores, labels, strict=True)) / len(scores)


def expected_calibration_error(
    scores: list[float], labels: list[bool], *, n_bins: int = 10
) -> tuple[float, list[dict[str, object]]]:
    """Returns `(ece, reliability_table)`. The table is structured data -
    bin range, count, mean predicted score, observed positive rate - not a
    rendered image, so no plotting dependency is needed to consume it.
    """
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for score, label in zip(scores, labels, strict=True):
        idx = min(int(score * n_bins), n_bins - 1)
        buckets[idx].append((score, label))

    total = len(scores)
    ece = 0.0
    table: list[dict[str, object]] = []
    for i, bucket in enumerate(buckets):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if not bucket:
            table.append({"bin": [lo, hi], "count": 0, "avg_predicted": None, "observed_rate": None})
            continue
        avg_predicted = sum(s for s, _ in bucket) / len(bucket)
        observed_rate = sum(1 for _, label in bucket if label) / len(bucket)
        ece += (len(bucket) / total) * abs(avg_predicted - observed_rate)
        table.append(
            {
                "bin": [lo, hi],
                "count": len(bucket),
                "avg_predicted": avg_predicted,
                "observed_rate": observed_rate,
            }
        )
    return ece, table


def precision_recall(predicted: list[bool], actual: list[bool]) -> tuple[float | None, float | None]:
    """Precision/recall over two boolean sequences - used for abstention
    scoring below. `None` (not `0.0`) when a denominator is 0: undefined,
    not zero.
    """
    tp = sum(1 for p, a in zip(predicted, actual, strict=True) if p and a)
    fp = sum(1 for p, a in zip(predicted, actual, strict=True) if p and not a)
    fn = sum(1 for p, a in zip(predicted, actual, strict=True) if not p and a)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return precision, recall


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)


def quote_verification_rate(outcomes: list[EvalOutcome]) -> float | None:
    """Fraction of `SUPPORTED`-labeled claims with `quote_verified: true`.
    Should always be 1.0 by construction - `detect.pipeline`'s
    `_verify_and_ground` downgrades any unverified SUPPORTED claim before
    it's ever returned - so this is reported as a health-check invariant,
    not assumed to hold without checking.
    """
    supported = [claim for outcome in outcomes for claim in outcome.result.claims if claim.label == Label.SUPPORTED]
    if not supported:
        return None
    return sum(1 for claim in supported if claim.quote_verified) / len(supported)


def provider_disagreement_rate(outcomes: list[EvalOutcome]) -> float | None:
    """Fraction of items where distinct recorded models disagreed on the
    verdict. Requires >=2 distinct `model_used.model` values recorded for
    the *same* item - this pass's `eval.runner` records one pinned model
    per run (see its module docstring), so this is `None` (explicitly
    skipped, not silently omitted) until a future pass records multiple
    providers per item.
    """
    models = {outcome.result.model_used.model for outcome in outcomes}
    if len(models) < 2:
        return None
    return None  # placeholder: no per-item multi-provider recordings exist yet.


def build_report(outcomes: list[EvalOutcome]) -> dict[str, object]:
    labels = [is_hallucinated_ground_truth(outcome.item.expected_verdict) for outcome in outcomes]
    scores = [outcome.result.p_hallucinated for outcome in outcomes]

    ece, reliability_table = expected_calibration_error(scores, labels)

    predicted_abstain = [outcome.result.verdict == Verdict.NOT_VERIFIABLE for outcome in outcomes]
    actual_abstain = [outcome.item.expected_verdict == Verdict.NOT_VERIFIABLE for outcome in outcomes]
    abstention_precision, abstention_recall = precision_recall(predicted_abstain, actual_abstain)

    latencies = [float(outcome.result.timings_ms.total) for outcome in outcomes]
    costs = [outcome.result.cost_usd for outcome in outcomes]

    return {
        "n_items": len(outcomes),
        "average_precision": average_precision(scores, labels),
        "prevalence_baseline": prevalence(labels),
        "brier_score": brier_score(scores, labels),
        "ece": ece,
        "reliability_table": reliability_table,
        "abstention_precision": abstention_precision,
        "abstention_recall": abstention_recall,
        "quote_verification_rate": quote_verification_rate(outcomes),
        "latency_ms_p50": percentile(latencies, 0.5),
        "latency_ms_p95": percentile(latencies, 0.95),
        "cost_usd_mean": sum(costs) / len(costs) if costs else 0.0,
        "provider_disagreement_rate": provider_disagreement_rate(outcomes),
    }
