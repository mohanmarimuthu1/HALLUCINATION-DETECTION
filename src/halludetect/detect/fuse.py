"""Calibrated scoring (Phase 4.4).

`p_hallucinated` is the observed rate of hallucinated answers among
golden-set items that received the same verdict, fitted by
`halludetect.eval.calibrate` from the committed replay files (sets A and
B, Laplace-smoothed). Fitting per verdict beat both the old claim-fraction
heuristic and a logistic model on held-out ECE when fitted on one set and
scored on the other. Re-run the fit and bump `CALIBRATION_VERSION`
whenever the golden sets or the verdict logic change.

`groundedness` stays the raw supported fraction, with its Wilson interval.

`n_verifiable_claims < 3` forces NOT_VERIFIABLE regardless of the label
mix (docs/contract.md rule 1) - there is deliberately no code path that
can produce a bare percentage for two or fewer verifiable claims.
"""
from __future__ import annotations

import math

from halludetect.detect.schemas import AbstentionReason, AnalysisResult, ClaimResult, Label, Signals, Verdict

CALIBRATION_VERSION = "verdict-rate-v1"
MIN_VERIFIABLE_CLAIMS = 3

P_HALLUCINATED_BY_VERDICT: dict[Verdict, float] = {
    Verdict.GROUNDED: 0.043,
    Verdict.NOT_ENOUGH_INFO: 0.660,
    Verdict.CONTRADICTED: 0.976,
}


def wilson_ci(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval for a binomial proportion. Returns
    (0.0, 0.0) for n == 0 rather than dividing by zero or fabricating an
    interval that implies more information than exists.
    """
    if n == 0:
        return (0.0, 0.0)

    phat = successes / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    low = (center - margin) / denom
    high = (center + margin) / denom
    return (max(0.0, low), min(1.0, high))


def summarize(claims: list[ClaimResult]) -> Signals:
    supported = sum(1 for c in claims if c.label == Label.SUPPORTED)
    contradicted = sum(1 for c in claims if c.label == Label.CONTRADICTED)
    not_enough_info = sum(1 for c in claims if c.label == Label.NOT_ENOUGH_INFO)
    return Signals(
        n_verifiable_claims=len(claims),
        supported=supported,
        contradicted=contradicted,
        not_enough_info=not_enough_info,
    )


def fuse(signals: Signals) -> tuple[Verdict, float, float, tuple[float, float]]:
    """Returns (verdict, p_hallucinated, groundedness, groundedness_ci)."""
    n = signals.n_verifiable_claims
    groundedness = signals.supported / n if n else 0.0
    ci = wilson_ci(signals.supported, n)

    if n < MIN_VERIFIABLE_CLAIMS:
        # Too few claims to score; the verdict says so and no rate is fitted for it.
        p_hallucinated = (signals.contradicted + signals.not_enough_info) / n if n else 1.0
        return Verdict.NOT_VERIFIABLE, p_hallucinated, groundedness, ci
    if signals.contradicted > 0:
        verdict = Verdict.CONTRADICTED
    elif signals.not_enough_info > 0:
        verdict = Verdict.NOT_ENOUGH_INFO
    else:
        verdict = Verdict.GROUNDED
    return verdict, P_HALLUCINATED_BY_VERDICT[verdict], groundedness, ci


def rescore(result: AnalysisResult) -> AnalysisResult:
    """Re-derives the scored fields from `result.claims` under the current
    calibration. Scoring is deterministic given the claims, so a result
    recorded or cached under an older `calibration_version` can be brought
    current without another LLM call.
    """
    if result.calibration_version == CALIBRATION_VERSION or result.reason == AbstentionReason.NO_EVIDENCE_CONFIGURED:
        return result
    verdict, p_hallucinated, groundedness, ci = fuse(summarize(result.claims))
    return result.model_copy(
        update={
            "verdict": verdict,
            "p_hallucinated": p_hallucinated,
            "groundedness": groundedness,
            "groundedness_ci": ci,
            "calibration_version": CALIBRATION_VERSION,
            "reason": AbstentionReason.INSUFFICIENT_VERIFIABLE_CLAIMS if verdict == Verdict.NOT_VERIFIABLE else None,
        }
    )
