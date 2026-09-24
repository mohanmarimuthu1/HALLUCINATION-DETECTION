"""Calibrated scoring (Phase 4.4).

`p_hallucinated`/`groundedness` are derived from a small, documented
signal vector (per-label counts), not a hardcoded floor and not a fitted
model - no labeled calibration set exists yet (that's Phase 7's golden
sets). `calibration_version` names this heuristic explicitly so a real
fitted model can replace it later without silently changing existing
callers' interpretation of the score.

`n_verifiable_claims < 3` forces NOT_VERIFIABLE regardless of the label
mix (docs/contract.md rule 1) - there is deliberately no code path that
can produce a bare percentage for two or fewer verifiable claims.
"""
from __future__ import annotations

import math

from halludetect.detect.schemas import ClaimResult, Label, Signals, Verdict

CALIBRATION_VERSION = "heuristic-v0"
MIN_VERIFIABLE_CLAIMS = 3


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
    p_hallucinated = (signals.contradicted + signals.not_enough_info) / n if n else 1.0

    if n < MIN_VERIFIABLE_CLAIMS:
        return Verdict.NOT_VERIFIABLE, p_hallucinated, groundedness, ci
    if signals.contradicted > 0:
        return Verdict.CONTRADICTED, p_hallucinated, groundedness, ci
    if signals.not_enough_info > 0:
        return Verdict.NOT_ENOUGH_INFO, p_hallucinated, groundedness, ci
    return Verdict.GROUNDED, p_hallucinated, groundedness, ci
