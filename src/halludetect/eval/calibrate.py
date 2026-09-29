"""Fits `detect.fuse.P_HALLUCINATED_BY_VERDICT` from the committed golden
replay files: `python -m halludetect.eval.calibrate`.

For each verdict the pipeline can reach with enough claims to score
(GROUNDED, NOT_ENOUGH_INFO, CONTRADICTED), the fitted value is the
Laplace-smoothed share of golden items with that verdict whose expected
verdict counts as hallucinated. Offline and deterministic: it reads only
replayed results, so the output only changes when the golden sets do.
"""
from __future__ import annotations

from pathlib import Path

from halludetect.detect.fuse import MIN_VERIFIABLE_CLAIMS, fuse, summarize
from halludetect.detect.schemas import Verdict
from halludetect.eval.metrics import is_hallucinated_ground_truth
from halludetect.eval.runner import run_suite

_SCORED_VERDICTS = (Verdict.GROUNDED, Verdict.NOT_ENOUGH_INFO, Verdict.CONTRADICTED)


def _golden_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "tests" / "data" / "golden"


def fit(suites: tuple[str, ...] = ("a", "b")) -> dict[Verdict, tuple[float, int]]:
    """Returns `{verdict: (fitted_rate, n_items)}`, rate rounded to 3 places."""
    counts = {v: [0, 0] for v in _SCORED_VERDICTS}
    for suite in suites:
        golden = _golden_dir() / f"eval_set_{suite}.yaml"
        replay = _golden_dir() / f"eval_set_{suite}.replay.json"
        for outcome in run_suite(golden, replay, record=False):
            signals = summarize(outcome.result.claims)
            if signals.n_verifiable_claims < MIN_VERIFIABLE_CLAIMS:
                continue
            verdict, *_ = fuse(signals)
            counts[verdict][0] += is_hallucinated_ground_truth(outcome.item.expected_verdict)
            counts[verdict][1] += 1
    return {v: (round((pos + 1) / (n + 2), 3), n) for v, (pos, n) in counts.items()}


def main() -> int:
    for verdict, (rate, n) in fit().items():
        print(f"{verdict.value}: {rate}  (n={n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
