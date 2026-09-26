"""CLI entry point (Phase 7.4): `python -m halludetect.eval --suite golden
[--record] [--out report.json]`.

First `python -m` module CLI in this codebase - no prior `argparse` or
`[project.scripts]` precedent existed to follow, so this establishes the
convention rather than extending one.

Exit code is non-zero if a replay lookup is missing (see
`eval.runner.MissingReplayError`) or if the eval gate fails. The gate only
checks two *correctness invariants*, not calibration-quality judgment
calls: abstention recall (every item whose golden label says the pipeline
should abstain must actually get NOT_VERIFIABLE) and the quote-verification
rate (every SUPPORTED claim must carry a verified quote - true by
construction in detect/pipeline.py, checked here as a regression guard).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from halludetect.eval import metrics
from halludetect.eval.runner import MissingReplayError, run_suite


def _repo_root() -> Path:
    # src/halludetect/eval/__main__.py -> eval -> halludetect -> src -> repo root
    return Path(__file__).resolve().parents[3]


def _suites() -> dict[str, tuple[Path, Path]]:
    golden_dir = _repo_root() / "tests" / "data" / "golden"
    return {
        "golden": (golden_dir / "eval_set_a.yaml", golden_dir / "eval_set_a.replay.json"),
    }


def main(argv: list[str] | None = None) -> int:
    suites = _suites()
    parser = argparse.ArgumentParser(prog="python -m halludetect.eval")
    parser.add_argument("--suite", choices=sorted(suites), required=True)
    parser.add_argument(
        "--record",
        action="store_true",
        help="Call the real pipeline and refresh the replay file instead of replaying it.",
    )
    parser.add_argument("--out", default="eval_report.json", help="Where to write the full metrics report JSON.")
    args = parser.parse_args(argv)

    golden_path, replay_path = suites[args.suite]

    try:
        outcomes = run_suite(golden_path, replay_path, record=args.record)
    except MissingReplayError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report = metrics.build_report(outcomes)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"{len(outcomes)} golden items evaluated (suite={args.suite}, record={args.record})")
    for key in (
        "average_precision",
        "prevalence_baseline",
        "brier_score",
        "ece",
        "abstention_precision",
        "abstention_recall",
        "quote_verification_rate",
        "latency_ms_p50",
        "latency_ms_p95",
        "cost_usd_mean",
        "provider_disagreement_rate",
    ):
        print(f"  {key}: {report[key]}")
    print(f"full report written to {args.out}")

    gate_ok = report["abstention_recall"] in (None, 1.0) and report["quote_verification_rate"] in (None, 1.0)
    if not gate_ok:
        print(
            "error: eval gate failed - abstention_recall or quote_verification_rate is below 1.0",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
