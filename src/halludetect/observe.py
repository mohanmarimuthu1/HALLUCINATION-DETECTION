"""Per-model performance counters for `GET /v1/models/stats`.

Two roles are tracked separately: `answer` (the model wrote a `/v1/chat`
answer) and `verify` (the model ran claim extraction and verification).
For answers, the verdict the answer later received is recorded against the
model that wrote it, which is what makes the unsupported/contradicted rates
a per-model hallucination measure. The checking model's verdicts are kept
too: a checker that drops quotes turns good answers into NOT_ENOUGH_INFO,
and that shows up as a high unsupported rate on its own row, not only on
the answerers it checked.

In memory, per process: numbers reset on restart and are not shared
between instances (same limits as `llm.health.HealthTracker`).
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

from halludetect.detect.schemas import AnalysisResult, Verdict

LATENCY_WINDOW = 200


class Role(str, Enum):
    ANSWER = "answer"
    VERIFY = "verify"


@dataclass
class _RoleStats:
    calls: int = 0
    failures: int = 0
    rate_limited: int = 0
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=LATENCY_WINDOW))

    def snapshot(self) -> dict:
        latencies = sorted(self.latencies_ms)
        avg = sum(latencies) / len(latencies) if latencies else None
        p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else None
        return {
            "calls": self.calls,
            "failures": self.failures,
            "rate_limited": self.rate_limited,
            "avg_latency_ms": round(avg, 1) if avg is not None else None,
            "p95_latency_ms": round(p95, 1) if p95 is not None else None,
        }


@dataclass
class _ModelStats:
    answer: _RoleStats = field(default_factory=_RoleStats)
    verify: _RoleStats = field(default_factory=_RoleStats)
    verdicts: dict[Verdict, int] = field(default_factory=lambda: dict.fromkeys(Verdict, 0))
    groundedness_sum: float = 0.0
    check_verdicts: dict[Verdict, int] = field(default_factory=lambda: dict.fromkeys(Verdict, 0))
    last_error: str | None = None
    last_used: datetime | None = None


_SCORED = (Verdict.GROUNDED, Verdict.CONTRADICTED, Verdict.NOT_ENOUGH_INFO)


def _verdict_summary(verdicts: dict[Verdict, int]) -> dict:
    scored = sum(verdicts[v] for v in _SCORED)
    contradicted = verdicts[Verdict.CONTRADICTED]
    unsupported = contradicted + verdicts[Verdict.NOT_ENOUGH_INFO]
    return {
        "verdicts": {v.value: n for v, n in verdicts.items()},
        "contradicted_rate": round(contradicted / scored, 3) if scored else None,
        "unsupported_rate": round(unsupported / scored, 3) if scored else None,
    }


def _iso(value: datetime | None) -> str | None:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ") if value is not None else None


class ModelObserver:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[tuple[str, str], _ModelStats] = {}
        self.since = datetime.now(UTC)

    def _get(self, provider: str, model: str) -> _ModelStats:
        return self._models.setdefault((provider, model), _ModelStats())

    def _role(self, stats: _ModelStats, role: Role) -> _RoleStats:
        return stats.answer if role == Role.ANSWER else stats.verify

    def record_success(self, provider: str, model: str, role: Role, latency_ms: float) -> None:
        with self._lock:
            stats = self._get(provider, model)
            role_stats = self._role(stats, role)
            role_stats.calls += 1
            role_stats.latencies_ms.append(latency_ms)
            stats.last_used = datetime.now(UTC)

    def record_failure(
        self, provider: str, model: str, role: Role, error: str, *, rate_limited: bool = False
    ) -> None:
        with self._lock:
            stats = self._get(provider, model)
            role_stats = self._role(stats, role)
            role_stats.calls += 1
            role_stats.failures += 1
            if rate_limited:
                role_stats.rate_limited += 1
            stats.last_error = error[:300]
            stats.last_used = datetime.now(UTC)

    def record_answer_verdict(self, provider: str, model: str, result: AnalysisResult) -> None:
        with self._lock:
            stats = self._get(provider, model)
            stats.verdicts[result.verdict] += 1
            if result.verdict in _SCORED:
                stats.groundedness_sum += result.groundedness

    def record_check_verdict(self, provider: str, model: str, result: AnalysisResult) -> None:
        with self._lock:
            self._get(provider, model).check_verdicts[result.verdict] += 1

    def snapshot(self, in_cooldown: dict[tuple[str, str], bool] | None = None) -> dict:
        in_cooldown = in_cooldown or {}
        with self._lock:
            rows = []
            for (provider, model), stats in self._models.items():
                scored = sum(stats.verdicts[v] for v in _SCORED)
                answer = {
                    **stats.answer.snapshot(),
                    **_verdict_summary(stats.verdicts),
                    "mean_groundedness": round(stats.groundedness_sum / scored, 3) if scored else None,
                }
                rows.append(
                    {
                        "provider": provider,
                        "model": model,
                        "answer": answer,
                        "verify": {**stats.verify.snapshot(), **_verdict_summary(stats.check_verdicts)},
                        "in_cooldown": in_cooldown.get((provider, model), False),
                        "last_error": stats.last_error,
                        "last_used": _iso(stats.last_used),
                    }
                )
            return {"since": _iso(self.since), "models": rows}

    def reset(self) -> None:
        with self._lock:
            self._models.clear()
            self.since = datetime.now(UTC)


observer = ModelObserver()
