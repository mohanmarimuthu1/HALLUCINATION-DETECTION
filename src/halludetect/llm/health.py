"""Per-model health tracking and circuit breaking for the free-model pool.

The router (router.py) asks this module which free models are currently
worth trying and in what order - it never inspects raw success/failure
counts itself. Keeping the ranking and cooldown logic here means the
breaker rule (Phase 2.5) is defined in exactly one place.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

FAILURE_THRESHOLD = 3
COOLDOWN_S = 5 * 60


@dataclass
class ModelHealth:
    attempts: int = 0
    successes: int = 0
    total_latency_ms: float = 0.0
    consecutive_failures: int = 0
    cooldown_until: float | None = None
    last_rate_limited_at: float | None = None

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.successes if self.successes else float("inf")


class HealthTracker:
    """Rolling health table, one `ModelHealth` per model id.

    Demotion rule (2.5): `FAILURE_THRESHOLD` consecutive failures puts a
    model in cooldown for `COOLDOWN_S`; a single success anywhere resets
    the consecutive-failure counter for that model. Cooldown is
    time-based, not a manual reset, so a model recovers on its own once
    it starts working again.
    """

    def __init__(self, *, failure_threshold: int = FAILURE_THRESHOLD, cooldown_s: float = COOLDOWN_S):
        self._health: dict[str, ModelHealth] = {}
        self._failure_threshold = failure_threshold
        self._cooldown_s = cooldown_s

    def _get(self, model: str) -> ModelHealth:
        return self._health.setdefault(model, ModelHealth())

    def record_success(self, model: str, latency_ms: float) -> None:
        health = self._get(model)
        health.attempts += 1
        health.successes += 1
        health.total_latency_ms += latency_ms
        health.consecutive_failures = 0
        health.cooldown_until = None

    def record_failure(self, model: str, *, rate_limited: bool = False, now: float | None = None) -> None:
        now = now if now is not None else time.monotonic()
        health = self._get(model)
        health.attempts += 1
        health.consecutive_failures += 1
        if rate_limited:
            health.last_rate_limited_at = now
        if health.consecutive_failures >= self._failure_threshold:
            health.cooldown_until = now + self._cooldown_s

    def is_available(self, model: str, *, now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        health = self._health.get(model)
        if health is None or health.cooldown_until is None:
            return True
        return now >= health.cooldown_until

    def get(self, model: str) -> ModelHealth:
        return self._get(model)

    def rank_available(self, models: list[str], *, now: float | None = None) -> list[str]:
        """Available models (not in active cooldown), best first: higher
        success rate, then lower average latency, then untried models
        (given a chance before ranking data exists).
        """
        now = now if now is not None else time.monotonic()
        available = [m for m in models if self.is_available(m, now=now)]

        def sort_key(model: str) -> tuple[float, float]:
            health = self._health.get(model)
            if health is None or health.attempts == 0:
                return (-1.0, 0.0)
            return (-health.success_rate, health.avg_latency_ms)

        return sorted(available, key=sort_key)
