"""Jittered backoff retry wrapper (Phase 6.2).

Wraps any `LLMProvider` so a transient failure gets retried with jittered
exponential backoff before giving up, instead of failing the whole
request on the first hiccup. Only `LLMTimeoutError` and `LLMRateLimitError`
are retried - per `llm/exceptions.py`'s own classification, both mean "the
call didn't go through, try again, maybe after a pause," which retrying
can actually fix. `LLMAuthError` (a bad key won't fix itself) and
`LLMResponseError`/`LLMSchemaValidationError` (the same malformed input
would just fail the same way again) propagate immediately - classified by
exception type per `plan.md`'s own wording for this phase, never by
inspecting an error message string.

Known, documented gap: a raw HTTP 5xx currently surfaces as the generic
`LLMResponseError` (`llm/_http.py:raise_for_provider_error`), not a
distinct retryable type, so a transient server error on the provider's
side is not retried here even though it likely should be. Splitting that
out was judged out of scope for this phase's "S" sizing; see process.md.
"""
from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass

from halludetect.llm.base import LLMProvider, LLMResponse
from halludetect.llm.exceptions import LLMRateLimitError, LLMTimeoutError

_RETRYABLE = (LLMTimeoutError, LLMRateLimitError)


@dataclass
class RetryingProvider:
    provider: LLMProvider
    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    # Injectable so tests never actually sleep or depend on real randomness.
    sleep: Callable[[float], None] = time.sleep
    jitter: Callable[[float, float], float] = random.uniform

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> LLMResponse:
        attempt = 0
        while True:
            try:
                return self.provider.complete(prompt, max_tokens=max_tokens)
            except _RETRYABLE:
                attempt += 1
                if attempt >= self.max_attempts:
                    raise
                delay = min(self.max_delay_s, self.base_delay_s * (2 ** (attempt - 1)))
                self.sleep(self.jitter(0, delay))

    def supports_json_schema(self) -> bool:
        return self.provider.supports_json_schema()
