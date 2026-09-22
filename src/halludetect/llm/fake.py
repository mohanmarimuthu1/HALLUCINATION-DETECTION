"""Deterministic LLMProvider test double - no network, no monkeypatching.

Used by router tests (and, later, Phase 6.4's fully-offline CI suite) to
script exact success/failure sequences without touching httpx.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from halludetect.llm.base import LLMResponse, TokenUsage
from halludetect.llm.exceptions import LLMError


class FakeProvider:
    """Replays `script` in order, one entry consumed per `complete()` call.

    Each entry is either an `LLMResponse` (returned) or an `LLMError`
    instance (raised). If the script runs out, the last entry repeats.
    """

    def __init__(
        self,
        script: Sequence[LLMResponse | LLMError],
        *,
        provider_name: str = "fake",
        model: str = "fake-model",
        supports_json_schema: bool = True,
    ):
        if not script:
            raise ValueError("FakeProvider needs at least one scripted entry")
        self._script = deque(script)
        self._last = script[-1]
        self._provider_name = provider_name
        self._model = model
        self._supports_json_schema = supports_json_schema
        self.call_count = 0

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> LLMResponse:
        self.call_count += 1
        entry = self._script.popleft() if self._script else self._last
        if isinstance(entry, LLMError):
            raise entry
        return entry

    def supports_json_schema(self) -> bool:
        return self._supports_json_schema


def fake_response(text: str = "ok", *, provider: str = "fake", model: str = "fake-model") -> LLMResponse:
    return LLMResponse(
        text=text,
        provider=provider,
        model=model,
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1),
    )
