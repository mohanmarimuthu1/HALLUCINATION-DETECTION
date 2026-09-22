"""Router chain (Phase 2.3): pinned model -> next healthy free model ->
user's paid key -> explicit fail.

The chain never guesses when every option is exhausted - it raises
`LLMResponseError` with the list of what was tried and why each failed.
Silently returning something (or falling back to model knowledge with no
evidence) is the exact class of bug docs/contract.md and CLAUDE.md both
call out as non-negotiable to avoid, and that applies here too: a caller
must see an explicit failure, not a guess dressed up as a result.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic

from halludetect.llm.base import LLMProvider, LLMResponse
from halludetect.llm.exceptions import LLMError, LLMRateLimitError, LLMResponseError
from halludetect.llm.health import HealthTracker
from halludetect.llm.openrouter import FreeModelCatalog

ProviderFactory = Callable[[str], LLMProvider]


@dataclass
class Router:
    catalog: FreeModelCatalog
    provider_factory: ProviderFactory
    health: HealthTracker = field(default_factory=HealthTracker)
    pinned_model: str | None = None
    user_provider: LLMProvider | None = None
    max_free_models_tried: int = 5

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> LLMResponse:
        tried: set[str] = set()
        errors: list[str] = []

        if self.pinned_model:
            result = self._try_model(self.pinned_model, prompt, max_tokens, errors)
            if result is not None:
                return result
            tried.add(self.pinned_model)

        try:
            free_models = self.catalog.get_models()
        except LLMError as exc:
            errors.append(f"free-model catalog fetch failed: {exc}")
            free_models = []

        ranked = [m for m in self.health.rank_available(free_models) if m not in tried]
        for model in ranked[: self.max_free_models_tried]:
            result = self._try_model(model, prompt, max_tokens, errors)
            if result is not None:
                return result
            tried.add(model)

        if self.user_provider is not None:
            try:
                return self.user_provider.complete(prompt, max_tokens=max_tokens)
            except LLMError as exc:
                errors.append(f"user-provided key failed: {exc}")

        reason = "; ".join(errors) if errors else "no pinned model, free pool, or user key configured"
        raise LLMResponseError(f"router: exhausted all options - {reason}")

    def _try_model(self, model: str, prompt: str, max_tokens: int, errors: list[str]) -> LLMResponse | None:
        provider = self.provider_factory(model)
        start = monotonic()
        try:
            response = provider.complete(prompt, max_tokens=max_tokens)
        except LLMError as exc:
            self.health.record_failure(model, rate_limited=isinstance(exc, LLMRateLimitError))
            errors.append(f"{model}: {exc}")
            return None
        self.health.record_success(model, (monotonic() - start) * 1000)
        return response
