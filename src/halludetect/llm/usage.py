"""Token-usage accounting wrapper (Phase 5.3).

`detect.pipeline.run()` makes two or more `LLMProvider.complete()` calls
per request - claim extraction, then verification, plus however many
repair retries `complete_structured` needed for each - and none of
`claims.py`/`verify.py`/`structured.py` know or care about cost tracking.
Rather than threading a usage accumulator through all three, this wraps
the single `LLMProvider` the pipeline already receives so every call made
*through it* is counted, transparently, without changing any of their
signatures.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from halludetect.llm.base import LLMProvider, LLMResponse, TokenUsage


@dataclass
class UsageTrackingProvider:
    provider: LLMProvider
    _prompt_tokens: int = field(default=0, init=False)
    _completion_tokens: int = field(default=0, init=False)
    # None once any wrapped call didn't report cost_usd directly - at that
    # point a running sum would understate the true cost for the calls that
    # weren't tracked, which is worse than admitting "no exact total" and
    # falling back to a token-based estimate over everything.
    _reported_cost_usd: float | None = field(default=0.0, init=False)

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> LLMResponse:
        response = self.provider.complete(prompt, max_tokens=max_tokens)
        self._prompt_tokens += response.usage.prompt_tokens
        self._completion_tokens += response.usage.completion_tokens
        if self._reported_cost_usd is not None and response.usage.cost_usd is not None:
            self._reported_cost_usd += response.usage.cost_usd
        else:
            self._reported_cost_usd = None
        return response

    def supports_json_schema(self) -> bool:
        return self.provider.supports_json_schema()

    @property
    def usage(self) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            cost_usd=self._reported_cost_usd,
        )
