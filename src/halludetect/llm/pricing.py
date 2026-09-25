"""Per-request USD cost estimation (Phase 5.3).

`AnalysisResult.cost_usd` (docs/contract.md) needs a real number, not
always `0.0`. Preference order:

1. `TokenUsage.cost_usd` if the provider reported it directly - OpenRouter
   does, per call, in its response payload (confirmed live). This is exact,
   not an estimate.
2. A static per-1K-token rate table, keyed by (provider, model), for
   providers that don't report cost in the response at all (OpenAI,
   Gemini, Anthropic, a generic OpenAI-compatible endpoint).
3. `0.0` for anything not in the table - most likely a self-hosted or
   otherwise free-to-this-deployment endpoint behind `custom`. Guessing a
   *cost* wrong is a low-stakes failure mode compared to guessing a
   *verdict* wrong; this table is deliberately not treated as
   authoritative pricing and should be kept current by whoever owns
   billing, not trusted blindly for invoicing.

Rates last checked against each provider's public pricing page at the time
they were added; they will drift as providers change prices; not the
`docs/contract.md` non-negotiable claims this project is otherwise built
around, so no phase gate protects them.
"""
from __future__ import annotations

from halludetect.llm.base import TokenUsage

# (provider, model) -> (usd_per_1k_prompt_tokens, usd_per_1k_completion_tokens)
_RATES_PER_1K_TOKENS: dict[tuple[str, str], tuple[float, float]] = {
    ("openai", "gpt-4o-mini"): (0.00015, 0.0006),
    ("gemini", "gemini-2.0-flash"): (0.0, 0.0),
    ("anthropic", "claude-3-5-haiku-latest"): (0.0008, 0.004),
}


def estimate_cost_usd(provider: str, model: str, usage: TokenUsage) -> float:
    if usage.cost_usd is not None:
        return usage.cost_usd

    prompt_rate, completion_rate = _RATES_PER_1K_TOKENS.get((provider, model), (0.0, 0.0))
    return (usage.prompt_tokens / 1000) * prompt_rate + (usage.completion_tokens / 1000) * completion_rate
