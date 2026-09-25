"""Cost estimation tests (Phase 5.3) - offline."""
from halludetect.llm.base import TokenUsage
from halludetect.llm.pricing import estimate_cost_usd


def test_reported_cost_is_used_directly_when_present():
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=1000, cost_usd=0.0042)
    assert estimate_cost_usd("openrouter", "any/model", usage) == 0.0042


def test_reported_zero_cost_is_trusted_not_overridden():
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=1000, cost_usd=0.0)
    assert estimate_cost_usd("openrouter", "free/model", usage) == 0.0


def test_falls_back_to_rate_table_when_cost_not_reported():
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=1000, cost_usd=None)
    cost = estimate_cost_usd("openai", "gpt-4o-mini", usage)
    assert cost == 0.00015 + 0.0006


def test_unknown_provider_model_defaults_to_zero():
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=1000, cost_usd=None)
    assert estimate_cost_usd("custom", "some-local-model", usage) == 0.0
