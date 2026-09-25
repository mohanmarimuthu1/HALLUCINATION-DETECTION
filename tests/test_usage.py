"""UsageTrackingProvider tests (Phase 5.3) - offline, FakeProvider scripted
responses stand in for real calls."""
from halludetect.llm.fake import FakeProvider, fake_response
from halludetect.llm.usage import UsageTrackingProvider


def test_accumulates_tokens_across_multiple_calls():
    provider = FakeProvider(
        [
            fake_response("a", provider="fake", model="m"),
            fake_response("b", provider="fake", model="m"),
        ]
    )
    tracked = UsageTrackingProvider(provider)
    tracked.complete("first")
    tracked.complete("second")
    assert tracked.usage.prompt_tokens == 2
    assert tracked.usage.completion_tokens == 2


def test_no_calls_means_zero_confidently_known_cost():
    tracked = UsageTrackingProvider(FakeProvider([fake_response("unused")]))
    assert tracked.usage.prompt_tokens == 0
    assert tracked.usage.cost_usd == 0.0


def test_reported_cost_sums_when_every_call_reports_it():
    from halludetect.llm.base import LLMResponse, TokenUsage

    class _Scripted:
        def __init__(self, responses):
            self._responses = iter(responses)

        def complete(self, prompt, *, max_tokens=1024):
            return next(self._responses)

        def supports_json_schema(self):
            return True

    responses = [
        LLMResponse(text="a", provider="openrouter", model="m", usage=TokenUsage(1, 1, cost_usd=0.01)),
        LLMResponse(text="b", provider="openrouter", model="m", usage=TokenUsage(1, 1, cost_usd=0.02)),
    ]
    tracked = UsageTrackingProvider(_Scripted(responses))
    tracked.complete("first")
    tracked.complete("second")
    assert tracked.usage.cost_usd == 0.03


def test_reported_cost_is_none_if_any_call_did_not_report_it():
    from halludetect.llm.base import LLMResponse, TokenUsage

    class _Scripted:
        def __init__(self, responses):
            self._responses = iter(responses)

        def complete(self, prompt, *, max_tokens=1024):
            return next(self._responses)

        def supports_json_schema(self):
            return True

    responses = [
        LLMResponse(text="a", provider="openai", model="m", usage=TokenUsage(1, 1, cost_usd=0.01)),
        LLMResponse(text="b", provider="openai", model="m", usage=TokenUsage(1, 1, cost_usd=None)),
    ]
    tracked = UsageTrackingProvider(_Scripted(responses))
    tracked.complete("first")
    tracked.complete("second")
    assert tracked.usage.cost_usd is None


def test_supports_json_schema_delegates_to_wrapped_provider():
    tracked = UsageTrackingProvider(FakeProvider([fake_response("x")], supports_json_schema=False))
    assert tracked.supports_json_schema() is False
