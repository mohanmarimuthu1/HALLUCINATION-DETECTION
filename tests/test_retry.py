"""Jittered backoff retry tests (Phase 6.2) - offline, no real sleeping.

`sleep`/`jitter` are injected so these tests run instantly and
deterministically instead of depending on real timing or randomness.
"""
import pytest

from halludetect.llm.exceptions import LLMAuthError, LLMRateLimitError, LLMResponseError, LLMTimeoutError
from halludetect.llm.fake import FakeProvider, fake_response
from halludetect.llm.retry import RetryingProvider


def _retrying(provider, **overrides):
    recorded_sleeps = []
    defaults = {
        "provider": provider,
        "sleep": lambda s: recorded_sleeps.append(s),
        "jitter": lambda lo, hi: hi,  # deterministic: always the upper bound
    }
    defaults.update(overrides)
    return RetryingProvider(**defaults), recorded_sleeps


def test_succeeds_first_try_without_retrying():
    provider = FakeProvider([fake_response("ok")])
    retrying, sleeps = _retrying(provider)
    result = retrying.complete("hi")
    assert result.text == "ok"
    assert provider.call_count == 1
    assert sleeps == []


def test_retries_on_timeout_then_succeeds():
    provider = FakeProvider([LLMTimeoutError("timed out"), fake_response("ok")])
    retrying, sleeps = _retrying(provider)
    result = retrying.complete("hi")
    assert result.text == "ok"
    assert provider.call_count == 2
    assert len(sleeps) == 1


def test_retries_on_rate_limit_then_succeeds():
    provider = FakeProvider([LLMRateLimitError("quota"), fake_response("ok")])
    retrying, sleeps = _retrying(provider)
    result = retrying.complete("hi")
    assert result.text == "ok"
    assert provider.call_count == 2


def test_gives_up_after_max_attempts():
    provider = FakeProvider([LLMTimeoutError("timed out")] * 5)
    retrying, sleeps = _retrying(provider, max_attempts=3)
    with pytest.raises(LLMTimeoutError):
        retrying.complete("hi")
    assert provider.call_count == 3
    assert len(sleeps) == 2  # a sleep happens between attempts, not after the final one


def test_auth_error_is_never_retried():
    provider = FakeProvider([LLMAuthError("bad key"), fake_response("should not be reached")])
    retrying, sleeps = _retrying(provider)
    with pytest.raises(LLMAuthError):
        retrying.complete("hi")
    assert provider.call_count == 1
    assert sleeps == []


def test_response_error_is_never_retried():
    provider = FakeProvider([LLMResponseError("malformed"), fake_response("should not be reached")])
    retrying, sleeps = _retrying(provider)
    with pytest.raises(LLMResponseError):
        retrying.complete("hi")
    assert provider.call_count == 1
    assert sleeps == []


def test_backoff_delay_grows_and_is_capped():
    provider = FakeProvider([LLMTimeoutError("t")] * 5)
    recorded_delays = []
    retrying = RetryingProvider(
        provider,
        max_attempts=4,
        base_delay_s=1.0,
        max_delay_s=3.0,
        sleep=lambda s: None,
        jitter=lambda lo, hi: recorded_delays.append(hi) or hi,
    )
    with pytest.raises(LLMTimeoutError):
        retrying.complete("hi")
    # base_delay_s * 2**0, 2**1, 2**2 = 1.0, 2.0, 4.0 - capped at max_delay_s=3.0
    assert recorded_delays == [1.0, 2.0, 3.0]


def test_supports_json_schema_delegates_to_wrapped_provider():
    retrying, _ = _retrying(FakeProvider([fake_response("x")], supports_json_schema=False))
    assert retrying.supports_json_schema() is False
