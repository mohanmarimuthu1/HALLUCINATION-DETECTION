"""Provider tests, run fully offline by monkeypatching httpx.stream - no live
API keys required. Covers: successful parse, missing-key auth error,
401/403/429 classification, and timeout classification, per provider.
"""
import contextlib

import httpx
import pytest

from halludetect.llm import _http, anthropic, custom_openai_compat, gemini, nvidia, openai, openrouter
from halludetect.llm.exceptions import (
    LLMAuthError,
    LLMModelAccessError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)


def fake_response(status_code: int, json_body: dict) -> httpx.Response:
    request = httpx.Request("POST", "https://example.invalid")
    return httpx.Response(status_code, json=json_body, request=request)


def patch_post(monkeypatch, post) -> None:
    """Serves `post(url, **kwargs)`'s response through `httpx.stream`,
    which is what every provider's chat call goes through.
    """

    @contextlib.contextmanager
    def stream(method, url, **kwargs):
        r = post(url, **kwargs)
        yield httpx.Response(r.status_code, headers=r.headers, stream=httpx.ByteStream(r.content), request=r.request)

    monkeypatch.setattr(httpx, "stream", stream)


# ---- OpenAI ----

def test_openai_success(monkeypatch):
    patch_post(
        monkeypatch,
        lambda *a, **k: fake_response(
            200,
            {"choices": [{"message": {"content": "hello"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
        ),
    )
    provider = openai.OpenAIProvider(api_key="key")
    result = provider.complete("hi")
    assert result.text == "hello"
    assert result.provider == "openai"
    assert result.usage.prompt_tokens == 5
    assert result.usage.completion_tokens == 2


def test_openai_no_key_raises_auth_error():
    provider = openai.OpenAIProvider(api_key=None)
    with pytest.raises(LLMAuthError):
        provider.complete("hi")


def test_openai_401_raises_auth_error(monkeypatch):
    patch_post(monkeypatch, lambda *a, **k: fake_response(401, {"error": "bad key"}))
    provider = openai.OpenAIProvider(api_key="key")
    with pytest.raises(LLMAuthError):
        provider.complete("hi")


def test_openai_403_raises_model_access_error(monkeypatch):
    patch_post(monkeypatch, lambda *a, **k: fake_response(403, {"error": "not allowed"}))
    provider = openai.OpenAIProvider(api_key="key")
    with pytest.raises(LLMModelAccessError):
        provider.complete("hi")


def test_openai_429_raises_rate_limit_error(monkeypatch):
    patch_post(monkeypatch, lambda *a, **k: fake_response(429, {"error": "quota"}))
    provider = openai.OpenAIProvider(api_key="key")
    with pytest.raises(LLMRateLimitError):
        provider.complete("hi")


def test_openai_timeout_raises_timeout_error(monkeypatch):
    def raise_timeout(*a, **k):
        raise httpx.TimeoutException("timed out")

    patch_post(monkeypatch, raise_timeout)
    provider = openai.OpenAIProvider(api_key="key")
    with pytest.raises(LLMTimeoutError):
        provider.complete("hi")


def test_openai_supports_json_schema():
    assert openai.OpenAIProvider(api_key="key").supports_json_schema() is True


# ---- Gemini ----

def test_gemini_success(monkeypatch):
    patch_post(
        monkeypatch,
        lambda *a, **k: fake_response(
            200,
            {
                "candidates": [{"content": {"parts": [{"text": "hola"}]}}],
                "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1},
            },
        ),
    )
    provider = gemini.GeminiProvider(api_key="key")
    result = provider.complete("hi")
    assert result.text == "hola"
    assert result.provider == "gemini"
    assert result.usage.prompt_tokens == 3


def test_gemini_no_key_raises_auth_error():
    with pytest.raises(LLMAuthError):
        gemini.GeminiProvider(api_key=None).complete("hi")


def test_gemini_429_raises_rate_limit_error(monkeypatch):
    patch_post(monkeypatch, lambda *a, **k: fake_response(429, {"error": "quota"}))
    with pytest.raises(LLMRateLimitError):
        gemini.GeminiProvider(api_key="key").complete("hi")


# ---- Anthropic ----

def test_anthropic_success(monkeypatch):
    patch_post(
        monkeypatch,
        lambda *a, **k: fake_response(
            200,
            {
                "content": [{"type": "text", "text": "bonjour"}],
                "usage": {"input_tokens": 4, "output_tokens": 2},
            },
        ),
    )
    provider = anthropic.AnthropicProvider(api_key="key")
    result = provider.complete("hi")
    assert result.text == "bonjour"
    assert result.provider == "anthropic"
    assert result.usage.completion_tokens == 2


def test_anthropic_no_key_raises_auth_error():
    with pytest.raises(LLMAuthError):
        anthropic.AnthropicProvider(api_key=None).complete("hi")


def test_anthropic_401_raises_auth_error(monkeypatch):
    patch_post(monkeypatch, lambda *a, **k: fake_response(401, {"error": "bad key"}))
    with pytest.raises(LLMAuthError):
        anthropic.AnthropicProvider(api_key="key").complete("hi")


def test_anthropic_does_not_support_json_schema():
    assert anthropic.AnthropicProvider(api_key="key").supports_json_schema() is False


# ---- Custom OpenAI-compatible ----

def test_custom_openai_compat_success_no_key(monkeypatch):
    patch_post(
        monkeypatch,
        lambda *a, **k: fake_response(
            200,
            {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        ),
    )
    provider = custom_openai_compat.CustomOpenAICompatProvider(
        base_url="https://local.invalid/v1", model="local-model"
    )
    result = provider.complete("hi")
    assert result.text == "ok"
    assert result.provider == "custom_openai_compat"


def test_custom_openai_compat_supports_json_schema_flag():
    provider = custom_openai_compat.CustomOpenAICompatProvider(
        base_url="https://local.invalid/v1", model="local-model", supports_json_schema=True
    )
    assert provider.supports_json_schema() is True


# ---- OpenRouter ----

def test_openrouter_success(monkeypatch):
    patch_post(
        monkeypatch,
        lambda *a, **k: fake_response(
            200,
            {"choices": [{"message": {"content": "hi there"}}], "usage": {"prompt_tokens": 2, "completion_tokens": 2}},
        ),
    )
    provider = openrouter.OpenRouterProvider(api_key="key", model="free/a")
    result = provider.complete("hi")
    assert result.text == "hi there"
    assert result.provider == "openrouter"


def test_openrouter_no_key_raises_auth_error():
    with pytest.raises(LLMAuthError):
        openrouter.OpenRouterProvider(api_key=None).complete("hi")


# ---- Null-content regression (openrouter/openai/custom_openai_compat) ----
#
# Confirmed live against a real OpenRouter free model: a reasoning model can
# return HTTP 200 with message.content: null when it exhausts max_tokens
# before finishing its reasoning (finish_reason: length) - the answer is
# never written. This must raise LLMResponseError, not silently hand back
# `None` as LLMResponse.text, which would crash later with an AttributeError
# when complete_structured calls .strip() on it.


def test_openrouter_null_content_raises_response_error(monkeypatch):
    patch_post(
        monkeypatch,
        lambda *a, **k: fake_response(
            200, {"choices": [{"message": {"content": None}, "finish_reason": "length"}]}
        ),
    )
    provider = openrouter.OpenRouterProvider(api_key="key", model="free/a")
    with pytest.raises(LLMResponseError):
        provider.complete("hi")


def test_openai_null_content_raises_response_error(monkeypatch):
    patch_post(
        monkeypatch,
        lambda *a, **k: fake_response(
            200, {"choices": [{"message": {"content": None}, "finish_reason": "length"}], "usage": {}}
        ),
    )
    provider = openai.OpenAIProvider(api_key="key")
    with pytest.raises(LLMResponseError):
        provider.complete("hi")


def test_custom_openai_compat_null_content_raises_response_error(monkeypatch):
    patch_post(
        monkeypatch,
        lambda *a, **k: fake_response(
            200, {"choices": [{"message": {"content": None}, "finish_reason": "length"}]}
        ),
    )
    provider = custom_openai_compat.CustomOpenAICompatProvider(base_url="https://local.invalid/v1", model="m")
    with pytest.raises(LLMResponseError):
        provider.complete("hi")


# ---- NVIDIA ----

def test_nvidia_success_reports_nvidia_as_provider(monkeypatch):
    seen = {}

    def _post(url, **kwargs):
        seen["url"] = url
        return fake_response(200, {"choices": [{"message": {"content": "hi"}}], "usage": {}})

    patch_post(monkeypatch, _post)
    result = nvidia.NvidiaProvider(api_key="key", model="nv/model").complete("hi")
    assert result.provider == "nvidia"
    assert result.model == "nv/model"
    assert seen["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"


def test_nvidia_no_key_raises_auth_error():
    with pytest.raises(LLMAuthError):
        nvidia.NvidiaProvider(api_key=None).complete("hi")


def test_nvidia_403_raises_model_access_error(monkeypatch):
    patch_post(monkeypatch, lambda *a, **k: fake_response(403, {"error": "no"}))
    with pytest.raises(LLMModelAccessError, match="nvidia"):
        nvidia.NvidiaProvider(api_key="key").complete("hi")


# ---- shared status mapping (_http) ----

def _status(code: int, headers: dict | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://example.invalid")
    return httpx.Response(code, json={"error": {}}, headers=headers or {}, request=request)


def test_402_is_model_access_not_a_generic_failure():
    from halludetect.llm._http import raise_for_provider_error

    with pytest.raises(LLMModelAccessError, match="HTTP 402"):
        raise_for_provider_error(_status(402), "openrouter")


def test_429_with_a_far_reset_is_a_used_up_quota():
    import time

    from halludetect.llm._http import raise_for_provider_error
    from halludetect.llm.exceptions import LLMQuotaExhaustedError

    reset_ms = int((time.time() + 5 * 3600) * 1000)
    headers = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(reset_ms)}
    with pytest.raises(LLMQuotaExhaustedError) as info:
        raise_for_provider_error(_status(429, headers), "openrouter")
    assert info.value.reset_at == pytest.approx(reset_ms / 1000)
    assert not isinstance(info.value, LLMRateLimitError)  # never retried


def test_429_with_a_near_reset_or_no_headers_is_a_retryable_rate_limit():
    import time

    from halludetect.llm._http import raise_for_provider_error

    near = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int((time.time() + 30) * 1000))}
    for headers in (near, {}, {"x-ratelimit-remaining": "3", "x-ratelimit-reset": "9999999999999"}):
        with pytest.raises(LLMRateLimitError):
            raise_for_provider_error(_status(429, headers), "openrouter")


# ---- total call deadline (_http.post_json) ----
#
# Seen live: OpenRouter answered 200 and then kept the body alive with
# whitespace for 5+ minutes. httpx's timeout is per read, so only a
# wall-clock deadline stops that.


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


class _TrickleStream(httpx.SyncByteStream):
    """Keep-alive whitespace every `step_s` of fake time, then `body` (if any)."""

    def __init__(self, clock: _Clock, step_s: float, chunks: int | None, body: bytes = b"") -> None:
        self.clock, self.step_s, self.chunks, self.body = clock, step_s, chunks, body
        self.sent = 0
        self.closed = False

    def __iter__(self):
        while self.chunks is None or self.sent < self.chunks:
            self.clock.now += self.step_s
            self.sent += 1
            yield b"\n"
        yield self.body

    def close(self) -> None:
        self.closed = True


def _patch_trickle(monkeypatch, stream: _TrickleStream, clock: _Clock) -> None:
    @contextlib.contextmanager
    def fake_stream(method, url, **kwargs):
        response = httpx.Response(200, stream=stream, request=httpx.Request(method, url))
        try:
            yield response
        finally:
            response.close()

    monkeypatch.setattr(httpx, "stream", fake_stream)
    monkeypatch.setattr(_http, "time", type("T", (), {"monotonic": clock.monotonic, "time": _http.time.time}))


def test_trickled_body_times_out_at_the_call_deadline(monkeypatch):
    clock = _Clock()
    stream = _TrickleStream(clock, step_s=5.0, chunks=None)
    _patch_trickle(monkeypatch, stream, clock)

    with pytest.raises(LLMTimeoutError, match="openrouter"):
        openrouter.OpenRouterProvider(api_key="key", model="free/a").complete("hi")
    assert clock.now <= _http.CALL_DEADLINE_S + 5.0
    assert stream.closed


def test_keepalive_whitespace_before_the_body_still_parses(monkeypatch):
    clock = _Clock()
    body = b'{"choices": [{"message": {"content": "late but fine"}}], "usage": {}}'
    stream = _TrickleStream(clock, step_s=5.0, chunks=6, body=body)
    _patch_trickle(monkeypatch, stream, clock)

    result = openrouter.OpenRouterProvider(api_key="key", model="free/a").complete("hi")
    assert result.text == "late but fine"


def test_deadline_applies_to_every_chat_provider(monkeypatch):
    providers = [
        openai.OpenAIProvider(api_key="key"),
        gemini.GeminiProvider(api_key="key"),
        anthropic.AnthropicProvider(api_key="key"),
        custom_openai_compat.CustomOpenAICompatProvider(base_url="https://local.invalid/v1", model="m"),
        nvidia.NvidiaProvider(api_key="key"),
    ]
    for provider in providers:
        clock = _Clock()
        _patch_trickle(monkeypatch, _TrickleStream(clock, step_s=5.0, chunks=None), clock)
        with pytest.raises(LLMTimeoutError):
            provider.complete("hi")
