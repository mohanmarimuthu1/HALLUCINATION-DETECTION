"""Provider tests, run fully offline by monkeypatching httpx.post - no live
API keys required. Covers: successful parse, missing-key auth error,
401/429 classification, and timeout classification, per provider.
"""
import httpx
import pytest

from halludetect.llm import anthropic, custom_openai_compat, gemini, openai, openrouter
from halludetect.llm.exceptions import LLMAuthError, LLMRateLimitError, LLMResponseError, LLMTimeoutError


def fake_response(status_code: int, json_body: dict) -> httpx.Response:
    request = httpx.Request("POST", "https://example.invalid")
    return httpx.Response(status_code, json=json_body, request=request)


# ---- OpenAI ----

def test_openai_success(monkeypatch):
    monkeypatch.setattr(
        openai.httpx,
        "post",
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
    monkeypatch.setattr(openai.httpx, "post", lambda *a, **k: fake_response(401, {"error": "bad key"}))
    provider = openai.OpenAIProvider(api_key="key")
    with pytest.raises(LLMAuthError):
        provider.complete("hi")


def test_openai_429_raises_rate_limit_error(monkeypatch):
    monkeypatch.setattr(openai.httpx, "post", lambda *a, **k: fake_response(429, {"error": "quota"}))
    provider = openai.OpenAIProvider(api_key="key")
    with pytest.raises(LLMRateLimitError):
        provider.complete("hi")


def test_openai_timeout_raises_timeout_error(monkeypatch):
    def raise_timeout(*a, **k):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(openai.httpx, "post", raise_timeout)
    provider = openai.OpenAIProvider(api_key="key")
    with pytest.raises(LLMTimeoutError):
        provider.complete("hi")


def test_openai_supports_json_schema():
    assert openai.OpenAIProvider(api_key="key").supports_json_schema() is True


# ---- Gemini ----

def test_gemini_success(monkeypatch):
    monkeypatch.setattr(
        gemini.httpx,
        "post",
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
    monkeypatch.setattr(gemini.httpx, "post", lambda *a, **k: fake_response(429, {"error": "quota"}))
    with pytest.raises(LLMRateLimitError):
        gemini.GeminiProvider(api_key="key").complete("hi")


# ---- Anthropic ----

def test_anthropic_success(monkeypatch):
    monkeypatch.setattr(
        anthropic.httpx,
        "post",
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
    monkeypatch.setattr(anthropic.httpx, "post", lambda *a, **k: fake_response(401, {"error": "bad key"}))
    with pytest.raises(LLMAuthError):
        anthropic.AnthropicProvider(api_key="key").complete("hi")


def test_anthropic_does_not_support_json_schema():
    assert anthropic.AnthropicProvider(api_key="key").supports_json_schema() is False


# ---- Custom OpenAI-compatible ----

def test_custom_openai_compat_success_no_key(monkeypatch):
    monkeypatch.setattr(
        custom_openai_compat.httpx,
        "post",
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
    monkeypatch.setattr(
        openrouter.httpx,
        "post",
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
    monkeypatch.setattr(
        openrouter.httpx,
        "post",
        lambda *a, **k: fake_response(
            200, {"choices": [{"message": {"content": None}, "finish_reason": "length"}]}
        ),
    )
    provider = openrouter.OpenRouterProvider(api_key="key", model="free/a")
    with pytest.raises(LLMResponseError):
        provider.complete("hi")


def test_openai_null_content_raises_response_error(monkeypatch):
    monkeypatch.setattr(
        openai.httpx,
        "post",
        lambda *a, **k: fake_response(
            200, {"choices": [{"message": {"content": None}, "finish_reason": "length"}], "usage": {}}
        ),
    )
    provider = openai.OpenAIProvider(api_key="key")
    with pytest.raises(LLMResponseError):
        provider.complete("hi")


def test_custom_openai_compat_null_content_raises_response_error(monkeypatch):
    monkeypatch.setattr(
        custom_openai_compat.httpx,
        "post",
        lambda *a, **k: fake_response(
            200, {"choices": [{"message": {"content": None}, "finish_reason": "length"}]}
        ),
    )
    provider = custom_openai_compat.CustomOpenAICompatProvider(base_url="https://local.invalid/v1", model="m")
    with pytest.raises(LLMResponseError):
        provider.complete("hi")
