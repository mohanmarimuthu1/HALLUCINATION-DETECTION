"""FastAPI layer tests (Phase 5.1) - offline, no network, no live keys.

`OpenRouterProvider.complete` and the free-model catalog fetch are
monkeypatched the same way test_openrouter_catalog.py and test_router.py
mock them; this suite is about request wiring (parsing -> resolution ->
pipeline -> response), not HTTP parsing, which is already covered per
provider elsewhere.
"""
import json
from collections import deque

import pytest
from fastapi.testclient import TestClient

from halludetect.api import main, ratelimit
from halludetect.api.resolve import _openrouter_health
from halludetect.llm import openrouter
from halludetect.settings import Settings, get_settings

_VALID_KEY = "test-key"
_AUTH_HEADERS = {"Authorization": f"Bearer {_VALID_KEY}"}

# Set per-test by _reset_state below, to an isolated tmp_path - the result
# cache (Phase 6.1) is file-backed and would otherwise persist real state
# across test runs/tests sharing the same request payload.
_default_cache_dir: object = None


def _settings(**overrides) -> Settings:
    defaults = {
        "openrouter_api_key": "key",
        "client_api_keys": _VALID_KEY,
        "cache_dir": str(_default_cache_dir),
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _use_settings(settings: Settings, monkeypatch) -> None:
    """`main.py` calls get_settings() directly (a plain module-global lookup,
    interceptable via monkeypatch); `auth.py`/`ratelimit.py` bind it as a
    FastAPI `Depends(get_settings)` default, resolved by object identity at
    request time - that needs FastAPI's own override mechanism instead,
    since monkeypatching the module attribute doesn't reach a reference
    already captured inside a `Depends()` marker.
    """
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    main.app.dependency_overrides[get_settings] = lambda: settings


@pytest.fixture(autouse=True)
def _reset_state(tmp_path, monkeypatch):
    global _default_cache_dir
    _default_cache_dir = tmp_path / "cache"
    _use_settings(_settings(), monkeypatch)
    main.app.state.custom_evidence_source = None
    _openrouter_health._health.clear()
    ratelimit._limiter = None
    main._cache_store = None
    yield
    main.app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(main.app)


def _script_openrouter_completions(monkeypatch, texts: list[str]):
    responses = deque(texts)

    def _complete(self, prompt, *, max_tokens=1024):
        from halludetect.llm.base import LLMResponse, TokenUsage

        return LLMResponse(
            text=responses.popleft(),
            provider="openrouter",
            model=self._model,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1),
        )

    monkeypatch.setattr(openrouter.OpenRouterProvider, "complete", _complete)


def test_healthz():
    client = TestClient(main.app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_verify_with_no_evidence_is_not_verifiable_without_calling_the_model(client, monkeypatch):
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/a"])

    def fail_if_called(self, prompt, *, max_tokens=1024):
        raise AssertionError("provider should never be called when evidence is empty")

    monkeypatch.setattr(openrouter.OpenRouterProvider, "complete", fail_if_called)

    response = client.post(
        "/v1/verify",
        json={"answer": "The sky is blue.", "evidence_source": "none"},
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "NOT_VERIFIABLE"
    assert body["model_used"] == {"provider": "openrouter", "model": "free/a"}


def test_verify_with_direct_evidence_runs_full_pipeline(client, monkeypatch):
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/a"])
    _script_openrouter_completions(
        monkeypatch,
        [
            json.dumps({"claims": [{"text": "Paris is the capital of France.", "claim_type": "FACTUAL"}]}),
            json.dumps(
                {
                    "claim_verdicts": [
                        {
                            "claim_id": "claim-0",
                            "label": "SUPPORTED",
                            "confidence": 0.9,
                            "evidence_chunk_ids": ["direct-0"],
                            "quote": "Paris is the capital of France.",
                        }
                    ]
                }
            ),
        ],
    )

    response = client.post(
        "/v1/verify",
        json={
            "answer": "Paris is the capital of France.",
            "evidence": ["Paris is the capital of France."],
            "evidence_source": "none",
        },
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["claims"][0]["label"] == "SUPPORTED"
    assert body["claims"][0]["quote_verified"] is True
    assert body["model_used"] == {"provider": "openrouter", "model": "free/a"}


def test_verify_custom_evidence_source_without_registration_is_400(client):
    response = client.post(
        "/v1/verify",
        json={"answer": "irrelevant", "evidence_source": "custom"},
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 400


def test_verify_custom_model_provider_without_base_url_is_400(client):
    response = client.post(
        "/v1/verify",
        json={
            "answer": "irrelevant",
            "evidence_source": "none",
            "model_prefs": {"provider": "custom", "pinned_model": "some-model"},
        },
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 400


def test_verify_no_free_models_and_no_pinned_is_503(client, monkeypatch):
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: [])
    response = client.post(
        "/v1/verify",
        json={"answer": "irrelevant", "evidence_source": "none"},
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 503


# --- Phase 5.2: auth + rate limiting ----------------------------------------


def test_verify_without_authorization_header_is_401(client):
    response = client.post("/v1/verify", json={"answer": "irrelevant", "evidence_source": "none"})
    assert response.status_code == 401


def test_verify_with_wrong_key_is_401(client):
    response = client.post(
        "/v1/verify",
        json={"answer": "irrelevant", "evidence_source": "none"},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert response.status_code == 401


def test_verify_with_malformed_header_is_401(client):
    response = client.post(
        "/v1/verify",
        json={"answer": "irrelevant", "evidence_source": "none"},
        headers={"Authorization": _VALID_KEY},
    )
    assert response.status_code == 401


def test_verify_with_no_keys_configured_rejects_everything(client, monkeypatch):
    _use_settings(_settings(client_api_keys=None), monkeypatch)
    response = client.post(
        "/v1/verify",
        json={"answer": "irrelevant", "evidence_source": "none"},
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 401


def test_verify_over_rate_limit_is_429(client, monkeypatch):
    _use_settings(_settings(rate_limit_capacity=1, rate_limit_refill_per_s=0.0), monkeypatch)
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/a"])
    body = {"answer": "irrelevant", "evidence_source": "none"}

    first = client.post("/v1/verify", json=body, headers=_AUTH_HEADERS)
    assert first.status_code == 200

    second = client.post("/v1/verify", json=body, headers=_AUTH_HEADERS)
    assert second.status_code == 429


def test_rate_limit_is_tracked_per_key_not_globally(client, monkeypatch):
    _use_settings(
        _settings(client_api_keys="key-a,key-b", rate_limit_capacity=1, rate_limit_refill_per_s=0.0),
        monkeypatch,
    )
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/a"])
    body = {"answer": "irrelevant", "evidence_source": "none"}

    first = client.post("/v1/verify", json=body, headers={"Authorization": "Bearer key-a"})
    assert first.status_code == 200

    second = client.post("/v1/verify", json=body, headers={"Authorization": "Bearer key-b"})
    assert second.status_code == 200


# --- Phase 6.1: result cache ------------------------------------------------

_CLAIM_RESPONSE = json.dumps({"claims": [{"text": "Paris is the capital of France.", "claim_type": "FACTUAL"}]})
_VERDICT_RESPONSE = json.dumps(
    {
        "claim_verdicts": [
            {
                "claim_id": "claim-0",
                "label": "SUPPORTED",
                "confidence": 0.9,
                "evidence_chunk_ids": ["direct-0"],
                "quote": "Paris is the capital of France.",
            }
        ]
    }
)
_CACHE_TEST_BODY = {
    "answer": "Paris is the capital of France.",
    "evidence": ["Paris is the capital of France."],
    "evidence_source": "none",
}


def test_identical_requests_are_served_from_cache(client, monkeypatch):
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/a"])
    # Only two scripted responses for two identical requests - if the second
    # request weren't served from cache, the third .popleft() call below
    # would raise IndexError on the empty deque.
    _script_openrouter_completions(monkeypatch, [_CLAIM_RESPONSE, _VERDICT_RESPONSE])

    first = client.post("/v1/verify", json=_CACHE_TEST_BODY, headers=_AUTH_HEADERS)
    assert first.status_code == 200
    first_body = first.json()

    second = client.post("/v1/verify", json=_CACHE_TEST_BODY, headers=_AUTH_HEADERS)
    assert second.status_code == 200
    second_body = second.json()

    assert second_body["claims"] == first_body["claims"]
    assert second_body["verdict"] == first_body["verdict"]
    assert second_body["request_id"] != first_body["request_id"]
    assert second_body["timings_ms"]["extraction"] == 0
    assert second_body["timings_ms"]["verification"] == 0


def test_cache_disabled_runs_the_pipeline_every_time(client, monkeypatch):
    _use_settings(_settings(cache_enabled=False), monkeypatch)
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/a"])

    responses = deque([_CLAIM_RESPONSE, _VERDICT_RESPONSE, _CLAIM_RESPONSE, _VERDICT_RESPONSE])
    call_count = {"n": 0}

    def _complete(self, prompt, *, max_tokens=1024):
        from halludetect.llm.base import LLMResponse, TokenUsage

        call_count["n"] += 1
        return LLMResponse(
            text=responses.popleft(),
            provider="openrouter",
            model=self._model,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1),
        )

    monkeypatch.setattr(openrouter.OpenRouterProvider, "complete", _complete)

    first = client.post("/v1/verify", json=_CACHE_TEST_BODY, headers=_AUTH_HEADERS)
    second = client.post("/v1/verify", json=_CACHE_TEST_BODY, headers=_AUTH_HEADERS)
    assert first.status_code == 200
    assert second.status_code == 200
    assert call_count["n"] == 4
