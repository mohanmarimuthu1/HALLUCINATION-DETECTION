"""FastAPI layer tests (Phase 5.1) - offline, no network, no live keys.

`OpenRouterProvider.complete` and the free-model catalog fetch are
monkeypatched the same way test_openrouter_catalog.py and test_router.py
mock them; this suite is about request wiring (parsing -> resolution ->
pipeline -> response), not HTTP parsing, which is already covered per
provider elsewhere.
"""
import errno
import json
from collections import deque

import pytest
from fastapi.testclient import TestClient

from halludetect.api import main, ratelimit
from halludetect.api.resolve import _nvidia_health, _openrouter_health
from halludetect.llm import nvidia, openrouter
from halludetect.llm.exceptions import LLMAuthError, LLMResponseError, LLMTimeoutError
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
    _nvidia_health._health.clear()
    ratelimit._limiter = None
    main._cache_store = None
    main._cache_unavailable = False
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
    assert body["reason"] == "no_evidence_configured"
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


def test_unwritable_cache_dir_runs_uncached_instead_of_failing(client, monkeypatch):
    """Vercel's filesystem is read-only outside /tmp. Opening the cache there
    used to raise inside the request, so every /v1/verify returned 500.
    """
    opened: list[str] = []

    def _read_only(directory):
        opened.append(directory)
        raise OSError(errno.EROFS, "Read-only file system", directory)

    monkeypatch.setattr(main, "DiskCacheStore", _read_only)
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/a"])
    body = {"answer": "irrelevant", "evidence_source": "none"}

    first = client.post("/v1/verify", json=body, headers=_AUTH_HEADERS)
    second = client.post("/v1/verify", json=body, headers=_AUTH_HEADERS)

    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["verdict"] == "NOT_VERIFIABLE"
    assert len(opened) == 1  # a failed open is remembered, not retried per request


def test_not_enough_info_is_not_cached_so_a_flaky_response_is_not_sticky(client, monkeypatch):
    """Seen in production: a free model returned no quotes once, turning a
    fully supported answer into NOT_ENOUGH_INFO, and the cache then served
    that to every identical request for an hour.
    """
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/a"])
    evidence = ["The tower is 330 metres tall.", "It was completed in 1889.", "It stands in Paris."]
    claims = json.dumps(
        {"claims": [{"text": text, "claim_type": "FACTUAL"} for text in ("330 m", "1889", "Paris")]}
    )

    def _verdicts(label: str, with_quotes: bool) -> str:
        return json.dumps(
            {
                "claim_verdicts": [
                    {
                        "claim_id": f"claim-{i}",
                        "label": label,
                        "confidence": 0.9,
                        "evidence_chunk_ids": [f"direct-{i}"],
                        "quote": evidence[i] if with_quotes else "",
                    }
                    for i in range(3)
                ]
            }
        )

    _script_openrouter_completions(
        monkeypatch,
        # The flaky request's recheck also comes back without quotes.
        [claims, _verdicts("NOT_ENOUGH_INFO", False), _verdicts("NOT_ENOUGH_INFO", False),
         claims, _verdicts("SUPPORTED", True)],
    )
    body = {"answer": "It is 330 m, done in 1889, in Paris.", "evidence": evidence, "evidence_source": "none"}

    flaky = client.post("/v1/verify", json=body, headers=_AUTH_HEADERS)
    retry = client.post("/v1/verify", json=body, headers=_AUTH_HEADERS)

    assert flaky.json()["verdict"] == "NOT_ENOUGH_INFO"
    assert retry.json()["verdict"] == "GROUNDED"  # recomputed, not served from cache


_ONE_SUPPORTED_CLAIM = [
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
]

_EVIDENCE_REQUEST = {
    "answer": "Paris is the capital of France.",
    "evidence": ["Paris is the capital of France."],
    "evidence_source": "none",
}


def _script_by_model(monkeypatch, cls, outcomes: dict[str, list]):
    """Patches `cls.complete` so each model plays its own script; an
    exception in a script is raised instead of returned.
    """
    from halludetect.llm.base import LLMResponse, TokenUsage

    queues = {model: deque(script) for model, script in outcomes.items()}
    calls: list[str] = []

    def _complete(self, prompt, *, max_tokens=1024):
        calls.append(self._model)
        item = queues[self._model].popleft()
        if isinstance(item, Exception):
            raise item
        usage = TokenUsage(prompt_tokens=1, completion_tokens=1)
        return LLMResponse(text=item, provider="x", model=self._model, usage=usage)

    monkeypatch.setattr(cls, "complete", _complete)
    return calls


def test_failed_free_model_fails_over_to_the_next_within_one_request(client, monkeypatch):
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/a", "free/b"])
    calls = _script_by_model(
        monkeypatch,
        openrouter.OpenRouterProvider,
        {"free/a": [LLMResponseError("no choices")], "free/b": list(_ONE_SUPPORTED_CLAIM)},
    )
    response = client.post("/v1/verify", json=_EVIDENCE_REQUEST, headers=_AUTH_HEADERS)
    assert response.status_code == 200
    assert response.json()["model_used"] == {"provider": "openrouter", "model": "free/b"}
    assert calls == ["free/a", "free/b", "free/b"]
    assert _openrouter_health.get("free/a").consecutive_failures == 1
    assert _openrouter_health.get("free/b").successes == 1


def test_nvidia_models_back_up_the_openrouter_pool(client, monkeypatch):
    _use_settings(_settings(nvidia_api_key="nv", nvidia_models="nv/one"), monkeypatch)
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/a"])
    _script_by_model(monkeypatch, openrouter.OpenRouterProvider, {"free/a": [LLMResponseError("down")]})
    _script_by_model(monkeypatch, nvidia.NvidiaProvider, {"nv/one": list(_ONE_SUPPORTED_CLAIM)})
    response = client.post("/v1/verify", json=_EVIDENCE_REQUEST, headers=_AUTH_HEADERS)
    assert response.status_code == 200
    assert response.json()["model_used"] == {"provider": "nvidia", "model": "nv/one"}
    assert _nvidia_health.get("nv/one").successes == 1


def test_nvidia_serves_alone_when_no_openrouter_key(client, monkeypatch):
    _use_settings(_settings(openrouter_api_key=None, nvidia_api_key="nv", nvidia_models="nv/one"), monkeypatch)
    _script_by_model(monkeypatch, nvidia.NvidiaProvider, {"nv/one": list(_ONE_SUPPORTED_CLAIM)})
    response = client.post("/v1/verify", json=_EVIDENCE_REQUEST, headers=_AUTH_HEADERS)
    assert response.status_code == 200
    assert response.json()["model_used"]["provider"] == "nvidia"


def test_rejected_openrouter_key_skips_its_other_models(client, monkeypatch):
    _use_settings(_settings(nvidia_api_key="nv", nvidia_models="nv/one"), monkeypatch)
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/a", "free/b"])
    calls = _script_by_model(
        monkeypatch, openrouter.OpenRouterProvider, {"free/a": [LLMAuthError("401")], "free/b": []}
    )
    _script_by_model(monkeypatch, nvidia.NvidiaProvider, {"nv/one": list(_ONE_SUPPORTED_CLAIM)})
    response = client.post("/v1/verify", json=_EVIDENCE_REQUEST, headers=_AUTH_HEADERS)
    assert response.status_code == 200
    assert calls == ["free/a"]
    assert _openrouter_health.get("free/a").consecutive_failures == 0


def test_every_candidate_failing_is_502_after_the_attempt_cap(client, monkeypatch):
    _use_settings(_settings(free_pool_max_attempts=2), monkeypatch)
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/a", "free/b", "free/c"])
    calls = _script_by_model(
        monkeypatch,
        openrouter.OpenRouterProvider,
        {m: [LLMResponseError("down")] for m in ("free/a", "free/b", "free/c")},
    )
    response = client.post("/v1/verify", json=_EVIDENCE_REQUEST, headers=_AUTH_HEADERS)
    assert response.status_code == 502
    assert calls == ["free/a", "free/b"]


def test_pinned_nvidia_model_is_used_as_is(client, monkeypatch):
    _use_settings(_settings(nvidia_api_key="nv"), monkeypatch)
    _script_by_model(monkeypatch, nvidia.NvidiaProvider, {"nv/pinned": list(_ONE_SUPPORTED_CLAIM)})
    response = client.post(
        "/v1/verify",
        json={**_EVIDENCE_REQUEST, "model_prefs": {"provider": "nvidia", "pinned_model": "nv/pinned"}},
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["model_used"] == {"provider": "nvidia", "model": "nv/pinned"}


def test_free_pool_timeout_moves_to_the_next_model_without_retrying(client, monkeypatch):
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/slow", "free/b"])
    calls = _script_by_model(
        monkeypatch,
        openrouter.OpenRouterProvider,
        {"free/slow": [LLMTimeoutError("timed out")], "free/b": list(_ONE_SUPPORTED_CLAIM)},
    )
    response = client.post("/v1/verify", json=_EVIDENCE_REQUEST, headers=_AUTH_HEADERS)
    assert response.status_code == 200
    assert calls.count("free/slow") == 1


def test_no_new_attempt_starts_once_the_time_budget_is_spent(client, monkeypatch):
    _use_settings(_settings(free_pool_budget_s=0.0), monkeypatch)
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/a", "free/b"])
    calls = _script_by_model(
        monkeypatch,
        openrouter.OpenRouterProvider,
        {"free/a": [LLMResponseError("down")], "free/b": list(_ONE_SUPPORTED_CLAIM)},
    )
    response = client.post("/v1/verify", json=_EVIDENCE_REQUEST, headers=_AUTH_HEADERS)
    assert response.status_code == 502
    assert "stopped after" in response.json()["detail"]
    assert calls == ["free/a"]


def test_allow_free_pool_false_without_a_pinned_model_is_400(client, monkeypatch):
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/a"])
    response = client.post(
        "/v1/verify",
        json={**_EVIDENCE_REQUEST, "model_prefs": {"allow_free_pool": False}},
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 400
    assert "allow_free_pool" in response.json()["detail"]


def test_allow_free_pool_false_with_a_pinned_model_runs(client, monkeypatch):
    _script_by_model(monkeypatch, openrouter.OpenRouterProvider, {"paid/model": list(_ONE_SUPPORTED_CLAIM)})
    response = client.post(
        "/v1/verify",
        json={**_EVIDENCE_REQUEST, "model_prefs": {"allow_free_pool": False, "pinned_model": "paid/model"}},
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["model_used"] == {"provider": "openrouter", "model": "paid/model"}


def test_a_request_that_never_calls_the_model_leaves_its_health_alone(client, monkeypatch):
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/a"])
    response = client.post(
        "/v1/verify", json={"answer": "The sky is blue.", "evidence_source": "none"}, headers=_AUTH_HEADERS
    )
    assert response.status_code == 200
    assert _openrouter_health.get("free/a").attempts == 0


# --- /v1/chat and /v1/models/stats (contract v1.3) --------------------------

_CHAT_REQUEST = {
    "question": "What is the capital of France?",
    "evidence": ["Paris is the capital of France."],
}


def test_chat_answer_is_checked_by_a_different_model(client, monkeypatch):
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/a", "free/b"])
    calls = _script_by_model(
        monkeypatch,
        openrouter.OpenRouterProvider,
        {"free/a": ["Paris is the capital of France."], "free/b": list(_ONE_SUPPORTED_CLAIM)},
    )
    response = client.post("/v1/chat", json=_CHAT_REQUEST, headers=_AUTH_HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Paris is the capital of France."
    assert body["answer_model"] == {"provider": "openrouter", "model": "free/a"}
    assert body["verification"]["model_used"] == {"provider": "openrouter", "model": "free/b"}
    assert body["verification"]["claims"][0]["label"] == "SUPPORTED"
    assert calls == ["free/a", "free/b", "free/b"]


def test_chat_without_evidence_answers_but_never_verifies_from_model_knowledge(client, monkeypatch):
    # Default evidence_source is web; with no TAVILY_API_KEY that is the
    # same as none, so the answer must come back NOT_VERIFIABLE.
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/a", "free/b"])
    calls = _script_by_model(
        monkeypatch, openrouter.OpenRouterProvider, {"free/a": ["Paris is the capital of France."]}
    )
    response = client.post("/v1/chat", json={"question": "Capital of France?"}, headers=_AUTH_HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Paris is the capital of France."
    assert body["verification"]["verdict"] == "NOT_VERIFIABLE"
    assert body["verification"]["reason"] == "no_evidence_configured"
    assert calls == ["free/a"]
    rows = {row["model"]: row for row in client.get("/v1/models/stats", headers=_AUTH_HEADERS).json()["models"]}
    assert "free/b" not in rows


def test_chat_fails_over_when_the_answering_model_fails(client, monkeypatch):
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/a", "free/b"])
    calls = _script_by_model(
        monkeypatch,
        openrouter.OpenRouterProvider,
        {
            "free/a": [LLMTimeoutError("slow")],
            "free/b": ["   "],
        },
    )
    response = client.post("/v1/chat", json=_CHAT_REQUEST, headers=_AUTH_HEADERS)
    # free/a times out and free/b's blank answer counts as a failure too.
    assert response.status_code == 502
    assert calls == ["free/a", "free/b"]


def test_chat_empty_answer_moves_to_next_model(client, monkeypatch):
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/a", "free/b"])
    _script_by_model(
        monkeypatch,
        openrouter.OpenRouterProvider,
        {"free/a": ["", *_ONE_SUPPORTED_CLAIM], "free/b": ["Paris is the capital of France."]},
    )
    response = client.post("/v1/chat", json=_CHAT_REQUEST, headers=_AUTH_HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["answer_model"]["model"] == "free/b"
    assert body["verification"]["model_used"]["model"] == "free/a"


def test_chat_sends_history_and_sources_to_the_answering_model(client, monkeypatch):
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/a"])
    prompts: list[str] = []

    def _complete(self, prompt, *, max_tokens=1024):
        from halludetect.llm.base import LLMResponse, TokenUsage

        prompts.append(prompt)
        text = "It is Paris." if len(prompts) == 1 else _ONE_SUPPORTED_CLAIM[len(prompts) - 2]
        return LLMResponse(text=text, provider="x", model=self._model, usage=TokenUsage(1, 1))

    monkeypatch.setattr(openrouter.OpenRouterProvider, "complete", _complete)
    request = {
        **_CHAT_REQUEST,
        "history": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."}],
    }
    response = client.post("/v1/chat", json=request, headers=_AUTH_HEADERS)
    assert response.status_code == 200
    assert "[1] Paris is the capital of France." in prompts[0]
    assert "USER: Hi\nASSISTANT: Hello." in prompts[0]
    assert prompts[0].endswith("USER: What is the capital of France?\nASSISTANT:")


def test_chat_rejects_bad_requests(client):
    assert client.post("/v1/chat", json={"question": ""}, headers=_AUTH_HEADERS).status_code == 422
    too_long = {"question": "q", "history": [{"role": "user", "content": "x"}] * 21}
    assert client.post("/v1/chat", json=too_long, headers=_AUTH_HEADERS).status_code == 422
    assert client.post("/v1/chat", json={"question": "q"}).status_code == 401


def test_model_stats_reports_answer_and_verify_roles(client, monkeypatch):
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/a", "free/b"])
    _script_by_model(
        monkeypatch,
        openrouter.OpenRouterProvider,
        {
            "free/a": [LLMResponseError("bad"), *_ONE_SUPPORTED_CLAIM],
            "free/b": ["Paris is the capital of France."],
        },
    )
    assert client.post("/v1/chat", json=_CHAT_REQUEST, headers=_AUTH_HEADERS).status_code == 200

    assert client.get("/v1/models/stats").status_code == 401
    stats = client.get("/v1/models/stats", headers=_AUTH_HEADERS).json()
    rows = {row["model"]: row for row in stats["models"]}
    assert rows["free/a"]["answer"]["calls"] == 1
    assert rows["free/a"]["answer"]["failures"] == 1
    assert rows["free/a"]["last_error"] == "bad"
    assert rows["free/b"]["answer"]["calls"] == 1
    assert rows["free/b"]["answer"]["verdicts"]["NOT_VERIFIABLE"] == 1
    # One claim is too few to score, so no rate yet.
    assert rows["free/b"]["answer"]["unsupported_rate"] is None
    assert rows["free/b"]["verify"]["calls"] == 0
    assert rows["free/a"]["verify"]["calls"] == 1
    assert rows["free/a"]["verify"]["avg_latency_ms"] is not None
    assert stats["since"].endswith("Z")


def test_chat_answer_and_check_share_one_time_budget(client, monkeypatch):
    import time

    from halludetect.llm.base import LLMResponse, TokenUsage

    _use_settings(_settings(free_pool_budget_s=0.05), monkeypatch)
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/a", "free/b"])
    calls: list[str] = []

    def _complete(self, prompt, *, max_tokens=1024):
        calls.append(self._model)
        if len(calls) == 1:
            time.sleep(0.1)
            return LLMResponse(text="Paris.", provider="x", model=self._model, usage=TokenUsage(1, 1))
        raise LLMResponseError("down")

    monkeypatch.setattr(openrouter.OpenRouterProvider, "complete", _complete)
    response = client.post("/v1/chat", json=_CHAT_REQUEST, headers=_AUTH_HEADERS)
    # The slow answer used up the budget: the check gets one attempt, not two.
    assert response.status_code == 502
    assert "stopped after" in response.json()["detail"]
    assert calls == ["free/a", "free/b"]


def test_used_up_free_quota_skips_free_variants_until_the_reset(client, monkeypatch):
    import time

    from halludetect.api import resolve
    from halludetect.api.schemas import ModelPrefsIn
    from halludetect.llm.exceptions import LLMQuotaExhaustedError

    _use_settings(_settings(nvidia_api_key="nv", nvidia_models="nv/one"), monkeypatch)
    monkeypatch.setattr(
        openrouter, "fetch_free_models", lambda api_key: ["x/a:free", "x/b:free", "stealth/zero", "x/c:free"]
    )
    quota = LLMQuotaExhaustedError("quota", time.time() + 3600)
    calls = _script_by_model(
        monkeypatch,
        openrouter.OpenRouterProvider,
        {"x/a:free": [quota], "stealth/zero": [LLMResponseError("down")]},
    )
    _script_by_model(monkeypatch, nvidia.NvidiaProvider, {"nv/one": list(_ONE_SUPPORTED_CLAIM) * 2})

    first = client.post("/v1/verify", json=_EVIDENCE_REQUEST, headers=_AUTH_HEADERS)
    assert first.status_code == 200
    assert first.json()["model_used"]["provider"] == "nvidia"
    # x/b:free and x/c:free are skipped; the unsuffixed zero-priced model is not.
    assert calls == ["x/a:free", "stealth/zero"]
    assert _openrouter_health.get("x/a:free").attempts == 0  # not the model's fault

    models = [c.model_used.model for c in resolve.resolve_candidates(ModelPrefsIn(), main.get_settings())]
    assert models == ["stealth/zero", "nv/one"]

    resolve._quota_reset["openrouter"] = time.time() - 1
    models = [c.model_used.model for c in resolve.resolve_candidates(ModelPrefsIn(), main.get_settings())]
    assert "x/b:free" in models


def test_refused_and_quota_models_are_left_out_and_cost_no_attempt(client, monkeypatch):
    import time

    from halludetect.api import resolve
    from halludetect.api.schemas import ModelPrefsIn
    from halludetect.llm.exceptions import LLMModelAccessError, LLMQuotaExhaustedError

    monkeypatch.setattr(
        openrouter,
        "fetch_free_models",
        lambda api_key: ["paid/a", "paid/b", "openrouter/free", "x/c:free", "free/ok"],
    )
    calls = _script_by_model(
        monkeypatch,
        openrouter.OpenRouterProvider,
        {
            "paid/a": [LLMModelAccessError("HTTP 402")],
            "paid/b": [LLMModelAccessError("HTTP 402")],
            "openrouter/free": [LLMQuotaExhaustedError("quota", time.time() + 3600)],
            "free/ok": list(_ONE_SUPPORTED_CLAIM),
        },
    )
    # Default free_pool_max_attempts is 3; the refusals don't use any.
    response = client.post("/v1/verify", json=_EVIDENCE_REQUEST, headers=_AUTH_HEADERS)
    assert response.status_code == 200
    assert response.json()["model_used"]["model"] == "free/ok"
    assert calls == ["paid/a", "paid/b", "openrouter/free", "free/ok", "free/ok"]

    models = [c.model_used.model for c in resolve.resolve_candidates(ModelPrefsIn(), main.get_settings())]
    assert models == ["free/ok"]
