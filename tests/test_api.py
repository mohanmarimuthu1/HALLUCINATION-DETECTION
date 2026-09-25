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

from halludetect.api import main
from halludetect.api.resolve import _openrouter_health
from halludetect.llm import openrouter
from halludetect.settings import Settings


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    monkeypatch.setattr(main, "get_settings", lambda: Settings(_env_file=None, openrouter_api_key="key"))
    main.app.state.custom_evidence_source = None
    _openrouter_health._health.clear()
    yield


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
    )
    assert response.status_code == 400


def test_verify_no_free_models_and_no_pinned_is_503(client, monkeypatch):
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: [])
    response = client.post(
        "/v1/verify",
        json={"answer": "irrelevant", "evidence_source": "none"},
    )
    assert response.status_code == 503
