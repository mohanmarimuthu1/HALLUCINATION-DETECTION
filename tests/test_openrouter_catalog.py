"""OpenRouter free-model catalog tests - offline, httpx.get monkeypatched.
Covers pricing-based filtering and the daily-refresh cache."""
import httpx
import pytest

from halludetect.llm import openrouter
from halludetect.llm.exceptions import LLMAuthError


def fake_models_response(models: list[dict]) -> httpx.Response:
    request = httpx.Request("GET", "https://example.invalid")
    return httpx.Response(200, json={"data": models}, request=request)


def test_fetch_free_models_filters_by_zero_pricing(monkeypatch):
    monkeypatch.setattr(
        openrouter.httpx,
        "get",
        lambda *a, **k: fake_models_response(
            [
                {"id": "free/model-a", "pricing": {"prompt": "0", "completion": "0"}},
                {"id": "paid/model-b", "pricing": {"prompt": "0.000002", "completion": "0.000006"}},
                {"id": "free/model-c", "pricing": {"prompt": "0.0", "completion": "0.0"}},
            ]
        ),
    )
    assert openrouter.fetch_free_models(api_key="key") == ["free/model-a", "free/model-c"]


def test_fetch_free_models_skips_malformed_pricing(monkeypatch):
    monkeypatch.setattr(
        openrouter.httpx,
        "get",
        lambda *a, **k: fake_models_response([{"id": "broken/model", "pricing": {}}]),
    )
    assert openrouter.fetch_free_models(api_key="key") == []


def test_fetch_free_models_auth_error(monkeypatch):
    def fake_get(*a, **k):
        request = httpx.Request("GET", "https://example.invalid")
        return httpx.Response(401, json={"error": "no key"}, request=request)

    monkeypatch.setattr(openrouter.httpx, "get", fake_get)
    with pytest.raises(LLMAuthError):
        openrouter.fetch_free_models(api_key=None)


def test_catalog_caches_within_ttl(monkeypatch):
    calls = {"n": 0}

    def fake_fetch(api_key):
        calls["n"] += 1
        return ["free/model-a"]

    monkeypatch.setattr(openrouter, "fetch_free_models", fake_fetch)
    clock = {"t": 0.0}
    monkeypatch.setattr(openrouter.time, "monotonic", lambda: clock["t"])

    catalog = openrouter.FreeModelCatalog(api_key="key", ttl_s=100)
    assert catalog.get_models() == ["free/model-a"]
    clock["t"] = 50.0
    assert catalog.get_models() == ["free/model-a"]
    assert calls["n"] == 1


def test_catalog_refreshes_after_ttl(monkeypatch):
    calls = {"n": 0}

    def fake_fetch(api_key):
        calls["n"] += 1
        return [f"free/model-{calls['n']}"]

    monkeypatch.setattr(openrouter, "fetch_free_models", fake_fetch)
    clock = {"t": 0.0}
    monkeypatch.setattr(openrouter.time, "monotonic", lambda: clock["t"])

    catalog = openrouter.FreeModelCatalog(api_key="key", ttl_s=100)
    assert catalog.get_models() == ["free/model-1"]
    clock["t"] = 150.0
    assert catalog.get_models() == ["free/model-2"]
    assert calls["n"] == 2


def test_catalog_force_refresh_ignores_ttl(monkeypatch):
    calls = {"n": 0}

    def fake_fetch(api_key):
        calls["n"] += 1
        return [f"free/model-{calls['n']}"]

    monkeypatch.setattr(openrouter, "fetch_free_models", fake_fetch)
    catalog = openrouter.FreeModelCatalog(api_key="key", ttl_s=10_000)
    assert catalog.get_models() == ["free/model-1"]
    assert catalog.get_models(force_refresh=True) == ["free/model-2"]
    assert calls["n"] == 2
