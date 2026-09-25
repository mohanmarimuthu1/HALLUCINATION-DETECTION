"""Phase 5.1 request-resolution tests - offline, no network, no live keys.

Covers `halludetect.api.resolve`: evidence-source selection per
docs/contract.md's `evidence` vs `evidence_source` rules, and model
resolution per `model_prefs` (openrouter free pool via Router.pick_model(),
named providers, the `custom` provider, and the error paths for
configuration this deployment can't satisfy).
"""
import pytest

from halludetect.api.resolve import ResolutionError, resolve_evidence_source, resolve_provider
from halludetect.api.schemas import EvidenceSourceKind, ModelPrefsIn, ModelProvider, VerifyRequestIn
from halludetect.evidence.direct import DirectEvidence
from halludetect.evidence.none import NoEvidenceSource
from halludetect.evidence.web_search import WebSearchEvidence
from halludetect.llm import openrouter
from halludetect.llm.exceptions import LLMResponseError
from halludetect.llm.gemini import GeminiProvider
from halludetect.settings import Settings


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def _request(**overrides) -> VerifyRequestIn:
    defaults = {"answer": "irrelevant", "evidence_source": EvidenceSourceKind.NONE}
    defaults.update(overrides)
    return VerifyRequestIn(**defaults)


# --- evidence source resolution -------------------------------------------


def test_direct_evidence_is_authoritative_even_with_evidence_source_set():
    request = _request(evidence=["some fact"], evidence_source=EvidenceSourceKind.CUSTOM)
    source = resolve_evidence_source(request, _settings(), custom_evidence_source=None)
    assert isinstance(source, DirectEvidence)


def test_custom_without_registered_source_raises_resolution_error():
    request = _request(evidence_source=EvidenceSourceKind.CUSTOM)
    with pytest.raises(ResolutionError):
        resolve_evidence_source(request, _settings(), custom_evidence_source=None)


def test_custom_with_registered_source_is_used_directly():
    request = _request(evidence_source=EvidenceSourceKind.CUSTOM)

    class _Registered:
        def fetch(self, query: str) -> list:
            return []

    registered = _Registered()
    source = resolve_evidence_source(request, _settings(), custom_evidence_source=registered)
    assert source is registered


def test_web_evidence_source_is_web_search_evidence():
    request = _request(evidence_source=EvidenceSourceKind.WEB)
    source = resolve_evidence_source(request, _settings(tavily_api_key="tvly-key"), custom_evidence_source=None)
    assert isinstance(source, WebSearchEvidence)


def test_none_evidence_source_is_no_evidence_source():
    request = _request(evidence_source=EvidenceSourceKind.NONE)
    source = resolve_evidence_source(request, _settings(), custom_evidence_source=None)
    assert isinstance(source, NoEvidenceSource)


# --- model resolution -------------------------------------------------------


def test_openrouter_pinned_model_bypasses_free_pool(monkeypatch):
    def fail_if_called(api_key):
        raise AssertionError("catalog should not be fetched when pinned_model is set")

    monkeypatch.setattr(openrouter, "fetch_free_models", fail_if_called)
    prefs = ModelPrefsIn(provider=ModelProvider.OPENROUTER, pinned_model="openrouter/pinned")
    provider, model_used = resolve_provider(prefs, _settings(openrouter_api_key="key"))
    assert model_used.provider == "openrouter"
    assert model_used.model == "openrouter/pinned"
    assert isinstance(provider, openrouter.OpenRouterProvider)


def test_openrouter_picks_top_ranked_free_model(monkeypatch):
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/a", "free/b"])
    prefs = ModelPrefsIn(provider=ModelProvider.OPENROUTER)
    provider, model_used = resolve_provider(prefs, _settings(openrouter_api_key="key"))
    assert model_used == type(model_used)(provider="openrouter", model="free/a")
    assert isinstance(provider, openrouter.OpenRouterProvider)


def test_openrouter_no_pinned_and_no_free_models_raises(monkeypatch):
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: [])
    prefs = ModelPrefsIn(provider=ModelProvider.OPENROUTER)
    with pytest.raises(LLMResponseError):
        resolve_provider(prefs, _settings())


def test_named_provider_uses_user_api_key_and_pinned_model():
    prefs = ModelPrefsIn(provider=ModelProvider.GEMINI, user_api_key="user-key", pinned_model="gemini-custom")
    provider, model_used = resolve_provider(prefs, _settings())
    assert isinstance(provider, GeminiProvider)
    assert model_used.provider == "gemini"
    assert model_used.model == "gemini-custom"


def test_named_provider_falls_back_to_deployment_key_and_default_model():
    prefs = ModelPrefsIn(provider=ModelProvider.GEMINI)
    provider, model_used = resolve_provider(prefs, _settings(gemini_api_key="deployment-key"))
    assert isinstance(provider, GeminiProvider)
    assert model_used.model  # default model id, non-empty


def test_custom_provider_requires_base_url():
    prefs = ModelPrefsIn(provider=ModelProvider.CUSTOM, pinned_model="some-model")
    with pytest.raises(ResolutionError):
        resolve_provider(prefs, _settings())


def test_custom_provider_requires_pinned_model():
    prefs = ModelPrefsIn(provider=ModelProvider.CUSTOM)
    with pytest.raises(ResolutionError):
        resolve_provider(prefs, _settings(custom_provider_base_url="https://example.invalid"))


def test_custom_provider_resolves_with_base_url_and_pinned_model():
    prefs = ModelPrefsIn(provider=ModelProvider.CUSTOM, pinned_model="local-model")
    provider, model_used = resolve_provider(
        prefs, _settings(custom_provider_base_url="https://example.invalid")
    )
    assert model_used.provider == "custom"
    assert model_used.model == "local-model"
