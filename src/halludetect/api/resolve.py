"""Turns a validated request into the concrete objects `detect.pipeline.run()`
needs: an `EvidenceSource` and a single resolved `LLMProvider` + `ModelUsed`.

This is the wiring Phase 4 explicitly left for Phase 5 (see
`detect/pipeline.py`'s module docstring and `process.md`'s Phase 5 notes):
`pipeline.run()` takes one already-resolved provider for the whole request,
not a `Router` - resolving *which* model that is, per `model_prefs`, happens
here, once, up front.
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import SecretStr

from halludetect.api.schemas import EvidenceSourceKind, ModelPrefsIn, ModelProvider, VerifyRequestIn
from halludetect.detect.schemas import ModelUsed
from halludetect.evidence.base import EvidenceSource
from halludetect.evidence.direct import DirectEvidence
from halludetect.evidence.none import NoEvidenceSource
from halludetect.evidence.web_search import WebSearchEvidence
from halludetect.llm.anthropic import DEFAULT_MODEL as ANTHROPIC_DEFAULT_MODEL
from halludetect.llm.anthropic import AnthropicProvider
from halludetect.llm.base import LLMProvider
from halludetect.llm.custom_openai_compat import CustomOpenAICompatProvider
from halludetect.llm.gemini import DEFAULT_MODEL as GEMINI_DEFAULT_MODEL
from halludetect.llm.gemini import GeminiProvider
from halludetect.llm.health import HealthTracker
from halludetect.llm.openai import DEFAULT_MODEL as OPENAI_DEFAULT_MODEL
from halludetect.llm.openai import OpenAIProvider
from halludetect.llm.openrouter import FreeModelCatalog, OpenRouterProvider
from halludetect.llm.router import Router
from halludetect.settings import Settings


class ResolutionError(Exception):
    """The request asked for an evidence/model configuration this deployment
    can't satisfy - e.g. `evidence_source: custom` with no plugin
    registered, or `model_prefs.provider: custom` with no base URL
    configured. Distinct from `LLMError`: this is a bad request (400), not a
    provider that was reachable but failed.
    """


@dataclass(frozen=True)
class _NamedProvider:
    provider_cls: type
    default_model: str
    settings_key: str


# One shared health tracker per process, not per request, so the circuit
# breaker (Phase 2.5) and success-rate ranking actually accumulate signal
# across requests instead of resetting on every call.
_openrouter_health = HealthTracker()

_NAMED_PROVIDERS: dict[ModelProvider, _NamedProvider] = {
    ModelProvider.GEMINI: _NamedProvider(GeminiProvider, GEMINI_DEFAULT_MODEL, "gemini_api_key"),
    ModelProvider.OPENAI: _NamedProvider(OpenAIProvider, OPENAI_DEFAULT_MODEL, "openai_api_key"),
    ModelProvider.ANTHROPIC: _NamedProvider(AnthropicProvider, ANTHROPIC_DEFAULT_MODEL, "anthropic_api_key"),
}


def _secret(value: SecretStr | None) -> str | None:
    return value.get_secret_value() if value is not None else None


def resolve_evidence_source(
    request: VerifyRequestIn,
    settings: Settings,
    custom_evidence_source: EvidenceSource | None,
) -> EvidenceSource:
    """Implements docs/contract.md's `evidence` vs `evidence_source` rules.

    `evidence` (non-empty) is authoritative and always wins. Otherwise
    `evidence_source` decides: `custom` needs a plugin registered on this
    deployment (Phase 3.4's hook isn't expressible over plain JSON - a
    caller can't hand us a Python callable in a request body), `web` is
    `WebSearchEvidence`, whose own missing-key check already degrades to
    the same behavior as `none` (docs/contract.md), and `none` is explicit.
    """
    if request.evidence:
        return DirectEvidence(request.evidence)

    if request.evidence_source == EvidenceSourceKind.CUSTOM:
        if custom_evidence_source is None:
            raise ResolutionError(
                "evidence_source: custom requested but no custom evidence source is registered on this deployment"
            )
        return custom_evidence_source

    if request.evidence_source == EvidenceSourceKind.WEB:
        return WebSearchEvidence(_secret(settings.tavily_api_key))

    return NoEvidenceSource()


def resolve_provider(model_prefs: ModelPrefsIn, settings: Settings) -> tuple[LLMProvider, ModelUsed]:
    """Resolves `model_prefs` to a single bound `LLMProvider` + `ModelUsed`.

    `provider: openrouter` (the default) goes through the free-model
    `Router` (`Router.pick_model()`: pinned model, else the top-ranked
    healthy free model) - this is the "auto-rotating free pool" that is
    plan.md's differentiator. Any other named provider is built directly
    from `model_prefs.user_api_key` (falling back to this deployment's own
    key if the caller didn't supply one) and `model_prefs.pinned_model`
    (falling back to that provider's default model).
    """
    if model_prefs.provider == ModelProvider.OPENROUTER:
        api_key = _secret(settings.openrouter_api_key)
        catalog = FreeModelCatalog(api_key)
        router = Router(
            catalog=catalog,
            provider_factory=lambda model: OpenRouterProvider(api_key, model),
            health=_openrouter_health,
            pinned_model=model_prefs.pinned_model,
        )
        model, provider = router.pick_model()
        return provider, ModelUsed(provider="openrouter", model=model)

    if model_prefs.provider == ModelProvider.CUSTOM:
        if not settings.custom_provider_base_url:
            raise ResolutionError(
                "model_prefs.provider: custom requested but no custom_provider_base_url is configured"
            )
        if not model_prefs.pinned_model:
            raise ResolutionError("model_prefs.provider: custom requires model_prefs.pinned_model")
        api_key = model_prefs.user_api_key or _secret(settings.custom_provider_api_key)
        provider = CustomOpenAICompatProvider(
            base_url=settings.custom_provider_base_url,
            model=model_prefs.pinned_model,
            api_key=api_key,
        )
        return provider, ModelUsed(provider="custom", model=model_prefs.pinned_model)

    named = _NAMED_PROVIDERS[model_prefs.provider]
    api_key = model_prefs.user_api_key or _secret(getattr(settings, named.settings_key))
    model = model_prefs.pinned_model or named.default_model
    provider = named.provider_cls(api_key, model)
    return provider, ModelUsed(provider=model_prefs.provider.value, model=model)
