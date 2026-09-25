"""OpenRouter backend: a single-model completion provider plus the free-model
catalog that the router (router.py) rotates across.

Two separate concerns live here on purpose:
- `OpenRouterProvider` talks to one pinned model, exactly like the other
  four providers - it doesn't know about rotation or health.
- `FreeModelCatalog` only knows how to fetch and cache the list of
  currently-free model ids. The router is what combines the two with
  health tracking (health.py) to decide which model to hand a
  `OpenRouterProvider` for on a given request.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx

from halludetect.llm._http import DEFAULT_TIMEOUT_S, extract_chat_content, raise_for_provider_error, wrap_transport_error
from halludetect.llm.base import LLMResponse, TokenUsage
from halludetect.llm.exceptions import LLMAuthError

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openrouter/auto"
CATALOG_TTL_S = 24 * 60 * 60


class OpenRouterProvider:
    def __init__(self, api_key: str | None, model: str = DEFAULT_MODEL, base_url: str = BASE_URL):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> LLMResponse:
        if not self._api_key:
            raise LLMAuthError("openrouter: no API key configured")

        try:
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                },
                timeout=DEFAULT_TIMEOUT_S,
            )
        except httpx.HTTPError as exc:
            raise wrap_transport_error(exc, "openrouter") from exc

        raise_for_provider_error(response, "openrouter")
        data = response.json()
        text = extract_chat_content(data, "openrouter", self._model)
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            provider="openrouter",
            model=self._model,
            usage=TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            ),
        )

    def supports_json_schema(self) -> bool:
        # Free models vary wildly in whether they honor response_format.
        # Phase 2.4's capability probe is the real answer for a given model;
        # this default stays conservative until that probe has run.
        return False


def _is_free(pricing: dict) -> bool:
    try:
        return float(pricing.get("prompt", 1)) == 0.0 and float(pricing.get("completion", 1)) == 0.0
    except (TypeError, ValueError):
        return False


def fetch_free_models(api_key: str | None) -> list[str]:
    """Fetch OpenRouter's /models list and return ids priced at zero.

    Raises the same halludetect.llm.exceptions hierarchy as every other
    provider call - a failed catalog fetch is a router-level failure, not
    a silent empty list, so the router can fail over instead of assuming
    "no free models exist."
    """
    try:
        response = httpx.get(
            f"{BASE_URL}/models",
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=DEFAULT_TIMEOUT_S,
        )
    except httpx.HTTPError as exc:
        raise wrap_transport_error(exc, "openrouter") from exc

    raise_for_provider_error(response, "openrouter")
    data = response.json()
    return [m["id"] for m in data.get("data", []) if _is_free(m.get("pricing", {}))]


@dataclass
class FreeModelCatalog:
    """Caches the free-model id list, refreshed at most once per `ttl_s`."""

    api_key: str | None
    ttl_s: float = CATALOG_TTL_S
    _models: list[str] = field(default_factory=list)
    _fetched_at: float | None = None

    def get_models(self, *, force_refresh: bool = False) -> list[str]:
        now = time.monotonic()
        stale = self._fetched_at is None or (now - self._fetched_at) >= self.ttl_s
        if force_refresh or stale:
            self._models = fetch_free_models(self.api_key)
            self._fetched_at = now
        return list(self._models)
