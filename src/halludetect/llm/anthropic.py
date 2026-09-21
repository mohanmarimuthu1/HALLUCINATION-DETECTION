"""Anthropic Messages API backend."""
import httpx

from halludetect.llm._http import DEFAULT_TIMEOUT_S, raise_for_provider_error, wrap_transport_error
from halludetect.llm.base import LLMResponse, TokenUsage
from halludetect.llm.exceptions import LLMAuthError

DEFAULT_MODEL = "claude-3-5-haiku-latest"
BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider:
    def __init__(self, api_key: str | None, model: str = DEFAULT_MODEL, base_url: str = BASE_URL):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> LLMResponse:
        if not self._api_key:
            raise LLMAuthError("anthropic: no API key configured")

        try:
            response = httpx.post(
                f"{self._base_url}/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": ANTHROPIC_VERSION,
                },
                json={
                    "model": self._model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=DEFAULT_TIMEOUT_S,
            )
        except httpx.HTTPError as exc:
            raise wrap_transport_error(exc, "anthropic") from exc

        raise_for_provider_error(response, "anthropic")
        data = response.json()
        text = "".join(block["text"] for block in data["content"] if block.get("type") == "text")
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            provider="anthropic",
            model=self._model,
            usage=TokenUsage(
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
            ),
        )

    def supports_json_schema(self) -> bool:
        # Anthropic has no native JSON-schema response mode as of this
        # writing; structured output goes through prompt-JSON + repair-retry
        # (Phase 2.4), not a provider-level guarantee.
        return False
