"""OpenAI chat-completions backend."""
import httpx

from halludetect.llm._http import DEFAULT_TIMEOUT_S, raise_for_provider_error, wrap_transport_error
from halludetect.llm.base import LLMResponse, TokenUsage
from halludetect.llm.exceptions import LLMAuthError

DEFAULT_MODEL = "gpt-4o-mini"
BASE_URL = "https://api.openai.com/v1"


class OpenAIProvider:
    def __init__(self, api_key: str | None, model: str = DEFAULT_MODEL, base_url: str = BASE_URL):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> LLMResponse:
        if not self._api_key:
            raise LLMAuthError("openai: no API key configured")

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
            raise wrap_transport_error(exc, "openai") from exc

        raise_for_provider_error(response, "openai")
        data = response.json()
        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LLMResponse(
            text=choice,
            provider="openai",
            model=self._model,
            usage=TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            ),
        )

    def supports_json_schema(self) -> bool:
        return True
