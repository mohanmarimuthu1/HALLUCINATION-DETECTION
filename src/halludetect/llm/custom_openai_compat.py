"""Generic OpenAI-compatible chat-completions backend, for any endpoint
that implements the same /chat/completions shape (local models, other
hosted providers, etc.). Caller must supply base_url and model; there is
no sane default for either.
"""
import httpx

from halludetect.llm._http import (
    DEFAULT_TIMEOUT_S,
    extract_chat_content,
    raise_for_provider_error,
    wrap_transport_error,
)
from halludetect.llm.base import LLMResponse, TokenUsage


class CustomOpenAICompatProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        supports_json_schema: bool = False,
        *,
        name: str = "custom_openai_compat",
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._supports_json_schema = supports_json_schema
        self._name = name
        self._timeout_s = timeout_s

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> LLMResponse:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

        try:
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                },
                timeout=self._timeout_s,
            )
        except httpx.HTTPError as exc:
            raise wrap_transport_error(exc, self._name) from exc

        raise_for_provider_error(response, self._name)
        data = response.json()
        text = extract_chat_content(data, self._name, self._model)
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            provider=self._name,
            model=self._model,
            usage=TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            ),
        )

    def supports_json_schema(self) -> bool:
        return self._supports_json_schema
