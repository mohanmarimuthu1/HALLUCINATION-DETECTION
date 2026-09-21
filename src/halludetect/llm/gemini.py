"""Google Gemini (generateContent) backend."""
import httpx

from halludetect.llm._http import DEFAULT_TIMEOUT_S, raise_for_provider_error, wrap_transport_error
from halludetect.llm.base import LLMResponse, TokenUsage
from halludetect.llm.exceptions import LLMAuthError, LLMResponseError

DEFAULT_MODEL = "gemini-2.0-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider:
    def __init__(self, api_key: str | None, model: str = DEFAULT_MODEL, base_url: str = BASE_URL):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> LLMResponse:
        if not self._api_key:
            raise LLMAuthError("gemini: no API key configured")

        try:
            response = httpx.post(
                f"{self._base_url}/models/{self._model}:generateContent",
                headers={"x-goog-api-key": self._api_key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": max_tokens},
                },
                timeout=DEFAULT_TIMEOUT_S,
            )
        except httpx.HTTPError as exc:
            raise wrap_transport_error(exc, "gemini") from exc

        raise_for_provider_error(response, "gemini")
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMResponseError(f"gemini: no candidates in response: {data}")
        parts = candidates[0].get("content", {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts)
        usage = data.get("usageMetadata", {})
        return LLMResponse(
            text=text,
            provider="gemini",
            model=self._model,
            usage=TokenUsage(
                prompt_tokens=usage.get("promptTokenCount", 0),
                completion_tokens=usage.get("candidatesTokenCount", 0),
            ),
        )

    def supports_json_schema(self) -> bool:
        return True
