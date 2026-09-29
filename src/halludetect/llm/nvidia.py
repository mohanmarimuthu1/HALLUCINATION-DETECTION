"""NVIDIA-hosted models (build.nvidia.com), a second free pool behind
OpenRouter's.

The endpoint is OpenAI-compatible. Its `/models` listing can't be used to
discover a pool the way OpenRouter's pricing field is: most listed chat
models return 404 (retired) and others take over a minute per call, so the
pool is an explicit list (`settings.nvidia_models`) of models checked live
against the real pipeline.
"""
from __future__ import annotations

from halludetect.llm.base import LLMResponse
from halludetect.llm.custom_openai_compat import CustomOpenAICompatProvider
from halludetect.llm.exceptions import LLMAuthError

BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"


class NvidiaProvider(CustomOpenAICompatProvider):
    def __init__(self, api_key: str | None, model: str = DEFAULT_MODEL):
        super().__init__(BASE_URL, model, api_key, name="nvidia")

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> LLMResponse:
        if not self._api_key:
            raise LLMAuthError("nvidia: no API key configured")
        return super().complete(prompt, max_tokens=max_tokens)
