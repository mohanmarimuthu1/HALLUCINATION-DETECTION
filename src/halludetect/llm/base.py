"""LLM provider protocol every backend (Gemini, OpenAI, Anthropic, generic
OpenAI-compatible, and later OpenRouter) must satisfy so the router (Phase 2)
and detection pipeline (Phase 4) can swap between them without caring which
one they're talking to.
"""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    usage: TokenUsage


@runtime_checkable
class LLMProvider(Protocol):
    """Every implementation must raise from halludetect.llm.exceptions on
    failure - never return a sentinel string and never swallow the error.
    """

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> LLMResponse:
        """Send prompt, return the completion. Raises LLMError subclasses."""
        ...

    def supports_json_schema(self) -> bool:
        """Whether this provider/model reliably honors a JSON response
        schema natively. Phase 2.4's capability probe is the authoritative
        runtime check; this is the static, provider-level default used
        before that probe has run.
        """
        ...
