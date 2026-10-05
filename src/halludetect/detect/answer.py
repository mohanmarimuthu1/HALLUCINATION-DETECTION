"""Answer generation for `/v1/chat`.

The answer produced here is the thing under test, not a source of truth:
it goes through the same pipeline as any caller-supplied answer, and its
claims are checked only against evidence (`evidence` or the configured
evidence source). Nothing in this module feeds back into verification.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from halludetect.llm.base import LLMProvider, LLMResponse
from halludetect.llm.exceptions import LLMResponseError

MAX_ANSWER_TOKENS = 1024


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


_INSTRUCTIONS = """You are a factual assistant. Answer the user's latest question directly
and concisely, in plain prose (no headings, no tables). State facts as
specific, self-contained sentences. If you do not know, say so instead of
guessing."""

_SOURCES_INSTRUCTIONS = """Answer using only the SOURCES below. If they do not contain the answer,
say that they don't."""


def build_prompt(question: str, history: Sequence[ChatTurn], evidence: Sequence[str]) -> str:
    parts = [_INSTRUCTIONS]
    if evidence:
        sources = "\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(evidence))
        parts.append(f"{_SOURCES_INSTRUCTIONS}\n\nSOURCES:\n{sources}")
    if history:
        lines = [f"{turn.role.upper()}: {turn.content}" for turn in history]
        parts.append("CONVERSATION SO FAR:\n" + "\n".join(lines))
    parts.append(f"USER: {question}\nASSISTANT:")
    return "\n\n".join(parts)


def generate_answer(
    provider: LLMProvider,
    question: str,
    *,
    history: Sequence[ChatTurn] = (),
    evidence: Sequence[str] = (),
) -> LLMResponse:
    """Raises `LLMResponseError` on an empty completion, so the caller's
    fail-over treats it like any other bad response.
    """
    response = provider.complete(build_prompt(question, history, evidence), max_tokens=MAX_ANSWER_TOKENS)
    if not response.text.strip():
        raise LLMResponseError("model returned an empty answer")
    return response
