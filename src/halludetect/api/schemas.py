"""API-facing request schemas for `POST /v1/verify`, mirroring
`docs/contract.md`'s request shape field-for-field.

Kept separate from `halludetect.detect.schemas` - those are the pipeline's
internal shapes (extraction/verification LLM call shapes, `AnalysisResult`);
this module is what actually gets validated against incoming JSON, before
`halludetect.api.resolve` turns it into the concrete objects the pipeline
needs.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from halludetect.detect.answer import ChatTurn
from halludetect.detect.schemas import AnalysisResult, ModelUsed

MAX_CHAT_HISTORY = 20


class ModelProvider(str, Enum):
    OPENROUTER = "openrouter"
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    NVIDIA = "nvidia"
    CUSTOM = "custom"


class EvidenceSourceKind(str, Enum):
    NONE = "none"
    WEB = "web"
    CUSTOM = "custom"


class ModelPrefsIn(BaseModel):
    provider: ModelProvider = ModelProvider.OPENROUTER
    allow_free_pool: bool = True
    pinned_model: str | None = None
    user_api_key: str | None = None


class VerifyRequestIn(BaseModel):
    answer: str
    question: str | None = None
    evidence: list[str] = Field(default_factory=list)
    evidence_source: EvidenceSourceKind
    model_prefs: ModelPrefsIn = Field(default_factory=ModelPrefsIn)


class ChatRequestIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=MAX_CHAT_HISTORY)
    evidence: list[str] = Field(default_factory=list)
    evidence_source: EvidenceSourceKind = EvidenceSourceKind.WEB
    model_prefs: ModelPrefsIn = Field(default_factory=ModelPrefsIn)


class ChatResult(BaseModel):
    request_id: str
    answer: str
    answer_model: ModelUsed
    answer_ms: int
    verification: AnalysisResult
