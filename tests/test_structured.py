"""Structured-output capability probe tests (Phase 2.4) - offline, using
FakeProvider instead of a real model. Covers the honored-on-first-try
signal, repair-retry recovery, and exhaustion raising instead of guessing."""
import pytest
from pydantic import BaseModel

from halludetect.llm.exceptions import LLMSchemaValidationError
from halludetect.llm.fake import FakeProvider, fake_response
from halludetect.llm.structured import complete_structured


class Verdict(BaseModel):
    label: str
    confidence: float


def test_valid_json_on_first_try():
    provider = FakeProvider([fake_response('{"label": "SUPPORTED", "confidence": 0.9}')])
    instance, honored_first_try = complete_structured(provider, "verify this", Verdict)
    assert instance == Verdict(label="SUPPORTED", confidence=0.9)
    assert honored_first_try is True
    assert provider.call_count == 1


def test_valid_json_inside_markdown_fence():
    provider = FakeProvider([fake_response('```json\n{"label": "SUPPORTED", "confidence": 0.5}\n```')])
    instance, honored_first_try = complete_structured(provider, "verify this", Verdict)
    assert instance.label == "SUPPORTED"
    assert honored_first_try is True


def test_repair_retry_recovers_on_second_attempt():
    provider = FakeProvider(
        [
            fake_response("not json at all"),
            fake_response('{"label": "CONTRADICTED", "confidence": 0.2}'),
        ]
    )
    instance, honored_first_try = complete_structured(provider, "verify this", Verdict)
    assert instance.label == "CONTRADICTED"
    assert honored_first_try is False
    assert provider.call_count == 2


def test_exhausted_repairs_raises_instead_of_guessing():
    provider = FakeProvider([fake_response("garbage")])
    with pytest.raises(LLMSchemaValidationError):
        complete_structured(provider, "verify this", Verdict, max_repairs=2)
    assert provider.call_count == 3


def test_schema_mismatch_triggers_repair():
    provider = FakeProvider(
        [
            fake_response('{"label": "SUPPORTED"}'),
            fake_response('{"label": "SUPPORTED", "confidence": 0.7}'),
        ]
    )
    instance, honored_first_try = complete_structured(provider, "verify this", Verdict)
    assert instance.confidence == 0.7
    assert honored_first_try is False
