"""Structured-JSON capability probe (Phase 2.4).

None of the four hand-written providers or OpenRouter's free pool can be
trusted to honor a native JSON-schema response mode - `supports_json_schema()`
is only ever a static, pre-probe default. This module is the actual probe:
ask the model for JSON matching a Pydantic schema via prompting, validate
strictly, and repair-retry a bounded number of times if it comes back wrong.
It never returns a best-effort parse - if the model can't produce valid
JSON within `max_repairs` retries, the caller gets `LLMSchemaValidationError`
and has to handle that explicitly.
"""
from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from halludetect.llm.base import LLMProvider
from halludetect.llm.exceptions import LLMSchemaValidationError

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

_SchemaT = TypeVar("_SchemaT", bound=BaseModel)


def _extract_json(text: str) -> object | None:
    candidates = [text.strip()]
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        candidates.insert(0, fence_match.group(1).strip())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def complete_structured(
    provider: LLMProvider,
    prompt: str,
    schema: type[_SchemaT],
    *,
    max_tokens: int = 1024,
    max_repairs: int = 2,
) -> tuple[_SchemaT, bool]:
    """Returns `(instance, honored_on_first_try)`.

    `honored_on_first_try` is the probe's real signal: True means this
    model returned schema-valid JSON with no repair round needed, so
    callers (the router's model ranking, eventually) can prefer it over
    one that only gets there after retries.
    """
    schema_json = json.dumps(schema.model_json_schema())
    base_instruction = (
        f"{prompt}\n\nRespond with ONLY valid JSON matching this schema, "
        f"no prose, no markdown code fences:\n{schema_json}"
    )

    last_text = ""
    last_error = ""
    for attempt in range(max_repairs + 1):
        if attempt == 0:
            query = base_instruction
        else:
            query = (
                f"{base_instruction}\n\nYour previous response did not satisfy the schema.\n"
                f"Previous response:\n{last_text}\n\nValidation error:\n{last_error}\n\n"
                "Return corrected JSON only."
            )

        response = provider.complete(query, max_tokens=max_tokens)
        last_text = response.text
        parsed = _extract_json(last_text)
        if parsed is None:
            last_error = "response was not valid JSON"
            continue

        try:
            instance = schema.model_validate(parsed)
        except ValidationError as exc:
            last_error = str(exc)
            continue

        return instance, attempt == 0

    raise LLMSchemaValidationError(
        f"structured output for {schema.__name__} still invalid after {max_repairs} repair attempts: {last_error}"
    )
