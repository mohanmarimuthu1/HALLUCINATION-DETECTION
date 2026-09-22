"""LLM provider error hierarchy.

Callers (the router in Phase 2, the pipeline in Phase 4) must branch on
exception type, never on substring-matching an error message or a sentinel
string in the response text - that string-matching pattern is the exact
class of bug that made v1's verifier unreliable.
"""


class LLMError(Exception):
    """Base class for all LLM provider failures."""


class LLMAuthError(LLMError):
    """No API key configured, or the provider rejected the key (401/403)."""


class LLMRateLimitError(LLMError):
    """Provider returned 429 / quota exhausted."""


class LLMTimeoutError(LLMError):
    """Request exceeded its timeout."""


class LLMResponseError(LLMError):
    """Provider returned a 2xx we could not parse, or a 4xx/5xx not covered above."""


class LLMSchemaValidationError(LLMResponseError):
    """Structured JSON output still didn't validate against the target
    schema after exhausting repair-retry attempts (Phase 2.4). Raised
    instead of returning a best-effort/partial parse - a caller must
    handle this explicitly, never treat missing structure as success.
    """
