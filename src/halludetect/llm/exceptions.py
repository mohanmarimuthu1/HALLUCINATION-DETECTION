"""LLM provider error hierarchy.

Callers (the router in Phase 2, the pipeline in Phase 4) must branch on
exception type, never on substring-matching an error message or a sentinel
string in the response text - that string-matching pattern is the exact
class of bug that made v1's verifier unreliable.
"""


class LLMError(Exception):
    """Base class for all LLM provider failures."""


class LLMAuthError(LLMError):
    """No API key configured, or the provider rejected the key (401)."""


class LLMModelAccessError(LLMError):
    """The key is valid but may not use this model (403). OpenRouter
    returns this for models restricted to specific clients; another model
    on the same key can still succeed, unlike `LLMAuthError`.
    """


class LLMRateLimitError(LLMError):
    """Provider returned 429 / quota exhausted."""


class LLMQuotaExhaustedError(LLMError):
    """429 whose limit resets far in the future: the account's quota is
    used up (OpenRouter's free tier allows 50 free-model requests a day),
    so every model on that key will fail until `reset_at` (epoch
    seconds). Not an `LLMRateLimitError`, so it is never retried.
    """

    def __init__(self, message: str, reset_at: float | None = None):
        super().__init__(message)
        self.reset_at = reset_at


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
