"""Cache key computation (Phase 6.1): `sha256(answer, evidence, provider,
settings)` per plan.md.

Deliberately keyed on what the *caller asked for*, not on which concrete
model actually answered - two identical requests routed to different free
OpenRouter models by the pool rotation should still be treated as the same
cached question, since the caller never asked for a specific model.
`user_api_key` is included directly in the hashed input (never stored or
logged in the clear) so two callers using different keys - potentially
different accounts/billing - never share a cached answer.
"""
from __future__ import annotations

import hashlib
import json


def compute_cache_key(
    *,
    answer: str,
    question: str | None,
    evidence: list[str],
    evidence_source: str,
    model_provider: str,
    allow_free_pool: bool,
    pinned_model: str | None,
    user_api_key: str | None,
) -> str:
    payload = {
        "answer": answer,
        "question": question,
        "evidence": evidence,
        "evidence_source": evidence_source,
        "model_provider": model_provider,
        "allow_free_pool": allow_free_pool,
        "pinned_model": pinned_model,
        "user_api_key": user_api_key,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
