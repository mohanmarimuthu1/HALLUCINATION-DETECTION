"""Per-key auth for `POST /v1/verify` (Phase 5.2).

A static, comma-separated key list from `settings.client_api_keys` - a full
user/key management system is out of scope for v1 (see `process.md`).
Every request must present a valid key via `Authorization: Bearer <key>`;
a missing, malformed, or unrecognized key is 401, and no keys configured
at all means every request is rejected rather than silently let through.
"""
from __future__ import annotations

from fastapi import Depends, Header, HTTPException

from halludetect.settings import Settings, get_settings


def _valid_keys(settings: Settings) -> set[str]:
    if settings.client_api_keys is None:
        return set()
    raw = settings.client_api_keys.get_secret_value()
    return {key.strip() for key in raw.split(",") if key.strip()}


def require_api_key(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> str:
    """FastAPI dependency: returns the validated key, or raises 401."""
    valid_keys = _valid_keys(settings)
    if not valid_keys:
        raise HTTPException(status_code=401, detail="no client API keys configured for this deployment")

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing Authorization: Bearer <key> header")

    key = authorization.removeprefix("Bearer ").strip()
    if key not in valid_keys:
        raise HTTPException(status_code=401, detail="invalid API key")

    return key
