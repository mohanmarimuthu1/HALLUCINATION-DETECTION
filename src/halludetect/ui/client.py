"""HTTP client the demo UI uses to call `/v1/verify`, `/v1/chat` and
`/v1/models/stats`.

Kept free of Streamlit imports so it is unit-testable on its own.

Every failure is raised as `ApiError` with a message a person can act on,
never turned into a result. That is the property the legacy v1 app
lacked: when all of its model calls failed it still rendered a verdict
("PARTIALLY SUPPORTED, 60%") built from hardcoded fallback scores. A
failed call here can only ever show up as an error.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_API_URL = "http://127.0.0.1:8000"
# Extraction plus verification against a free reasoning model regularly
# takes 15-45s (golden set B p95 was ~46s); leave room above that.
DEFAULT_TIMEOUT_S = 180.0

# Matches the API's own cap (api.schemas.MAX_CHAT_HISTORY).
MAX_HISTORY_TURNS = 20

START_API_HINT = "Start it with: .venv/Scripts/python.exe -m uvicorn halludetect.api.main:app"


@dataclass
class ApiError(Exception):
    message: str
    hint: str = ""
    status: int | None = None

    def __str__(self) -> str:
        return self.message


def default_api_url() -> str:
    return os.environ.get("HALLUDETECT_API_URL", DEFAULT_API_URL).rstrip("/")


def default_api_key() -> str | None:
    """`HALLUDETECT_API_KEY` if set, else the first of the service's own
    `CLIENT_API_KEYS`. The fallback is for the local case, where the UI
    and the API run from the same `.env` and a second key would just be
    one more thing to configure. A UI deployed separately from the API
    should get its own key through `HALLUDETECT_API_KEY`.
    """
    explicit = os.environ.get("HALLUDETECT_API_KEY")
    if explicit:
        return explicit.strip()

    from halludetect.settings import get_settings

    configured = get_settings().client_api_keys
    if configured is None:
        return None
    first = configured.get_secret_value().split(",")[0].strip()
    return first or None


def parse_evidence(text: str) -> list[str]:
    """One passage per blank-line-separated block. Blank input means no
    evidence, which the API answers with NOT_VERIFIABLE - it never guesses.
    """
    blocks = [block.strip() for block in text.replace("\r\n", "\n").split("\n\n")]
    return [block for block in blocks if block]


def is_healthy(base_url: str, *, timeout_s: float = 3.0) -> bool:
    try:
        response = httpx.get(f"{base_url}/healthz", timeout=timeout_s)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        return response.text[:300]
    return detail if isinstance(detail, str) else str(detail)


def _require_key(api_key: str | None) -> str:
    if not api_key:
        raise ApiError(
            "No API key to send.",
            hint="Set CLIENT_API_KEYS in .env (the service's own key list), or HALLUDETECT_API_KEY for this UI.",
        )
    return api_key


def _request(
    method: str,
    url: str,
    api_key: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout_s: float,
) -> dict[str, Any]:
    base_url = url.split("/v1/")[0]
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        if method == "POST":
            response = httpx.post(url, json=payload, headers=headers, timeout=timeout_s)
        else:
            response = httpx.get(url, headers=headers, timeout=timeout_s)
    except httpx.ConnectError as exc:
        raise ApiError(f"Could not reach the API at {base_url}.", hint=START_API_HINT) from exc
    except httpx.TimeoutException as exc:
        raise ApiError(
            f"The API did not answer within {timeout_s:.0f}s.",
            hint="The free model pool is slow or overloaded. Try again in a minute.",
        ) from exc
    except httpx.HTTPError as exc:
        raise ApiError(f"Request to the API failed: {exc}") from exc

    status = response.status_code
    if status == 200:
        body: dict[str, Any] = response.json()
        return body
    if status == 401:
        raise ApiError(
            "The API rejected the key.",
            hint="The key must be one of CLIENT_API_KEYS in the service's .env.",
            status=status,
        )
    if status == 429:
        raise ApiError(
            "Rate limit reached for this key.",
            hint="Wait a few seconds, or raise RATE_LIMIT_CAPACITY / RATE_LIMIT_REFILL_PER_S in .env.",
            status=status,
        )
    if status in (502, 503):
        raise ApiError(
            f"The LLM provider failed: {_detail(response)}",
            hint="Usually a free model being briefly unavailable. Retry; a different model may be picked.",
            status=status,
        )
    if status in (400, 422):
        raise ApiError(f"The API rejected the request: {_detail(response)}", status=status)
    raise ApiError(f"Unexpected HTTP {status} from the API: {_detail(response)}", status=status)


def verify(
    base_url: str,
    api_key: str | None,
    *,
    answer: str,
    question: str | None = None,
    evidence: list[str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    key = _require_key(api_key)
    payload: dict[str, Any] = {"answer": answer, "evidence": evidence or [], "evidence_source": "none"}
    if question:
        payload["question"] = question
    return _request("POST", f"{base_url}/v1/verify", key, payload=payload, timeout_s=timeout_s)


def chat(
    base_url: str,
    api_key: str | None,
    *,
    question: str,
    history: list[dict[str, str]] | None = None,
    evidence: list[str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """With no evidence this asks for `web`, which the service treats as
    `none` (NOT_VERIFIABLE) unless it has a search key.
    """
    key = _require_key(api_key)
    payload: dict[str, Any] = {
        "question": question,
        "history": (history or [])[-MAX_HISTORY_TURNS:],
        "evidence": evidence or [],
        "evidence_source": "none" if evidence else "web",
    }
    return _request("POST", f"{base_url}/v1/chat", key, payload=payload, timeout_s=timeout_s)


def model_stats(base_url: str, api_key: str | None, *, timeout_s: float = 10.0) -> dict[str, Any]:
    return _request("GET", f"{base_url}/v1/models/stats", _require_key(api_key), timeout_s=timeout_s)
