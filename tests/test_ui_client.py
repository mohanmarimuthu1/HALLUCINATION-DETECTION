"""ui/client.py tests (Phase 5.4) - offline, httpx monkeypatched."""
import httpx
import pytest

from halludetect.ui import client
from halludetect.ui.client import ApiError, parse_evidence, verify


class _Response:
    def __init__(self, status_code: int, body=None, text: str = ""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def _post_returning(monkeypatch, response, calls=None):
    def _post(url, *, json, headers, timeout):
        if calls is not None:
            calls.append({"url": url, "json": json, "headers": headers})
        return response

    monkeypatch.setattr(client.httpx, "post", _post)


def test_parse_evidence_splits_on_blank_lines():
    text = "First passage.\n\nSecond passage\nspans two lines.\r\n\r\n\n  \nThird."
    assert parse_evidence(text) == ["First passage.", "Second passage\nspans two lines.", "Third."]


def test_parse_evidence_blank_means_no_evidence():
    assert parse_evidence("  \n\n ") == []


def test_verify_sends_bearer_key_and_direct_evidence(monkeypatch):
    calls: list = []
    _post_returning(monkeypatch, _Response(200, {"verdict": "GROUNDED"}), calls)

    body = verify("http://api", "k-1", answer="A.", question="Q?", evidence=["e1"])

    assert body == {"verdict": "GROUNDED"}
    assert calls[0]["url"] == "http://api/v1/verify"
    assert calls[0]["headers"] == {"Authorization": "Bearer k-1"}
    assert calls[0]["json"] == {"answer": "A.", "question": "Q?", "evidence": ["e1"], "evidence_source": "none"}


def test_verify_omits_empty_question(monkeypatch):
    calls: list = []
    _post_returning(monkeypatch, _Response(200, {}), calls)
    verify("http://api", "k", answer="A.", question=None, evidence=[])
    assert "question" not in calls[0]["json"]


def test_verify_without_a_key_fails_before_any_request(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("must not call the API without a key")

    monkeypatch.setattr(client.httpx, "post", _boom)
    with pytest.raises(ApiError, match="No API key"):
        verify("http://api", None, answer="A.")


def test_unreachable_api_says_how_to_start_it(monkeypatch):
    def _refused(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(client.httpx, "post", _refused)
    with pytest.raises(ApiError) as excinfo:
        verify("http://api", "k", answer="A.")
    assert "Could not reach" in excinfo.value.message
    assert "uvicorn halludetect.api.main:app" in excinfo.value.hint


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, "rejected the key"),
        (429, "Rate limit"),
        (502, "LLM provider failed"),
        (503, "LLM provider failed"),
        (400, "rejected the request"),
        (500, "Unexpected HTTP 500"),
    ],
)
def test_error_statuses_become_api_errors_never_results(monkeypatch, status, expected):
    _post_returning(monkeypatch, _Response(status, {"detail": "boom"}))
    with pytest.raises(ApiError, match=expected) as excinfo:
        verify("http://api", "k", answer="A.")
    assert excinfo.value.status == status


def test_default_api_key_prefers_explicit_env(monkeypatch):
    monkeypatch.setenv("HALLUDETECT_API_KEY", " ui-key ")
    assert client.default_api_key() == "ui-key"


def test_default_api_key_falls_back_to_first_client_key(monkeypatch):
    from pydantic import SecretStr

    from halludetect import settings as settings_module

    monkeypatch.delenv("HALLUDETECT_API_KEY", raising=False)
    fake = settings_module.Settings(_env_file=None, client_api_keys=SecretStr("first, second"))
    monkeypatch.setattr(settings_module, "get_settings", lambda: fake)
    assert client.default_api_key() == "first"


def test_default_api_key_is_none_when_nothing_configured(monkeypatch):
    from halludetect import settings as settings_module

    monkeypatch.delenv("HALLUDETECT_API_KEY", raising=False)
    monkeypatch.delenv("CLIENT_API_KEYS", raising=False)
    fake = settings_module.Settings(_env_file=None)
    monkeypatch.setattr(settings_module, "get_settings", lambda: fake)
    assert client.default_api_key() is None


def test_chat_asks_for_web_evidence_when_none_is_given(monkeypatch):
    calls: list = []
    _post_returning(monkeypatch, _Response(200, {"answer": "a"}), calls)
    history = [{"role": "user", "content": str(i)} for i in range(25)]
    assert client.chat("http://api", "k", question="q", history=history) == {"answer": "a"}
    [call] = calls
    assert call["url"] == "http://api/v1/chat"
    assert call["json"]["evidence_source"] == "web"
    assert len(call["json"]["history"]) == client.MAX_HISTORY_TURNS
    assert call["json"]["history"][-1]["content"] == "24"


def test_chat_with_sources_checks_against_them_only(monkeypatch):
    calls: list = []
    _post_returning(monkeypatch, _Response(200, {}), calls)
    client.chat("http://api", "k", question="q", evidence=["src"])
    assert calls[0]["json"]["evidence"] == ["src"]
    assert calls[0]["json"]["evidence_source"] == "none"


def test_model_stats_sends_key_and_maps_errors(monkeypatch):
    seen: list = []

    def _get(url, *, headers, timeout):
        seen.append((url, headers))
        return _Response(401, {"detail": "invalid API key"})

    monkeypatch.setattr(client.httpx, "get", _get)
    with pytest.raises(ApiError, match="rejected the key"):
        client.model_stats("http://api", "k")
    assert seen == [("http://api/v1/models/stats", {"Authorization": "Bearer k"})]
    with pytest.raises(ApiError, match="No API key"):
        client.model_stats("http://api", None)
