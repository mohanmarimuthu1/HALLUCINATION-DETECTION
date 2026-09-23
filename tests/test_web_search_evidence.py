"""WebSearchEvidence tests, run fully offline by monkeypatching httpx.post -
no live Tavily key or network required.
"""
import httpx
import pytest

from halludetect.evidence import web_search
from halludetect.evidence.base import EvidenceSource
from halludetect.evidence.web_search import WebSearchEvidence


def fake_response(status_code: int, json_body: dict) -> httpx.Response:
    request = httpx.Request("POST", "https://example.invalid")
    return httpx.Response(status_code, json=json_body, request=request)


def test_web_search_evidence_satisfies_evidence_source_protocol() -> None:
    assert isinstance(WebSearchEvidence(api_key="key"), EvidenceSource)


def test_no_api_key_returns_no_evidence_without_a_network_call(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("httpx.post must not be called with no API key")

    monkeypatch.setattr(web_search.httpx, "post", fail_if_called)
    source = WebSearchEvidence(api_key=None)

    assert source.fetch("some query") == []


def test_successful_search_returns_evidence_chunks(monkeypatch) -> None:
    monkeypatch.setattr(
        web_search.httpx,
        "post",
        lambda *a, **k: fake_response(
            200,
            {
                "results": [
                    {"content": "first result content", "url": "https://a.example"},
                    {"content": "second result content", "url": "https://b.example"},
                ]
            },
        ),
    )
    source = WebSearchEvidence(api_key="key")

    chunks = source.fetch("some query")

    assert [c.chunk_id for c in chunks] == ["web-0", "web-1"]
    assert [c.text for c in chunks] == ["first result content", "second result content"]
    assert all(c.source == "web" for c in chunks)


def test_results_are_capped_at_max_results(monkeypatch) -> None:
    monkeypatch.setattr(
        web_search.httpx,
        "post",
        lambda *a, **k: fake_response(
            200,
            {"results": [{"content": f"result {i}"} for i in range(10)]},
        ),
    )
    source = WebSearchEvidence(api_key="key", max_results=3)

    chunks = source.fetch("some query")

    assert len(chunks) == 3


def test_results_with_blank_content_are_skipped(monkeypatch) -> None:
    monkeypatch.setattr(
        web_search.httpx,
        "post",
        lambda *a, **k: fake_response(
            200,
            {"results": [{"content": "   "}, {"content": ""}, {"content": "real content"}]},
        ),
    )
    source = WebSearchEvidence(api_key="key")

    chunks = source.fetch("q")

    assert len(chunks) == 1
    assert chunks[0].text == "real content"


def test_http_error_status_degrades_to_no_evidence(monkeypatch) -> None:
    monkeypatch.setattr(web_search.httpx, "post", lambda *a, **k: fake_response(429, {"error": "rate limited"}))
    source = WebSearchEvidence(api_key="key")

    assert source.fetch("q") == []


def test_transport_error_degrades_to_no_evidence(monkeypatch) -> None:
    def raise_timeout(*args, **kwargs):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(web_search.httpx, "post", raise_timeout)
    source = WebSearchEvidence(api_key="key")

    assert source.fetch("q") == []


def test_invalid_json_degrades_to_no_evidence(monkeypatch) -> None:
    request = httpx.Request("POST", "https://example.invalid")
    response = httpx.Response(200, content=b"not json", request=request)
    monkeypatch.setattr(web_search.httpx, "post", lambda *a, **k: response)
    source = WebSearchEvidence(api_key="key")

    assert source.fetch("q") == []


def test_api_key_and_query_are_sent_in_the_request_body(monkeypatch) -> None:
    captured = {}

    def capture_post(url, *, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return fake_response(200, {"results": []})

    monkeypatch.setattr(web_search.httpx, "post", capture_post)
    source = WebSearchEvidence(api_key="secret-key", max_results=7, timeout=5.0)

    source.fetch("what is the capital of france")

    assert captured["json"]["api_key"] == "secret-key"
    assert captured["json"]["query"] == "what is the capital of france"
    assert captured["json"]["max_results"] == 7
    assert captured["timeout"] == 5.0


@pytest.mark.parametrize("bad_key", [None, ""])
def test_falsy_api_key_never_reaches_the_network(monkeypatch, bad_key) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("httpx.post must not be called")

    monkeypatch.setattr(web_search.httpx, "post", fail_if_called)
    source = WebSearchEvidence(api_key=bad_key)

    assert source.fetch("q") == []
