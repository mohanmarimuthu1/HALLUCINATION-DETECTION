"""Smoke tests for the Streamlit demo page (Phase 5.4), run in-process with
Streamlit's AppTest. The API client is stubbed, so no server is needed.

The property that matters most is the last test: a failed API call shows
an error and no verdict. The legacy v1 app rendered "PARTIALLY SUPPORTED,
60%" when every one of its model calls had failed.
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from halludetect.ui import client  # noqa: E402
from halludetect.ui.client import ApiError  # noqa: E402

_APP = str(Path(__file__).resolve().parents[1] / "src" / "halludetect" / "ui" / "app.py")

_GROUNDED = {
    "request_id": "r1",
    "verdict": "GROUNDED",
    "reason": None,
    "p_hallucinated": 0.0,
    "groundedness": 1.0,
    "groundedness_ci": [0.44, 1.0],
    "claims": [
        {
            "claim_id": "claim-0",
            "text": "The tower is 330 metres tall.",
            "label": "SUPPORTED",
            "confidence": 1.0,
            "evidence_chunk_ids": ["direct-0"],
            "quote": "It is 330 metres (1,083 ft) tall.",
            "quote_verified": True,
        }
    ],
    "n_verifiable_claims": 3,
    "model_used": {"provider": "openrouter", "model": "stealth/space-bunny-alpha"},
    "cost_usd": 0.0,
    "timings_ms": {"total": 1234, "retrieval": 0, "extraction": 600, "verification": 600},
    "calibration_version": "heuristic-v0",
}


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(client, "is_healthy", lambda url, **kw: True)
    monkeypatch.setattr(client, "default_api_key", lambda: "test-key")


def _run() -> AppTest:
    at = AppTest.from_file(_APP, default_timeout=30)
    at.run()
    return at


def test_page_loads_without_exceptions():
    at = _run()
    assert not at.exception
    assert [b.label for b in at.button] == ["Verify"]


def test_loading_an_example_fills_the_inputs():
    at = _run()
    at.selectbox(key="example").select("No evidence - the service must refuse to guess").run()
    assert "330 metres" in at.text_area(key="answer").value
    assert at.text_area(key="evidence").value == ""


def test_verify_renders_the_api_result(monkeypatch):
    sent: dict = {}

    def _verify(api_url, api_key, **kwargs):
        sent.update(kwargs, api_key=api_key)
        return _GROUNDED

    monkeypatch.setattr(client, "verify", _verify)
    at = _run()
    at.text_area(key="answer").input("The tower is 330 metres tall.")
    at.text_area(key="evidence").input("Para one.\n\nPara two.")
    at.button[0].click().run()

    assert not at.exception
    assert sent["evidence"] == ["Para one.", "Para two."]
    assert sent["api_key"] == "test-key"
    assert any("GROUNDED" in s.value for s in at.main.success)
    assert any("330 metres (1,083 ft)" in m.value for m in at.markdown)


def test_not_verifiable_explains_the_reason(monkeypatch):
    result = dict(_GROUNDED, verdict="NOT_VERIFIABLE", reason="no_evidence_configured", claims=[])
    monkeypatch.setattr(client, "verify", lambda *a, **k: result)
    at = _run()
    at.text_area(key="answer").input("Some answer.")
    at.button[0].click().run()

    assert any("No evidence was supplied" in i.value for i in at.main.info)
    assert [m.value for m in at.metric if m.label == "Groundedness"] == ["-"]


def test_a_failed_call_shows_an_error_and_never_a_verdict(monkeypatch):
    def _fail(*args, **kwargs):
        raise ApiError("The LLM provider failed: every model is down", hint="Retry.", status=502)

    monkeypatch.setattr(client, "verify", _fail)
    at = _run()
    at.text_area(key="answer").input("Some answer.")
    at.button[0].click().run()

    assert not at.exception
    assert any("LLM provider failed" in e.value for e in at.main.error)
    # at.main, not at: the sidebar's own "API is running" is an st.success.
    assert not at.main.success and not at.main.warning and not at.main.info
    assert not [m for m in at.main.metric if m.label == "Groundedness"]


def test_repo_root_app_py_opens_the_v2_ui():
    """`streamlit run app.py` is the command people already use; it must
    open this UI, rendered once, not the legacy v1 app.
    """
    root_app = str(Path(__file__).resolve().parents[1] / "app.py")
    at = AppTest.from_file(root_app, default_timeout=30)
    at.run()

    assert not at.exception
    assert [b.label for b in at.button] == ["Verify"]
    assert [t.value for t in at.title] == ["HALLUDETECT"]
