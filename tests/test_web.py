"""The browser page served at `/` - offline, no network."""
from fastapi.testclient import TestClient

from halludetect.api import main
from halludetect.settings import Settings, get_settings


def _client_with_secrets(monkeypatch) -> TestClient:
    settings = Settings(_env_file=None, openrouter_api_key="sk-or-secret-value", client_api_keys="client-secret-value")
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    main.app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(main.app)


def test_index_serves_the_page(monkeypatch):
    response = _client_with_secrets(monkeypatch).get("/")
    main.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "Check answer" in response.text
    for call in ('"POST", "/v1/verify"', '"POST", "/v1/chat"', '"GET", "/v1/models/stats"'):
        assert f"callApi({call}" in response.text


def test_index_never_carries_a_server_side_key(monkeypatch):
    """The page is public. Visitors enter their own access key; the service's
    keys must never be written into what it serves.
    """
    page = _client_with_secrets(monkeypatch).get("/").text
    main.app.dependency_overrides.clear()

    assert "client-secret-value" not in page
    assert "sk-or-secret-value" not in page


def test_index_is_not_part_of_the_api_schema():
    schema = TestClient(main.app).get("/openapi.json").json()
    assert "/" not in schema["paths"]
    assert "/v1/verify" in schema["paths"]
