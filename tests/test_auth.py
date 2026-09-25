"""Per-key auth unit tests (Phase 5.2) - offline, calls the FastAPI
dependency function directly rather than going through a client."""
import pytest
from fastapi import HTTPException

from halludetect.api.auth import require_api_key
from halludetect.settings import Settings


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_valid_key_is_accepted():
    settings = _settings(client_api_keys="key-a,key-b")
    assert require_api_key("Bearer key-a", settings) == "key-a"
    assert require_api_key("Bearer key-b", settings) == "key-b"


def test_whitespace_around_keys_in_the_list_is_ignored():
    settings = _settings(client_api_keys=" key-a , key-b ")
    assert require_api_key("Bearer key-a", settings) == "key-a"


def test_no_keys_configured_rejects_everything():
    settings = _settings()
    with pytest.raises(HTTPException) as exc_info:
        require_api_key("Bearer anything", settings)
    assert exc_info.value.status_code == 401


def test_missing_header_is_401():
    settings = _settings(client_api_keys="key-a")
    with pytest.raises(HTTPException) as exc_info:
        require_api_key(None, settings)
    assert exc_info.value.status_code == 401


def test_header_without_bearer_prefix_is_401():
    settings = _settings(client_api_keys="key-a")
    with pytest.raises(HTTPException) as exc_info:
        require_api_key("key-a", settings)
    assert exc_info.value.status_code == 401


def test_unrecognized_key_is_401():
    settings = _settings(client_api_keys="key-a")
    with pytest.raises(HTTPException) as exc_info:
        require_api_key("Bearer wrong-key", settings)
    assert exc_info.value.status_code == 401
