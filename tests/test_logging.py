"""structlog JSON logging tests - fully offline, no network. Verifies the
request_id contextvar propagates into rendered JSON log lines without being
threaded through call signatures, and that logging without a bound
request_id doesn't crash or fabricate one.
"""
import json

import pytest

from halludetect import logging as halludetect_logging


@pytest.fixture(autouse=True)
def _reset_request_id():
    token = halludetect_logging._request_id.set(None)
    yield
    halludetect_logging._request_id.reset(token)


@pytest.fixture(autouse=True)
def _configure():
    halludetect_logging.configure_logging()


def test_bind_request_id_generates_uuid_when_omitted():
    request_id = halludetect_logging.bind_request_id()
    assert request_id
    assert halludetect_logging.get_request_id() == request_id


def test_bind_request_id_uses_given_value():
    request_id = halludetect_logging.bind_request_id("abc-123")
    assert request_id == "abc-123"
    assert halludetect_logging.get_request_id() == "abc-123"


def test_get_request_id_is_none_before_bind():
    assert halludetect_logging.get_request_id() is None


def test_log_line_is_json_and_carries_request_id(capsys):
    halludetect_logging.bind_request_id("req-42")
    logger = halludetect_logging.get_logger("test")
    logger.info("hello", extra_field="value")

    captured = capsys.readouterr().out.strip()
    payload = json.loads(captured)
    assert payload["event"] == "hello"
    assert payload["extra_field"] == "value"
    assert payload["request_id"] == "req-42"
    assert payload["level"] == "info"


def test_log_line_without_bound_request_id_omits_it(capsys):
    logger = halludetect_logging.get_logger("test")
    logger.info("hello")

    captured = capsys.readouterr().out.strip()
    payload = json.loads(captured)
    assert "request_id" not in payload
