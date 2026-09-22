"""structlog JSON logging with a per-request request_id.

Every log line must carry the request_id of the call that produced it so a
single `/v1/verify` request can be traced across the router (Phase 2),
evidence acquisition (Phase 3), and detection pipeline (Phase 4) in log
aggregation. The FastAPI layer (Phase 5) sets the request_id per-request via
`bind_request_id`; anything logged inside that request picks it up
automatically without threading an argument through every function call.
"""
from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from uuid import uuid4

import structlog

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def _add_request_id(logger, method_name, event_dict):
    request_id = _request_id.get()
    if request_id is not None:
        event_dict["request_id"] = request_id
    return event_dict


def configure_logging(level: int = logging.INFO) -> None:
    """Call once at process startup (API entrypoint, eval runner, etc.)."""
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_request_id,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def bind_request_id(request_id: str | None = None) -> str:
    """Set the request_id for the current context; generate one if omitted.

    Returns the id that was set, so callers can echo it back (e.g. in the
    AnalysisResult.request_id field of docs/contract.md).
    """
    resolved = request_id or str(uuid4())
    _request_id.set(resolved)
    return resolved


def get_request_id() -> str | None:
    return _request_id.get()
