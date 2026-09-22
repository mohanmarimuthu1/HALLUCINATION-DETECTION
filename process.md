# Process Log

Tracks what has been done, what is pending, and the next step. Update this
file at the end of every work session — do not let it drift from reality.

## Status: Phase 0 complete (spec lock). Phase 1 complete (all 3 tasks, exit criterion verified).

## What is done

- **Phase 1.3 — structlog JSON logging with `request_id` contextvar**:
  - `src/halludetect/logging.py` — `configure_logging()` sets up
    `structlog` with a JSON renderer (`structlog.processors.JSONRenderer`),
    ISO timestamps, and log level; `get_logger(name)` returns a bound
    logger. A private `ContextVar[str | None]` holds the current
    `request_id`; a processor (`_add_request_id`) injects it into every
    log line's JSON payload when set, and omits the key entirely when not
    (no fabricated ID). `bind_request_id(request_id=None)` sets the
    contextvar - generating a `uuid4` if the caller passes nothing - and
    returns whatever ID was set, so Phase 5's API layer can bind one per
    request and echo it back as `AnalysisResult.request_id`
    (`docs/contract.md`) without threading it through every function call.
  - **Verification**: `tests/test_logging.py`, 5 tests, all offline (no
    network, no live keys): request_id generation when omitted, explicit
    request_id round-trip, `get_request_id()` returns `None` pre-bind,
    a bound request_id appears in the rendered JSON log line (parsed with
    `json.loads` via `capsys`), and logging with no bound request_id
    produces valid JSON with the key absent rather than empty/null.
  - Added `structlog>=24.0` to `pyproject.toml` dependencies.
- **Phase 1 blocker resolved — `setup.py`/`pip install -e .` collision**:
  renamed the legacy interactive init script `setup.py` -> `init_legacy_app.py`
  (`git mv`, preserves history) since `setuptools` was picking it up as a
  legacy build hook and crashing with `UnicodeEncodeError` on Windows'
  `cp1252` console encoding before `pyproject.toml`-based install could even
  run. Updated `README.md`'s two references (`project structure` listing,
  `python setup.py` quick-start command) to match. No behavior change to
  the script itself.
  - **Verification**: `pip install -e .` now succeeds end-to-end in
    `.venv` (previously failed). Re-imported `halludetect`, `halludetect.settings`,
    `halludetect.logging`, and all four LLM provider modules with no
    `PYTHONPATH` manipulation - confirms Phase 1's actual exit criterion
    ("swapping provider is a config change, not a code change," which
    presupposes the package installs normally). Full test suite (20 tests:
    15 provider + 5 logging) still passes after the install.
- **Phase 1.2 — `LLMProvider` Protocol + Gemini/OpenAI/Anthropic/generic-OpenAI-compatible implementations**:
  - `src/halludetect/llm/base.py` — `LLMProvider` Protocol (`@runtime_checkable`):
    `complete(prompt, *, max_tokens) -> LLMResponse` and `supports_json_schema() -> bool`.
    `LLMResponse`/`TokenUsage` are frozen dataclasses.
  - `src/halludetect/llm/exceptions.py` — `LLMError` base with
    `LLMAuthError`/`LLMRateLimitError`/`LLMTimeoutError`/`LLMResponseError`
    subclasses. Every provider raises one of these on failure; none of
    them return a sentinel string or swallow the error — that string-based
    failure signaling is the exact class of bug Phase 6.2 exists to avoid,
    so it was built right from the start rather than retrofitted later.
  - `src/halludetect/llm/_http.py` — shared status-code -> exception
    classification (401/403 -> auth, 429 -> rate limit, timeout exception
    -> timeout, else -> response error) so all four providers classify
    failures identically instead of each reimplementing it.
  - `src/halludetect/llm/openai.py`, `gemini.py`, `anthropic.py`,
    `custom_openai_compat.py` — one class per provider, each a thin
    `httpx.post` wrapper against that provider's native REST shape (not an
    SDK dependency). `supports_json_schema()` is a static per-provider
    default (`True` for OpenAI/Gemini, `False` for Anthropic which has no
    native JSON-schema response mode, caller-supplied for the generic
    compat provider) - Phase 2.4's capability probe is the authoritative
    runtime check, this is just the pre-probe default.
  - **Verification**: added `tests/test_llm_providers.py`, 15 tests, all
    passing, fully offline (`httpx.post` monkeypatched per provider, no
    live API keys, no network). Covers: successful response parsing per
    provider, missing-API-key -> `LLMAuthError`, HTTP 401 -> `LLMAuthError`,
    HTTP 429 -> `LLMRateLimitError`, timeout exception -> `LLMTimeoutError`,
    and the `supports_json_schema()` default per provider. Also confirmed
    all four provider classes structurally satisfy `isinstance(x,
    LLMProvider)` via the runtime-checkable Protocol.
  - Added `httpx` to `pyproject.toml` dependencies and `pytest` under a new
    `[project.optional-dependencies] dev` group; added
    `[tool.pytest.ini_options]` with `pythonpath = ["src"]` and
    `testpaths = ["tests"]` so `pytest` runs correctly without manually
    setting `PYTHONPATH` (this only fixes *test* discovery, not the
    `pip install -e .` blocker below, which is a separate mechanism).

- **Phase 1.1 — `src/` layout, `pyproject.toml`, `settings.py`**:
  - `pyproject.toml` at repo root: package name `halludetect`,
    `requires-python = ">=3.11,<3.13"` per `plan.md`, setuptools build
    backend, package discovered under `src/`.
  - `src/halludetect/__init__.py` — package entry point, no logic yet.
  - `src/halludetect/settings.py` — `pydantic-settings` `Settings` class
    with one `SecretStr | None` field per provider key
    (`openrouter_api_key`, `gemini_api_key`, `openai_api_key`,
    `anthropic_api_key`, `custom_provider_api_key` + `custom_provider_base_url`,
    `tavily_api_key` for the Phase 0.2 web-search decision), loaded from
    `.env` via `env_file` config, `get_settings()` cached with `lru_cache`.
    Verified `SecretStr` masks values in `repr()`/logs while
    `get_secret_value()` still returns the real value when needed.
  - **Verification**: ran under the project's existing `.venv` (Python
    3.11.4, which satisfies the `<3.13` pin — the system `python` on this
    machine is 3.14, which does *not* satisfy it, so `.venv` must be used
    for this package). Imported `halludetect.settings` via
    `PYTHONPATH=src` and confirmed settings load, defaults are `None`
    with no `.env` present, and a real value passed as an env var is
    correctly masked in `repr()` but retrievable via `get_secret_value()`.
  - The `setup.py`/`pip install -e .` collision noted here previously is
    now resolved — see "Phase 1 blocker resolved" above.

- **Repo hygiene**: removed committed binaries/caches that never belonged in
  git (`chroma_db/` vector store, `__pycache__/*.pyc`, `config_error.txt`
  debug dump) and added them to `.gitignore`. Deleted one-off local debug
  artifacts (`demo_out.txt`, `files.zip`, `model_test_results.txt`,
  `models.txt`, `test_25.txt`, `test_api_out.txt`, `test_error.txt`,
  `test_all_models.py`, `test_demo.py`) that were scratch output, not tests.
- **Security fix**: `config.py` no longer hardcodes live API keys. Keys now
  load from environment variables via `python-dotenv` (`GOOGLE_API_KEY`,
  `OPENROUTER_API_KEY_1`, `OPENROUTER_API_KEY_2`). Added `.env.example` as
  the template; real values go in a local, gitignored `.env`.
  Note: the previously hardcoded keys were already committed to git history
  in earlier commits and are considered burned — rotate/revoke them at the
  provider if not already done; deleting them from the working tree does not
  remove them from history.
- **Bug fix**: removed the "self-learning" block in `app.py` that appended
  every low-risk-scored answer back into `data/knowledge_base.txt` as
  trusted fact and re-indexed it into the vector store. This let hallucinated
  or wrong answers get treated as ground truth on the next query
  (self-reinforcing data poisoning). `data/knowledge_base.txt` was reverted
  to its last clean, non-poisoned commit.
- **Phase 0 — Spec lock** (see `plan.md`):
  - `docs/contract.md` — frozen `AnalysisResult` response schema, verdict
    enum, request schema, and the three non-negotiable contract rules
    (low claim count forces `NOT_VERIFIABLE`; no evidence never falls back
    to model knowledge; `SUPPORTED` requires a verified quote or gets
    downgraded server-side).
  - `docs/openapi.yaml` — OpenAPI 3.0.3 spec for `POST /v1/verify` and
    `GET /healthz`, validated to parse cleanly.
  - Web-search provider decision (0.2): **Tavily**, for its perpetual free
    tier and LLM-ready output vs. Serper's raw-SERP, trial-credit-only model.
    Documented in `docs/contract.md`.

## What is pending

Phase 1 is complete. Phases 2-8 in `plan.md` are pending — the v2 rewrite
is only just started. Note the four providers built in 1.2 and the logging
in 1.3 are not wired into anything yet - there is no router, no pipeline,
nothing calls them outside the tests. That wiring is Phase 2 (router
chain) and Phase 4 (detection pipeline). The current root-level code
(`app.py`, `config.py`, `detection/`, `rag/`, `knowledge_base/`) is the
**legacy v1 app** described in `plan.md`'s Context section; it is not yet
superseded and still runs, but it is not where new work should go.

Known outstanding issue not yet fixed in the legacy app: the hardcoded API
keys committed in prior git history are still exposed in git log/GitHub even
though `config.py` no longer contains them on disk. They should be treated
as compromised.

## How this was done

Phase 0: read `plan.md`'s API contract section (frozen request/response
shape) and transcribed it verbatim into `docs/contract.md` as prose +
tables, then mirrored the same shape into `docs/openapi.yaml` using
`$ref`-based `components/schemas` so the two files cannot drift silently.
No provider or detection code was written — Phase 0 is spec-only per the
plan.

Phase 1.1: added `pyproject.toml` + `src/halludetect/` per the plan's
target repo layout, with `settings.py` as a thin `pydantic-settings`
wrapper — one `SecretStr` field per provider, nothing else, since 1.1 is
scoped to layout/config, not provider logic (that's 1.2). Verified by
import, not by `pip install`, because of the `setup.py` collision noted
above.

Phase 1.3: wrote `configure_logging()`/`get_logger()`/`bind_request_id()`
around a single module-level `ContextVar` rather than a logging adapter
per call site, so any code path (router, pipeline, API middleware) can
bind a request_id once per request and every log line downstream picks
it up automatically. Verified by parsing rendered JSON log lines in
tests, not by eyeballing console output. Also fixed the `setup.py`
naming collision flagged as a Phase 1.1 blocker (`git mv` to
`init_legacy_app.py`) so `pip install -e .` - Phase 1's actual exit
criterion - could be verified for real instead of deferred again.

Phase 1.2: defined the `LLMProvider` Protocol and exception hierarchy
first, then wrote each provider against that shape using raw `httpx`
calls to each vendor's REST API directly rather than pulling in four
separate SDKs. Verified with an offline pytest suite (monkeypatched
HTTP layer) instead of live keys, since no real provider keys are
configured in this environment - correctness of request/response
shape and error classification was verified; correctness against the
*actual* live APIs was not, and should be spot-checked once real keys
are available.

## Next process

Phase 1 is done and its exit criterion is verified via a real
`pip install -e .`, not `PYTHONPATH`/pytest config tricks alone. Start
**Phase 2 — OpenRouter free-model engine** in `plan.md`: fetch
`/models`, filter zero-pricing free models, build the per-model health
table, router chain (pinned -> free pool -> user paid key -> explicit
fail), capability probe for structured JSON output, and the circuit
breaker. Phase 2 and Phase 3 (evidence acquisition) are independent and
parallelizable per `plan.md`'s critical path.

When real provider keys become available, spot-check each of the four
1.2 providers against its live API at least once - the current test
suite only proves the code handles the *shapes* it was told to expect.
