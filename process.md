# Process Log

Tracks what has been done, what is pending, and the next step. Update this
file at the end of every work session — do not let it drift from reality.

## Status: Phase 0 and Phase 1 complete. Phase 2 (OpenRouter free-model
engine) complete (all 5 tasks).

## What is done

- **Phase 2 — OpenRouter free-model engine**:
  - `src/halludetect/llm/openrouter.py` — `OpenRouterProvider` (2.1's
    completion half): a thin `httpx.post` wrapper against
    `openrouter.ai/api/v1/chat/completions`, same shape as the other four
    providers. `fetch_free_models(api_key)` hits `/models` and keeps only
    ids where `pricing.prompt == 0` and `pricing.completion == 0` (cast to
    float since OpenRouter returns pricing as strings). `FreeModelCatalog`
    wraps that fetch with a cache keyed on `time.monotonic()` and a
    24-hour TTL (`get_models(force_refresh=False)`), so the router doesn't
    refetch the model list on every request.
  - `src/halludetect/llm/health.py` — `HealthTracker`/`ModelHealth` (2.2):
    per-model `attempts`, `successes`, `success_rate`, `avg_latency_ms`,
    `consecutive_failures`, `last_rate_limited_at`. `rank_available()`
    sorts by success rate then latency, and gives untried models a fair
    shot rather than ranking them last by default. Circuit breaker (2.5)
    lives in the same class: `record_failure()` puts a model into a
    time-boxed cooldown (`cooldown_until`) after `FAILURE_THRESHOLD`
    (default 3) consecutive failures; `record_success()` resets the
    consecutive-failure counter; `is_available()` checks the cooldown
    against a clock, so a model self-recovers once the cooldown window
    passes without any manual reset.
  - `src/halludetect/llm/router.py` — `Router.complete()` (2.3): tries
    `pinned_model` first if set, then free-pool models in
    `HealthTracker.rank_available()` order (capped at
    `max_free_models_tried`, default 5), then `user_provider` (a caller's
    own paid-key provider) if given. Every attempt records success/failure
    into the shared `HealthTracker`. If nothing succeeds, raises
    `LLMResponseError` listing what was tried and why each failed - it
    never returns a guess or silently picks a different result shape.
    `provider_factory: Callable[[str], LLMProvider]` is injected rather
    than the router constructing `OpenRouterProvider` itself, so tests
    (and later, real callers) can swap in anything satisfying the
    `LLMProvider` protocol per model id.
  - `src/halludetect/llm/structured.py` — `complete_structured()` (2.4):
    the actual capability probe. Since none of the providers built in 1.2
    or 2.1 support a native `response_format`/schema mode at the HTTP
    layer, this always goes through prompt-JSON: it asks for JSON matching
    a Pydantic schema's `model_json_schema()`, strips markdown fences if
    present, validates strictly, and on failure retries with the bad
    output + validation error appended to the prompt, up to
    `max_repairs` (default 2) times. Returns `(instance,
    honored_on_first_try)` - the second value is the actual probe signal
    future model-ranking logic can use, since `supports_json_schema()` on
    a provider is still just a static pre-probe default. Raises
    `LLMSchemaValidationError` (new, `LLMResponseError` subclass in
    `exceptions.py`) after exhausting repairs - never returns a
    best-effort partial parse.
  - `src/halludetect/llm/fake.py` — `FakeProvider`/`fake_response()`: a
    deterministic `LLMProvider` double that replays a scripted sequence of
    responses/errors, one per `complete()` call. Used instead of
    monkeypatching `httpx` for router/structured-output tests, since those
    tests are about chain and retry logic, not HTTP parsing (that's
    already covered per-provider in `test_llm_providers.py`).
  - **Verification**: 25 new offline tests, all passing, no network, no
    live keys: `tests/test_openrouter_catalog.py` (pricing filter, cache
    TTL/refresh/force-refresh, auth-error propagation), `tests/test_health.py`
    (success rate/latency math, cooldown trigger/expiry, reset-on-success,
    ranking order, cooldown exclusion), `tests/test_router.py` (pinned
    success skips free pool; pinned failure falls over to free pool; free
    pool exhausted falls over to user key; total exhaustion raises instead
    of guessing; health gets updated either way; `max_free_models_tried`
    cap is respected), `tests/test_structured.py` (valid JSON first try,
    valid JSON inside a markdown fence, repair-retry recovering on the
    second attempt, schema-mismatch triggering a repair, exhausted repairs
    raising `LLMSchemaValidationError`). Full suite is now 45/45.
  - Added `LLMSchemaValidationError` to `src/halludetect/llm/exceptions.py`.

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

Phases 1 and 2 are complete. Phases 3-8 in `plan.md` are pending. Note
that Phase 2's router, health tracker, and structured-output probe are
still not wired into anything outside their own tests - there is no
evidence source, no detection pipeline, no API. `Router` is usable as a
class but nothing constructs one against a real `.env`/settings.py yet;
that wiring, and the actual OpenRouter model list at runtime, happens
once Phase 4 (detection pipeline) needs to call an LLM for real. The
current root-level code (`app.py`, `config.py`, `detection/`, `rag/`,
`knowledge_base/`) is the **legacy v1 app** described in `plan.md`'s
Context section; it is not yet superseded and still runs, but it is not
where new work should go.

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

Phase 2: built the free-model catalog (2.1) and health tracker with its
circuit breaker (2.2/2.5) as independent, separately-tested units before
writing the router (2.3), since the router's only job is to combine
"which free models exist" with "which ones are currently healthy" plus
the pinned/user-key fallbacks - it shouldn't know how either of those
lists is produced. `Router` takes a `provider_factory` callable instead
of importing `OpenRouterProvider` directly, which is what let the whole
chain (including multi-model fail-over and the "everything fails" path)
get tested with `FakeProvider` instead of mocking `httpx` four different
ways. The capability probe (2.4) was scoped to what actually exists: none
of the six provider classes built so far (four in 1.2, `OpenRouterProvider`
in 2.1) send a native `response_format`/schema parameter, so "does the
model honor response_schema" is answered by prompting for JSON and
checking whether a repair round was needed - not by a native-mode branch
that doesn't have an HTTP implementation behind it yet. Verified per-unit
first, then the router's exit criterion (fail one model mid-run, confirm
fail-over, confirm no crash) with `FakeProvider` scripted to raise on
specific models - equivalent to killing a real model's key/endpoint,
without needing live OpenRouter access to prove it.

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

Phase 1 and Phase 2 are done. Start **Phase 3 — Evidence acquisition**
in `plan.md`: `EvidenceSource` Protocol + `DirectEvidence` (caller-
supplied, chunked), `WebSearchEvidence` (Tavily, per the Phase 0.2
decision - active only if `tavily_api_key` is configured), the explicit
`NOT_VERIFIABLE / no_evidence_configured` path with no silent
LLM-knowledge fallback, and a plug-in hook for a caller's own
retriever. Phase 3 has no dependency on Phase 2 (`plan.md`'s critical
path marks them parallelizable) so this can proceed independently of
anything the router does.

Two things still open from earlier phases, not yet acted on:
- When real provider keys become available, spot-check each of the four
  1.2 providers, `OpenRouterProvider`, and the free-model catalog fetch
  against their live APIs at least once - the test suites only prove the
  code handles the *shapes* it was told to expect, not real responses.
- Phase 2's `Router`/`HealthTracker`/`complete_structured` are built and
  tested in isolation but not yet wired to `settings.py` or called from
  anywhere real - that wiring happens naturally once Phase 4 needs to
  issue actual verification calls.
