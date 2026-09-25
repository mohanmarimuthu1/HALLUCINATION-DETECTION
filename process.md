# Process Log

Tracks what has been done, what is pending, and the next step. Update this
file at the end of every work session — do not let it drift from reality.

## Status: Phase 0, Phase 1, Phase 2, Phase 3 complete. Phase 4 — 4.1-4.4 complete, 4.5 partial (protocol/hook only, not wired). Phase 5 — 5.1-5.2 complete, 5.3-5.4 pending.

## What is done

- **Phase 5.2 — Per-key auth + token-bucket rate limiting**:
  - `src/halludetect/settings.py` — added `client_api_keys` (comma-separated
    list of valid caller keys, `SecretStr | None`, deliberately distinct
    from the provider keys already on this class - those authenticate
    *this service* to an LLM/search backend, `client_api_keys`
    authenticates a *caller* to this service) and
    `rate_limit_capacity`/`rate_limit_refill_per_s` (floats, defaulted so
    an empty env var falls back to the default rather than failing
    validation - verified directly, since a non-Optional numeric field
    behaves differently than the `SecretStr | None` fields elsewhere on
    this class when its env var is present but empty).
  - `src/halludetect/api/auth.py` — `require_api_key()`, a FastAPI
    dependency: validates `Authorization: Bearer <key>` against
    `settings.client_api_keys`. No keys configured at all is a 401 for
    every request (never a silent allow-through); a missing header, a
    header without the `Bearer ` prefix, or an unrecognized key are all
    401 too - never distinguished in the response, so a caller can't probe
    which failure mode occurred.
  - `src/halludetect/api/ratelimit.py` — `RateLimiter` (plain token
    bucket: capacity, continuous refill, per-key state in a dict) and
    `enforce_rate_limit()`, a FastAPI dependency chained after
    `require_api_key` so an invalid key is always 401, never 429 - rate
    limiting only ever applies to a key already known to be valid. One
    `RateLimiter` instance is shared per process (module-level, same
    pattern as `api.resolve`'s shared `HealthTracker`), not created per
    request, since per-key bucket state must persist across calls.
    In-memory/per-process only - documented as a v1 scope limit, same as
    `HealthTracker`; a multi-instance deployment needs a shared store.
  - `src/halludetect/api/main.py` — wired `enforce_rate_limit` onto
    `POST /v1/verify` only (`GET /healthz` stays unauthenticated, since
    it's a liveness probe, not a billable/rate-limited operation).
  - `.env.example` — documented `CLIENT_API_KEYS`,
    `RATE_LIMIT_CAPACITY`, `RATE_LIMIT_REFILL_PER_S`.
  - **Verification**: `tests/test_auth.py` (6 tests, calls the dependency
    function directly - no client needed): valid key(s) accepted,
    whitespace around keys in the list is trimmed, no keys configured
    rejects everything, missing header / missing `Bearer` prefix /
    unrecognized key are all 401. `tests/test_ratelimit.py` (4 tests, a
    `now` clock passed explicitly rather than relying on real elapsed
    time): allows up to capacity then blocks, refills over time, refill
    never exceeds capacity, different keys tracked independently. 10 new
    end-to-end cases added to `tests/test_api.py`: every existing Phase
    5.1 test updated to carry a valid `Authorization` header (they'd
    otherwise now 401), plus new cases for no header, wrong key, malformed
    header (no `Bearer` prefix), no keys configured, over-rate-limit
    (429 on the second call with `capacity=1`), and that two different
    keys each get their own independent bucket. One non-obvious fix
    needed here: `auth.py`/`ratelimit.py` bind `settings: Settings =
    Depends(get_settings)` as a default-parameter object captured by
    FastAPI at import time - monkeypatching the module-level
    `get_settings` attribute (which works fine for `main.py`'s own direct
    `get_settings()` call inside the route body) does **not** reach that
    already-captured reference. Fixed by using FastAPI's own
    `app.dependency_overrides[get_settings] = ...` mechanism for the two
    dependency-injected call sites, alongside the existing
    `monkeypatch.setattr(main, "get_settings", ...)` for the direct-call
    site - both are needed since they're two different resolution
    mechanisms. Full suite: 141/141 passing, offline, no network, no live
    keys.

- **Phase 5.1 — FastAPI `/v1/verify` + `/healthz`**:
  - `src/halludetect/api/schemas.py` — `VerifyRequestIn`/`ModelPrefsIn`,
    the API-facing request shapes validated against incoming JSON,
    mirroring `docs/contract.md` field-for-field. Deliberately separate
    from `detect/schemas.py` (the pipeline's internal shapes) - this is
    the boundary where untrusted request bodies get typed.
  - `src/halludetect/api/resolve.py` — the wiring Phase 4 explicitly left
    open: `resolve_evidence_source()` implements the `evidence` vs
    `evidence_source` precedence from `docs/contract.md` (caller-supplied
    `evidence` always wins; `custom` needs a plugin registered on the
    deployment via `app.state.custom_evidence_source`, since a Python
    callable can't be expressed in a JSON body, so an unregistered
    `custom` request is a 400, never a silent `NoEvidenceSource`
    fallback; `web`/`none` construct `WebSearchEvidence`/`NoEvidenceSource`
    directly - `WebSearchEvidence`'s own missing-key check already
    degrades to the contract's required `none`-equivalent behavior, so no
    extra logic was needed here). `resolve_provider()` turns `model_prefs`
    into one resolved `LLMProvider` + `ModelUsed`: `provider: openrouter`
    (default) goes through `Router.pick_model()` (new method, see below);
    `gemini`/`openai`/`anthropic`/`custom` are built directly from
    `model_prefs.user_api_key` (falling back to this deployment's own key)
    and `model_prefs.pinned_model` (falling back to that provider's
    default model, or a `ResolutionError` for `custom` which has no sane
    default). A module-level `HealthTracker` singleton is shared across
    requests (not created per-request) so the circuit breaker and
    success-rate ranking actually accumulate signal over the process
    lifetime.
  - `src/halludetect/llm/router.py` — added `Router.pick_model()`:
    resolves which model the router would try first (pinned, else
    top-ranked available free model) **without making any request**.
    Added because `detect.pipeline.run()` (Phase 4) takes one
    already-resolved provider for the whole request, not a `Router` - so
    Phase 5 has to decide which model that is once, up front, rather than
    getting per-call failover the way `Router.complete()` provides.
    `Router.complete()` itself is untouched; `pick_model()` doesn't
    consult `user_provider` since a non-openrouter `model_prefs.provider`
    is resolved directly by `api.resolve`, never routed through the free
    pool.
  - `src/halludetect/api/main.py` — the FastAPI app: `GET /healthz`
    returns `{"status": "ok"}`; `POST /v1/verify` parses the request,
    calls `resolve_evidence_source`/`resolve_provider`, then
    `pipeline.run()`, and serializes `AnalysisResult` straight back
    (FastAPI's `response_model` handles that). `ResolutionError` (bad
    request config, e.g. unregistered `custom` evidence source) maps to
    400; `LLMError` from provider resolution (e.g. no pinned model and no
    free models available) maps to 503; `LLMError` raised *during*
    `pipeline.run()` (a resolved model actually failing mid-request) maps
    to 502 and is logged as a warning - this is the first real caller of
    both `resolve_evidence_source`/`resolve_provider` and
    `detect.pipeline.run()` together. `app.state.custom_evidence_source`
    is the registration point for an embedding deployment's own
    retriever; unset by default. Auth (5.2) and rate limiting (5.2) are
    not implemented yet - there is no `curl` protection on this endpoint
    today.
  - Added `fastapi`/`uvicorn` to `pyproject.toml` dependencies; installed
    into `.venv` (`fastapi` was missing there - `uvicorn` happened to
    already be present as some other package's dependency).
  - **Verification**: `tests/test_resolve.py` (13 tests) and
    `tests/test_api.py` (6 tests), all offline (`OpenRouterProvider.complete`
    and `fetch_free_models` monkeypatched, no live keys, no network):
    evidence resolution precedence (direct evidence wins even when
    `evidence_source` is also set; unregistered `custom` raises; a
    registered `custom` source is used directly; `web`/`none` resolve to
    the right concrete class); model resolution (pinned model skips the
    catalog fetch entirely - asserted by making the fetch raise if
    called; top-ranked free model is picked; no pinned model + empty free
    pool raises; named providers use `user_api_key`/`pinned_model` or fall
    back to the deployment key/default model; `custom` provider requires
    both `custom_provider_base_url` and `pinned_model`); end-to-end API
    tests (`/healthz`; a no-evidence request never reaches the LLM at all,
    same defense-in-depth property Phase 4's own pipeline test asserts,
    now proven through the HTTP layer; a direct-evidence request runs the
    full extraction -> verification -> quote-grounding chain through a
    scripted `OpenRouterProvider`; `custom` evidence source without
    registration and `custom` model provider without a base URL both
    return 400; no free models and no pinned model returns 503). Full
    suite: 125/125 passing.

- **Phase 4 — Detection core (4.1-4.4 complete, 4.5 partial)**:
  - `src/halludetect/detect/schemas.py` — pydantic models for the whole
    pipeline: `ClaimType`, `Label` (the smaller per-claim enum -
    `docs/contract.md` has no per-claim `NOT_VERIFIABLE`), `Verdict`
    (the top-level enum, which does), `ExtractedClaim`/`ExtractedClaims`
    (extraction LLM call shape), `Claim` (extracted claim + locally
    assigned `claim_id`), `RawClaimVerdict`/`RawVerdict` (verification LLM
    call shape), `ClaimResult`/`ModelUsed`/`Timings`/`AnalysisResult`
    (mirrors `docs/contract.md`'s response schema field-for-field),
    `Signals` (intermediate per-label counts for `fuse.py`). Pydantic
    `BaseModel`, not the plain dataclasses `llm/base.py`/`evidence/base.py`
    use, because `ExtractedClaims`/`RawVerdict` are passed straight to
    `complete_structured()` which needs `model_json_schema()`.
  - `src/halludetect/detect/claims.py` — `extract_claims()` (4.1): prompts
    for FACTUAL/OPINION/INSTRUCTION/META-tagged claims via
    `complete_structured`, then assigns `claim_id` itself
    (`claim-0`, `claim-1`, ...) rather than trusting the model to produce
    one - v1's `claim_extractor.py` had no id concept at all, which is
    exactly what let its verifier fall back to joining by position later.
    `max_claims` (default 12) is the only cap and is always honored
    exactly - v1 had a second, undocumented hardcoded cap
    (`claims[:2]` in `_parse_claims`) on top of whatever the caller
    configured.
  - `src/halludetect/detect/verify.py` — `verify_claims()` (4.2): sends
    only FACTUAL claims + evidence chunks to `complete_structured` against
    `RawVerdict`, then joins each returned `RawClaimVerdict` back to its
    `Claim` **by `claim_id`**, explicitly - `dict` lookup, not position.
    A claim the model didn't return a verdict for gets an explicit
    `NOT_ENOUGH_INFO`/confidence-0.0 default; a verdict for a claim_id
    never asked about is dropped. v1's `fact_verifier.py`
    (`_parse_batch_verification`) looked for a `CLAIM_N` marker per line
    and fell back to `lines[i]` by raw index when the marker was missing -
    a dropped or reordered line silently verified the wrong claim.
  - `src/halludetect/detect/quote_check.py` — `quote_is_grounded()` (4.3):
    exact substring match after whitespace normalization only - no fuzzy/
    similarity fallback. A `rapidfuzz.partial_ratio` prototype was tried
    and rejected: it scored a quote with an altered number ("...built in
    1999" against evidence reading "...built in 1932.") at 96/100,
    above any threshold that would still only catch real paraphrasing.
    Accepting a factually altered quote as "grounded" is worse than no
    fuzzy matching, so the dependency was removed rather than shipped
    with an unsafe threshold. v1's `fact_verifier.py` never checked a
    quote against evidence at all - a `SUPPORTED` verdict was accepted on
    the model's freeform "explanation" string alone.
  - `src/halludetect/detect/fuse.py` — `wilson_ci()`, `summarize()`,
    `fuse()` (4.4): Wilson 95% CI over per-claim label counts;
    `n_verifiable_claims < 3` forces `Verdict.NOT_VERIFIABLE` regardless of
    label mix (`docs/contract.md` rule 1); any `CONTRADICTED` claim forces
    `Verdict.CONTRADICTED` (outranks `NOT_ENOUGH_INFO`); all-supported is
    `GROUNDED`. `p_hallucinated`/`groundedness` come from a documented,
    named heuristic (`calibration_version = "heuristic-v0"`), not a fitted
    model - there's no labeled calibration set until Phase 7's golden
    sets exist to fit one against.
  - `src/halludetect/detect/pipeline.py` — `run()`: orchestrates
    evidence-fetch → `extract_claims` → filter to FACTUAL → `verify_claims`
    → `quote_is_grounded` downgrade → `fuse`. Enforces the no-evidence
    rule itself (defense in depth, not just documented): an empty
    `evidence_source.fetch()` result skips extraction/verification
    entirely and returns `NOT_VERIFIABLE` without ever calling the LLM -
    this is the first real caller of both Phase 2's
    `LLMProvider`/`complete_structured` and Phase 3's `EvidenceSource`
    implementations together. Takes one already-resolved `LLMProvider` for
    the whole request (not a `Router`) so `model_used` in the response is
    attributable to one actual model even though extraction and
    verification are two separate LLM calls - resolving *which* model via
    the router is left to Phase 5's API layer.
  - `src/halludetect/detect/nli.py` — **4.5, partial**: `NLIScorer`
    Protocol + `CrossEncoderNLIScorer` lazy-loader skeleton only.
    `sentence_transformers`/`torch` import happens inside
    `CrossEncoderNLIScorer.__init__`, never at module import, so nothing
    needs that (heavyweight) dependency installed unless a real model is
    configured. **Not done**: no `settings.nli_model` field, not wired
    into `pipeline.py`, and no concrete `label_order` verified against a
    real checkpoint - NLI cross-encoder output label ordering isn't
    standardized across checkpoints, so guessing one here would risk
    silently inverting entailment/contradiction, which is exactly the
    class of unverified assumption this project exists to avoid. Left
    unchecked in `plan.md` rather than marked done.
  - **Verification**: `tests/test_claims.py`, `tests/test_verify.py`,
    `tests/test_quote_check.py`, `tests/test_fuse.py`,
    `tests/test_pipeline.py` (33 new tests) plus **`tests/test_parsers.py`**
    (Phase 4's exit criteria file, 6 tests) - every v1 parser bug listed
    above reproduced as a failing-if-regressed case, each run against two
    distinct `FakeProvider` instances (different `provider_name`/`model`)
    to confirm the fix isn't tied to one model's output quirks. Full
    suite: 106/106 passing, offline, no network, no live keys.

- **Phase 3 — Evidence acquisition (complete)**:
  - `src/halludetect/evidence/custom.py` — `CustomEvidenceSource` (3.4):
    the plug-in hook for `evidence_source: custom`. Any object already
    exposing `fetch(query) -> list[Evidence]` satisfies `EvidenceSource`
    directly and needs no adapter; this wraps the more common case of a
    caller's retriever returning plain `list[str]` chunks for a query,
    chunking oversized strings the same way `DirectEvidence` does
    (chunk ids `custom-{i}` / `custom-{i}-{j}`, `source="custom"`).
    Unlike `WebSearchEvidence`, a retriever exception is *not* caught and
    degraded to no evidence - it is the caller's own plugin, so a bug in
    it should surface to them, not be hidden behind a silent empty
    result.
  - `src/halludetect/evidence/chunk.py` — extracted `chunk_text()` out of
    `direct.py` (no behavior change) so `DirectEvidence` and
    `CustomEvidenceSource` share one chunking implementation instead of
    each maintaining their own copy of the same paragraph/sentence
    splitting logic.
  - **Verification**: `tests/test_custom_evidence.py` (7 tests): Protocol
    conformance, the query is actually passed to the retriever, short
    results become one chunk each, blank results are skipped, long
    results split into bounded chunks under the shared chunker, a
    retriever exception propagates rather than being swallowed, an empty
    retriever result returns no evidence. Full suite re-run after the
    `chunk.py` extraction confirmed no regression in `DirectEvidence`'s
    existing tests. Full suite: 73/73 passing, offline.
  - `src/halludetect/evidence/web_search.py` — `WebSearchEvidence` (3.2):
    Tavily-backed, per the Phase 0.2 decision in `docs/contract.md`. A
    missing `api_key` returns `[]` before any network call, matching the
    contract's rule that a `web` request with no key configured behaves
    identically to `NoEvidenceSource`. A live search failure (non-200
    response, transport error, unparseable JSON) also degrades to `[]`
    rather than raising - logged as a warning via `halludetect.logging`
    so the failure is visible without turning into a request-level
    crash. Result count is capped at `max_results`; blank-content
    results are skipped the same way `DirectEvidence` skips blank
    strings.
  - **Verification**: `tests/test_web_search_evidence.py` (11 tests,
    `httpx.post` monkeypatched, no live Tavily key or network):
    Protocol conformance, no-key path never calls `httpx.post` at all
    (asserted via a monkeypatch that raises if called), successful
    parse into `Evidence` chunks, result count capped at
    `max_results`, blank-content results skipped, HTTP error status
    degrades to no evidence, transport/timeout error degrades to no
    evidence, invalid JSON body degrades to no evidence, and the
    request body actually carries the configured `api_key`/`query`/
    `max_results`/`timeout`. Full suite: 66/66 passing, offline.
  - `src/halludetect/evidence/base.py` — `Evidence` frozen dataclass
    (`chunk_id`, `text`, `source`) and the `EvidenceSource` Protocol
    (`@runtime_checkable`): `fetch(query) -> list[Evidence]`. An empty
    list is the explicit "no evidence" signal every evidence source
    (direct, web, none, a caller's plugin) must be able to return — it is
    what Phase 4's pipeline will use to force `NOT_VERIFIABLE` instead of
    guessing.
  - `src/halludetect/evidence/direct.py` — `DirectEvidence` (3.1): wraps
    caller-supplied `request.evidence` strings (`docs/contract.md`:
    "if evidence is present, verify against THIS ONLY"). `query` is
    accepted only to satisfy the Protocol and is otherwise ignored.
    Chunks each input string only if it exceeds `max_chunk_chars`
    (default 1000): splits on paragraph boundaries first, then sentence
    boundaries within an oversized paragraph, greedily repacking under
    the limit. A single sentence longer than the limit is kept whole
    rather than truncated — losing evidence text silently would be worse
    than one oversized chunk. Chunk ids are `direct-{i}` for an input
    that didn't need splitting, `direct-{i}-{j}` for the pieces of one
    that did, so multiple inputs never collide.
  - `src/halludetect/evidence/none.py` — `NoEvidenceSource` (3.3): always
    returns `[]`, deliberately, not as a stub. This is the concrete
    implementation of `evidence_source: none`, and per `docs/contract.md`
    is also what a `web` request must fall back to identically once the
    Tavily key check (3.2) is wired in — the module docstring records
    that dependency for when 3.2 is built. Having this as a real,
    tested, always-empty source now means the "no evidence → no guess"
    rule already has a concrete object to point at, rather than being an
    implicit gap that 3.2 or Phase 4 could quietly special-case around.
  - **Verification**: `tests/test_evidence.py` (9 tests: Protocol
    conformance, short strings each become their own chunk with the
    unsplit id scheme, `query` is ignored, blank strings are skipped
    entirely — never emitted as empty `Evidence`, long input splits on
    paragraph-then-sentence boundaries with every original sentence
    still present in the rejoined output, an oversized single sentence
    is kept whole rather than dropped/truncated, multiple inputs chunk
    independently without id collisions) and `tests/test_no_evidence.py`
    (2 tests: Protocol conformance, always-empty regardless of query).
    Full suite: 54/54 passing, offline, no network, no live keys.

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

Phases 1, 2, and 3 are complete. Phase 4 is done except 4.5 (optional NLI
signal, only a protocol/lazy-loader skeleton exists - see above). Phase 5
is done through 5.2 - 5.3 (cost/usage tracking) and 5.4 (optional thin demo
UI) are pending. `POST /v1/verify` now requires a valid `Authorization:
Bearer <key>` (401 without one, or if no keys are configured at all) and
is rate-limited per key (429 over the configured token-bucket capacity),
but there is still no per-key *usage* tracking - the rate limiter knows a
key made a request, not how many tokens or how much it cost. `cost_usd` in
the response is currently always whatever `pipeline.run()`'s default
(`0.0`) is - Phase 5.3 needs to compute a real per-request cost from token
usage, which `LLMResponse.usage` already carries but nothing reads yet.
Phases 6-8 in `plan.md` are pending. The current root-level code
(`app.py`, `config.py`, `detection/`, `rag/`, `knowledge_base/`) is the
**legacy v1 app** described in `plan.md`'s Context section; it is not yet
superseded and still runs, but it is not where new work should go.

Known outstanding issue not yet fixed in the legacy app: the hardcoded API
keys committed in prior git history are still exposed in git log/GitHub even
though `config.py` no longer contains them on disk. They should be treated
as compromised.

## How this was done

Phase 5.2: kept auth and rate limiting as two separate, independently
testable FastAPI dependencies (`auth.require_api_key`,
`ratelimit.enforce_rate_limit`) rather than one combined function, and
chained rate limiting *after* auth (`enforce_rate_limit` takes
`api_key: str = Depends(require_api_key)`) specifically so an invalid key
can never consume rate-limit budget or return 429 instead of 401 - the
plan's own exit criterion lists 401 and 429 as distinct, checkable
outcomes. Used a plain in-process token bucket instead of reaching for
Redis/an external store, matching the same scope decision already made for
`HealthTracker` in Phase 2 - both are documented as single-instance-only,
with the same "Phase 6 is where a shared store would first be justified"
note. Hit and fixed a real FastAPI gotcha while wiring tests: a
`Depends(get_settings)` default parameter captures the function object at
import time, so `monkeypatch.setattr(module, "get_settings", ...)` (which
worked fine for `main.py`'s own direct `get_settings()` call) silently did
nothing for `auth.py`/`ratelimit.py`'s dependency-injected settings -
diagnosed by the tests failing with the *old* settings' values still in
effect, then fixed with `app.dependency_overrides[get_settings]`, FastAPI's
actual mechanism for this, rather than monkeypatching harder.

Phase 5.1: read `pipeline.py`'s and `router.py`'s module docstrings first,
since both already recorded exactly what Phase 5 was expected to do
(resolve a single provider up front, not hand the pipeline a `Router`) -
implemented that literally rather than re-deriving a design: added
`Router.pick_model()` as a narrow, request-free extension of the existing
router (no change to `Router.complete()`'s behavior or tests) and a new
`api.resolve` module that owns the `model_prefs`/`evidence_source` ->
concrete-object mapping, so `api/main.py` itself stays a thin
parse -> resolve -> pipeline.run() -> serialize sequence. Deliberately
did *not* try to make provider resolution lazy (skip it when evidence
turns out empty) even though that would save a catalog fetch on the
no-evidence path - `process.md`'s own prior note said Phase 5 should
resolve the model "once, up front" regardless, and changing Phase 4's
`pipeline.run()` signature to support lazy resolution wasn't worth
reopening already-tested code for a minor efficiency gain. Verified with
an offline `TestClient`-based suite (`OpenRouterProvider.complete` and
`fetch_free_models` monkeypatched, same pattern as `test_router.py`/
`test_openrouter_catalog.py`) rather than a live server, confirming both
the resolution logic in isolation (`test_resolve.py`) and the full
HTTP-request-to-`AnalysisResult` path (`test_api.py`), including that a
no-evidence request still never reaches the LLM even through the API
layer - the same property Phase 4's pipeline test already proved at the
pipeline level, now proved again at the boundary a real caller actually
hits.

Phase 4: built `detect/schemas.py` first (same protocol-first pattern as
Phases 1-3), then `claims.py`/`verify.py`/`quote_check.py`/`fuse.py` as
independently testable units before `pipeline.py` wired them together -
each of those four modules exists specifically to close one named v1 bug
(see the Phase 4 entry above and `tests/test_parsers.py`'s module
docstring), so each got its own regression test before being composed.
`quote_check.py`'s fuzzy-matching prototype was built, empirically tested
against an adversarial case (a quote with a materially wrong number), and
then deliberately reverted to exact-substring-only when that prototype
scored the wrong-number case as "grounded" - verified by running
`rapidfuzz.fuzz.partial_ratio` directly against the adversarial pair
before deciding, not by assumption. `pipeline.py`'s no-evidence short
circuit was written as a hard `if not evidence: return NOT_VERIFIABLE`
before any LLM call, specifically so the rule holds even if a future
caller forgets to check `evidence_source` - verified by asserting
`provider.call_count == 0` in `tests/test_pipeline.py` and
`tests/test_parsers.py`, not just by checking the returned verdict.
4.5 (NLI cross-encoder) was scoped down to a Protocol + lazy-loader
skeleton rather than a full concrete implementation, because a real
cross-encoder checkpoint's output label ordering isn't standardized and
guessing at one without a model available to verify against would be
exactly the kind of unverified assumption this project exists to avoid;
left unchecked in `plan.md` rather than marked done.

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

Phase 3 (3.1, 3.3): built `EvidenceSource` as a `Protocol` first, same
pattern as `LLMProvider` in Phase 1.2, so `DirectEvidence`,
`NoEvidenceSource`, and `WebSearchEvidence`/the still-pending custom
plug-in (3.2/3.4) are interchangeable to whatever calls `fetch()` later.
Did 3.1 and 3.3 together rather than 3.1 alone because both are
self-contained (no external API, no settings/HTTP dependency) and 3.3's
"no evidence" object is what 3.1's docstring and the contract's `web`
fallback rule both point at — building them apart would have left a
one-commit-later gap where "no evidence" was documented but had no
concrete implementation.

Phase 3 (3.2): built `WebSearchEvidence` against the same
`httpx.post` + monkeypatch-in-tests pattern used by every Phase 1.2/2.1
provider, rather than pulling in a Tavily SDK. The no-key check happens
before any network call (mirrors `NoEvidenceSource`, and matches
`docs/contract.md`'s rule that a missing key must be indistinguishable
from `evidence_source: none`). Live failures (bad status, transport
error, bad JSON) degrade to `[]` instead of raising, because Phase 3 has
no fail-over chain the way Phase 2's `Router` does - there is nothing to
fail over *to* for a single web-search call, so the only two honest
options were "raise and crash the request" or "return no evidence and
let the non-negotiable no-evidence rule apply." Chose the latter, but
logged each degrade path as a `structlog` warning so the failure isn't
silently invisible to operators - "never swallow the error" (the
convention `llm/exceptions.py` states for providers) means never hide it
from logs, not necessarily always propagate an exception when there's no
caller who could do anything with it yet.

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

Phase 5.1 and 5.2 are done. Continue **Phase 5 — API & interface** with
5.3 (cost/usage tracking per request), then 5.4 (optional thin demo UI,
zero business logic) if still in scope. 5.3 needs real cost computation:
`LLMResponse.usage` (`TokenUsage.prompt_tokens`/`completion_tokens`) is
already returned by every provider call but nothing currently reads it -
`pipeline.run()` would need to either return usage alongside
`AnalysisResult` or accept a cost-tracking callback, and per-token USD
rates need a lookup table per provider/model (free-pool models are `0.0`
by definition; named providers are not). `plan.md`'s verification
checklist for Phase 5 (`curl -X POST /v1/verify` succeeds with a valid
key, 401 without one, 429 over rate limit) can now actually be run against
a live process - it wasn't checkable before 5.2 existed.

To run the service live (not just the offline test suite), real values are
needed in `.env` that this session cannot supply on its own:
- `OPENROUTER_API_KEY` (or another provider's key) - without one,
  `resolve_provider()` for the default `openrouter` path will 503 on
  every request (no pinned model, no free models reachable).
- At least one value in `CLIENT_API_KEYS` - without one, every
  `/v1/verify` call 401s by design (see Phase 5.2 above). Any string(s)
  work; these are caller-facing keys this deployment invents and hands
  out, not something obtained from a vendor.
Ask the user for these before attempting to actually start
`uvicorn halludetect.api.main:app` against real traffic; the offline test
suite (141/141 passing) does not need either.

Two things still open from earlier phases, not yet acted on:
- When real provider keys become available, spot-check each of the four
  1.2 providers, `OpenRouterProvider`, and the free-model catalog fetch
  against their live APIs at least once - the test suites only prove the
  code handles the *shapes* it was told to expect, not real responses.
  This now also includes an end-to-end live `/v1/verify` call once a real
  `OPENROUTER_API_KEY` is available - the offline suite proves the wiring,
  not that a real free model actually answers usefully.

One thing open from this session:
- Phase 4.5 (NLI cross-encoder second signal) has only a `Protocol` +
  lazy-loader skeleton in `src/halludetect/detect/nli.py` - no
  `settings.nli_model` field, no pipeline wiring, no verified
  `label_order` for a real checkpoint. Picking this up means choosing an
  actual cross-encoder model, adding `sentence-transformers`/`torch` as a
  dependency (heavyweight - worth confirming with whoever owns deploy
  size/build time before adding), and verifying that model's real output
  label ordering before wiring `label_order`, not assuming one.
