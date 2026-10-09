# Changelog

## Unreleased

### Removed

- The unused NLI scorer stub (`detect/nli.py`, plan 4.5). Tested on the
  golden sets, the NLI model either added false flags or accepted false
  claims, so it won't be wired in.

### Fixed

- A model call now times out after 60s in total. OpenRouter keeps a slow
  call open with keep-alive whitespace, so the old 30s per-read timeout
  never fired and a request could hang for minutes and end in a 502.

## 1.1.0 - 2026-10-08

### API (contract v1.3)

- `POST /v1/chat`: a model answers a question (with optional chat
  history and sources), then a different model checks the answer exactly
  like `/v1/verify`. Without sources or a search key the answer is
  returned unchecked (`NOT_VERIFIABLE`).
- `GET /v1/models/stats`: per-model calls, failures, latency, and the
  verdicts each model's answers received and gave as checker.
- Web page and Streamlit UI: new Ask and Models modes alongside Check an
  answer.

### Verification

- Claims left `NOT_ENOUGH_INFO` get one quote-first recheck on the same
  model. A label only changes to `SUPPORTED` or `CONTRADICTED` with a
  verified quote; a failed recheck keeps the first result.

### Models (contract v1.2)

- NVIDIA provider, and an NVIDIA tier in the default free pool after
  OpenRouter's free models.
- A model that fails mid-request is replaced by the next one in the pool.
  Free-pool timeouts move on instead of retrying, and a request stops
  starting new attempts after `FREE_POOL_BUDGET_S` (90s).
- `allow_free_pool: false` is enforced (400 unless a model is pinned or
  another provider is chosen).
- Model health is only updated for models that were actually called.
- The OpenRouter free-model list is cached per process instead of fetched
  on every request.
- A used-up daily quota (429 with a far reset) is no longer retried; the
  models it covers are skipped until the reset. Models that answer 402 or
  403 are left out for 6 hours. Neither uses up a request's attempts.

## 1.0.0 - 2026-09-29

First release of HALLUDETECT v2, a standalone service that checks whether
an answer is supported by evidence. It replaces the v1 Streamlit demo,
kept in `legacy/` for reference.

### API

- `POST /v1/verify` and `GET /healthz`, per `docs/contract.md`
  (contract v1.1) and `docs/openapi.yaml`.
- Bearer-key auth and per-key token-bucket rate limiting (401 / 429).
- Per-request `cost_usd` from provider-reported token usage, estimated
  from a rate table when the provider doesn't report it.
- A web page at `/` and a Streamlit client (`streamlit run app.py`),
  both thin clients of the API.

### Verification

- Claims are extracted and typed; only `FACTUAL` claims are verified,
  up to 12 per answer.
- Verdicts are joined to claims by `claim_id`, never by position.
- A `SUPPORTED` label needs a quote found in the cited evidence chunk;
  otherwise it is downgraded to `NOT_ENOUGH_INFO`.
- No evidence, or fewer than 3 verifiable claims, gives
  `NOT_VERIFIABLE` with a `reason`. The model's own knowledge is never
  used as evidence.
- `p_hallucinated` is fitted per verdict on the two golden sets
  (`calibration_version: verdict-rate-v1`). Held-out ECE 0.063 / 0.082,
  fitting on one set and scoring the other.

### Evidence

- Caller-supplied evidence, chunked when long.
- Web search via Tavily, only when `TAVILY_API_KEY` is set.
- A plug-in hook for a caller's own retriever.

### Models

- OpenRouter free-model pool: catalog refreshed daily, per-model health
  tracking, circuit breaker, fail-over across models.
- Bring-your-own key for Gemini, OpenAI, Anthropic, or any
  OpenAI-compatible endpoint.
- JSON-schema capability probe with repair retries for models that
  don't honour structured output.
- Jittered backoff on timeouts and 429s. A rejected key (401) stops
  fail-over across free models; a model the key can't use (403) fails
  over to the next one.

### Operations

- Result cache on disk (`diskcache`); `NOT_ENOUGH_INFO` results are not
  cached, and the service runs uncached if the cache directory is
  read-only.
- Dockerfile and docker-compose; Vercel config for the API and web page.
- CI: ruff, mypy, pytest, and offline eval gates over golden set A
  (90 hand-written items) and golden set B (60 items from HaluEval QA).

### Known limitations

- The optional NLI cross-encoder signal (plan 4.5) is not wired in.
- Gemini, OpenAI and Anthropic providers are tested only against mocked
  responses, not live APIs.
- Rate limiter, health table and cache are per process; a multi-instance
  deployment would need a shared store.
- HTTP 5xx from a provider is not retried.
