# Changelog

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
