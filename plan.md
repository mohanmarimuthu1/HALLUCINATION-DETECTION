# Plan — HALLUDETECT v2: Standalone Hallucination Verification Service

> On approval this file gets copied to the repo root as `plan.md`. Work phase by phase, in order. Do not start a phase until the previous phase's Exit criteria pass. Update this file's checkboxes as tasks complete.

---

## Context

This supersedes the original `HALLUCINATION_DETECTION` repo, which was a self-RAG demo whose detector could not detect hallucinations (substring verdict matching, no quote grounding, self-learning KB poisoning — see prior `plan.md` in repo history if present).

**v2 is a different product, not a patch:** a standalone service that verifies whether **any given answer**, from **any source** (not just its own RAG output), is grounded — with evidence either supplied by the caller or fetched by the service. LLM backend must be pluggable: primary is OpenRouter's free-model pool (auto-rotating across free models), with support for user-supplied API keys (Gemini, OpenAI, Anthropic, or any OpenAI-compatible endpoint).

**Non-negotiable rule (the exact defect that killed v1):** the model must never fall back on its own general/world knowledge and label that "supported." No evidence available → verdict is `NOT_VERIFIABLE`. Every `SUPPORTED` verdict must carry a verbatim quote, programmatically verified against the cited evidence chunk.

**Target bar:** production service — typed, tested, CI-gated, containerized, rate-limited, cost-tracked, calibrated. Not a side project.

---

## API contract (freeze before writing code)

```
POST /v1/verify
{
  "answer": "string, required",
  "question": "string, optional — improves claim extraction",
  "evidence": ["string", "..."],          // optional; if present, verify against THIS ONLY
  "evidence_source": "none | web | custom",
  "model_prefs": {
    "provider": "openrouter | gemini | openai | anthropic | custom",
    "allow_free_pool": true,
    "pinned_model": null,
    "user_api_key": null                  // caller's own key, if provider != openrouter free pool
  }
}

→ 200 AnalysisResult
{
  "request_id": "...",
  "verdict": "GROUNDED | CONTRADICTED | NOT_ENOUGH_INFO | NOT_VERIFIABLE",
  "p_hallucinated": 0.0,
  "groundedness": 0.0,
  "groundedness_ci": [0.0, 0.0],           // Wilson 95%
  "claims": [
    {"claim_id": "...", "text": "...", "label": "SUPPORTED|CONTRADICTED|NOT_ENOUGH_INFO",
     "confidence": 0.0, "evidence_chunk_ids": ["..."], "quote": "...", "quote_verified": true}
  ],
  "n_verifiable_claims": 0,
  "model_used": {"provider": "...", "model": "..."},
  "cost_usd": 0.0,
  "timings_ms": {"total": 0, "retrieval": 0, "extraction": 0, "verification": 0},
  "calibration_version": "..."
}
```

Rules baked into the contract:
- `n_verifiable_claims < 3` → verdict forced to `NOT_VERIFIABLE`, never a bare percentage shown for n≤2.
- `evidence_source: none` with no web-search key configured → verdict `NOT_VERIFIABLE`, reason `no_evidence_configured`. Never silently answered from model knowledge.
- Every `SUPPORTED` claim requires `quote_verified: true`; if the quote check fails, the label is downgraded to `NOT_ENOUGH_INFO` server-side before the response is returned — never surfaced as `SUPPORTED` with an unverified quote.

---

## Target repo layout

```
pyproject.toml                # requires-python >=3.11,<3.13
src/halludetect/
  settings.py                 # pydantic-settings, SecretStr per provider key
  logging.py                  # structlog JSON, request_id contextvar
  llm/
    base.py                   # Protocol: complete(), supports_json_schema()
    openrouter.py             # free-model pool: fetch, filter pricing=0, rotate, circuit-break
    gemini.py
    openai.py
    anthropic.py
    custom_openai_compat.py
    router.py                 # chain: pinned -> free pool -> user paid key -> explicit fail
    fake.py                   # deterministic test double
  evidence/
    base.py                   # Protocol: EvidenceSource.fetch(query) -> list[Evidence]
    direct.py                 # caller-supplied evidence, chunked
    web_search.py             # Tavily/Serper — only active if key configured
    none.py                   # explicit NOT_VERIFIABLE path
  detect/
    schemas.py                # Label, RawVerdict, ClaimResult, AnalysisResult, Signals
    claims.py                 # typed extraction: FACTUAL/OPINION/INSTRUCTION/META, cap 12
    verify.py                 # structured-JSON verification, join by claim_id
    quote_check.py            # deterministic substring + rapidfuzz fallback
    fuse.py                   # calibrated scoring, Wilson CI
    pipeline.py                # orchestrates the above
  eval/
    datasets.py  runner.py  metrics.py
  api/
    main.py                   # FastAPI app: /v1/verify, /healthz
    auth.py                    # per-key auth
    ratelimit.py               # token bucket
  cache/
    store.py                  # sha256(answer, evidence, provider, settings) -> AnalysisResult
  ui/                          # optional thin client only, no logic here
tests/
  test_parsers.py              # every v1 parser bug as a regression case
  test_quote_check.py
  test_router.py
  data/golden/eval_set_a.yaml  # evidence-provided, 80-100 items
  data/golden/eval_set_b.yaml  # open-domain, sampled from FEVER/TruthfulQA/HaluEval
Dockerfile
docker-compose.yml
.github/workflows/ci.yml
.env.example
docs/contract.md              # the API contract above, versioned
```

---

## Phase 0 — Spec lock (1–2 days)

- [ ] 0.1 Freeze `AnalysisResult` schema + verdict enum in `docs/contract.md` (S)
- [ ] 0.2 Pick web-search provider for no-evidence mode — Tavily or Serper, whichever has usable free tier (S)
- [ ] 0.3 Write OpenAPI contract for `/v1/verify` before any implementation code (S)

**Exit:** schema + contract committed to `docs/contract.md`, not touched again without a version bump.

## Phase 1 — Core skeleton + provider abstraction (2–3 days)

- [x] 1.1 `src/` layout, `pyproject.toml`, `settings.py` with `SecretStr` per provider (M)
- [ ] 1.2 `LLMProvider` Protocol + `Gemini`/`OpenAI`/`Anthropic`/generic-OpenAI-compatible implementations (M)
- [ ] 1.3 structlog JSON logging with `request_id` contextvar (S)

**Exit:** swapping provider is a config change, not a code change.

## Phase 2 — OpenRouter free-model engine (2–3 days) — the differentiator

- [ ] 2.1 Fetch OpenRouter `/models`, filter `pricing.prompt == 0`, cache list, refresh daily (S)
- [ ] 2.2 Per-model health table: rolling success rate, latency, last-rate-limited-at (M)
- [ ] 2.3 Router chain: pinned model → next healthy free model → user's paid key (if given) → explicit fail (M)
- [ ] 2.4 Capability probe: does model honor `response_schema`? If not, prompt-JSON + repair-retry + strict Pydantic validate; reject after 2 failed repairs (M)
- [ ] 2.5 Circuit breaker: demote model after N consecutive failures, cooldown-based re-enable (S)

**Exit:** pipeline runs correctly against ≥5 different free OpenRouter models with zero code changes; degrades gracefully when one model is down.

## Phase 3 — Evidence acquisition (3–4 days)

- [ ] 3.1 `EvidenceSource` Protocol; `DirectEvidence` — caller-supplied, chunked if long (S)
- [ ] 3.2 `WebSearchEvidence` — active only if search API key configured (M)
- [ ] 3.3 Explicit `NOT_VERIFIABLE / no_evidence_configured` path — no silent LLM-knowledge fallback (S)
- [ ] 3.4 Plug-in hook so a caller's own retriever/RAG can act as an evidence source (S)

**Exit:** all three evidence modes work independently; absence of evidence is never papered over.

## Phase 4 — Detection core (4–5 days)

- [ ] 4.1 Typed claim extraction (`FACTUAL/OPINION/INSTRUCTION/META`), verify only `FACTUAL`, cap `max_claims=12` (M)
- [ ] 4.2 Structured verdict schema; join verdicts to claims by `claim_id`, never by position (M)
- [ ] 4.3 Quote-grounding check: `SUPPORTED` without a verbatim quote in a cited chunk → downgraded to `NOT_ENOUGH_INFO`, flagged `quote_unverified` (M)
- [ ] 4.4 Calibrated scoring: signal vector → fitted probability, Wilson CI, no hardcoded floor; `n_verifiable_claims < 3` → `NOT_VERIFIABLE` (M)
- [ ] 4.5 NLI cross-encoder as optional second signal, lazy-loaded behind `settings.nli_model` (L)

**Exit:** `tests/test_parsers.py` covers every v1 parser bug as a regression case, passing across ≥2 different providers.

## Phase 5 — API & interface (3–4 days)

- [ ] 5.1 FastAPI `/v1/verify`, `/healthz`, auto-generated OpenAPI docs (M)
- [ ] 5.2 Per-key auth + token-bucket rate limiting (S)
- [ ] 5.3 Cost/usage tracking per request: tokens, provider, USD (S)
- [ ] 5.4 Optional thin demo UI calling the API — zero business logic in the UI layer (M)

**Exit:** service fully usable via `curl`; UI is replaceable without touching the core.

## Phase 6 — Reliability & ops (3–4 days)

- [ ] 6.1 Result cache: `sha256(answer, evidence, provider, settings)` → `AnalysisResult`, diskcache/Redis (M)
- [ ] 6.2 Timeouts + jittered backoff on every provider call; classify failures by exception type, never string-match (S)
- [ ] 6.3 Dockerfile + docker-compose (API + cache) (M)
- [ ] 6.4 CI: ruff + mypy + pytest, fully offline via `FakeProvider`, no live API keys required (M)

**Exit:** `docker compose up` runs the full service; CI green with zero live keys set.

## Phase 7 — Eval & golden sets (3–4 days)

- [ ] 7.1 Golden set A — evidence-provided, 80–100 items: `answerable_in_evidence`, `unanswerable_in_evidence`, `injected_contradiction` (M)
- [ ] 7.2 Golden set B — open-domain, sampled from FEVER / TruthfulQA / HaluEval (public, labeled) (M)
- [ ] 7.3 `metrics.py`: PR-AUC vs prevalence baseline, Brier, ECE + reliability diagram, abstention P/R, quote-verification rate, provider disagreement rate, latency p50/p95, cost/query. Never report bare "accuracy" (M)
- [ ] 7.4 Wire golden set A as a CI gate, cached responses, reproducible offline (S)

**Exit:** `python -m halludetect.eval --suite golden` runs twice, second run is a byte-identical cache hit with network disabled; wired into CI.

## Phase 8 — Hardening & v1.0 launch (2–3 days)

- [ ] 8.1 Secrets review: every provider key is `SecretStr`, none reachable in logs/tracebacks (S)
- [ ] 8.2 Load-test rate limiter and free-model rotation under concurrent load (M)
- [ ] 8.3 README + API docs + "bring your own key" setup guide (S)
- [ ] 8.4 Tag `v1.0.0`, write `CHANGELOG.md` (S)

---

## Critical path

Phase 2 (provider engine) and Phase 3 (evidence acquisition) have no dependency on each other — parallelizable. Phase 4 depends on both being done. Everything from Phase 5 onward is sequential.

## Biggest known risk

Free OpenRouter models are inconsistent at honoring structured JSON output. Do not shortcut Phase 2.4 (capability probe + repair-retry) — skipping it reintroduces v1's parsing bugs in a new shape, just spread across more providers instead of one.

## Verification checklist (run after each phase, in order)

- **After Phase 0:** `docs/contract.md` exists, is the only source of truth for the schema, reviewed.
- **After Phase 1:** any provider swappable via `.env` alone; no vendor-specific import outside `llm/`.
- **After Phase 2:** kill one free model's API key/endpoint mid-run — router fails over to the next healthy free model without a crash.
- **After Phase 3:** submit a query with no evidence and no search key configured — response is `NOT_VERIFIABLE / no_evidence_configured`, not a guess.
- **After Phase 4:** run `tests/test_parsers.py` — all historical parser-bug cases pass across ≥2 providers; no `SUPPORTED` verdict survives without a substring-verified quote.
- **After Phase 5:** `curl -X POST /v1/verify` with a valid key succeeds; without a key, 401; over rate limit, 429.
- **After Phase 6:** submit the same request twice — second is a cache hit (log event + sub-second latency); pull the network — request fails within configured timeout, not indefinitely.
- **After Phase 7:** `python -m halludetect.eval --suite golden` passes with network disabled on the second run.
- **After Phase 8:** fresh clone → `pip install -e .` → `pytest` green with zero API keys set in the environment.
