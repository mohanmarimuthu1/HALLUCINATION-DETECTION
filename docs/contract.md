# HALLUDETECT API Contract

`contract_version: v1`

This document is the single source of truth for the `/v1/verify` request and
response schema. Per the project plan (Phase 0, "Spec lock"), it is frozen
once committed and must not change without a version bump to
`contract_version`. Implementation code in later phases must conform to this
document, not the other way around.

Source of truth precedence: this file, then `docs/openapi.yaml` (machine-
readable mirror of the same schema), then code.

---

## Endpoint

```
POST /v1/verify
```

Verifies whether a given answer is grounded in supplied or fetched evidence.
Never falls back to the model's own world knowledge to justify a `SUPPORTED`
or `GROUNDED` verdict — absence of evidence always resolves to
`NOT_VERIFIABLE`, never a guess.

---

## Request schema

| Field | Type | Required | Meaning |
|---|---|---|---|
| `answer` | string | required | The answer text to verify. |
| `question` | string | optional | The question the answer responds to. Improves claim extraction; not required for verification itself. |
| `evidence` | array of strings | optional | Caller-supplied evidence chunks. If present, the answer is verified against this evidence **only** — no other evidence source is consulted. |
| `evidence_source` | enum: `none \| web \| custom` | required | Where evidence comes from when `evidence` is not supplied directly. `none` = no evidence acquisition attempted. `web` = service performs a web search (only if a search provider key is configured). `custom` = caller's own retriever/RAG plugin acts as the evidence source (Phase 3.4). |
| `model_prefs` | object | optional | See `ModelPrefs` below. If omitted, defaults to the OpenRouter free-model pool. |
| `model_prefs.provider` | enum: `openrouter \| gemini \| openai \| anthropic \| custom` | optional | LLM backend to use. Default `openrouter`. |
| `model_prefs.allow_free_pool` | boolean | optional | Whether the OpenRouter free-model pool may be used. Default `true`. |
| `model_prefs.pinned_model` | string or null | optional | Force a specific model id, bypassing rotation. Default `null`. |
| `model_prefs.user_api_key` | string or null | optional | Caller's own provider API key. Required if `provider` is not `openrouter` and the caller wants to use their own key rather than the shared free pool. Never logged, never echoed back in the response. |

### `evidence` vs `evidence_source`

- If `evidence` is a non-empty array, it is authoritative: the service verifies against exactly this evidence and ignores `evidence_source`.
- If `evidence` is absent or empty, `evidence_source` determines behavior:
  - `custom` — the caller's registered evidence-source plugin is invoked.
  - `web` — the web-search evidence source is invoked, only if a search provider key is configured server-side (see "Web search provider decision" below). If no key is configured, this is treated identically to `none`.
  - `none` — no evidence is acquired. Response is forced to `verdict: NOT_VERIFIABLE`, `reason: no_evidence_configured`.

---

## Response schema — `AnalysisResult`

| Field | Type | Meaning |
|---|---|---|
| `request_id` | string | Unique id for this verification request, for tracing/logging. |
| `verdict` | enum: `GROUNDED \| CONTRADICTED \| NOT_ENOUGH_INFO \| NOT_VERIFIABLE` | Overall verdict for the answer, derived from the fused per-claim labels. |
| `p_hallucinated` | float, 0.0–1.0 | Calibrated probability the answer contains at least one unsupported/contradicted claim. |
| `groundedness` | float, 0.0–1.0 | Calibrated fraction of verifiable claims that are supported. |
| `groundedness_ci` | tuple of 2 floats | Wilson 95% confidence interval for `groundedness`, as `[low, high]`. |
| `claims` | array of `ClaimResult` | Per-claim breakdown. See below. |
| `n_verifiable_claims` | integer | Count of extracted claims that were of type `FACTUAL` and eligible for verification (see Phase 4.1: `OPINION`/`INSTRUCTION`/`META` claims are excluded and do not count here). |
| `model_used` | object: `{provider: string, model: string}` | The LLM backend actually used to answer this request (after routing/fallback). |
| `cost_usd` | float | Estimated cost in USD for this request. `0.0` when using the free model pool. |
| `timings_ms` | object: `{total, retrieval, extraction, verification: integer}` | Latency breakdown in milliseconds. |
| `calibration_version` | string | Identifier for the scoring/calibration model version that produced `p_hallucinated`/`groundedness`, so results remain comparable across deployments. |

### `ClaimResult`

| Field | Type | Meaning |
|---|---|---|
| `claim_id` | string | Stable id for this claim within the request. Verdicts are joined to claims by this id, never by array position (Phase 4.2). |
| `text` | string | The extracted claim text. |
| `label` | enum: `SUPPORTED \| CONTRADICTED \| NOT_ENOUGH_INFO` | Per-claim verification label. Note this is a distinct, smaller enum than the top-level `verdict` field — there is no per-claim `NOT_VERIFIABLE`; that concept only applies to the whole response when `n_verifiable_claims < 3` or no evidence was available. |
| `confidence` | float, 0.0–1.0 | Model confidence in `label`. |
| `evidence_chunk_ids` | array of strings | Ids of the evidence chunks cited for this claim. |
| `quote` | string | Verbatim quote from the cited evidence chunk supporting `label`. |
| `quote_verified` | boolean | Whether `quote` was programmatically confirmed to be a verbatim (or near-verbatim, see Phase 4.3 quote-grounding check) substring of the cited evidence chunk. |

---

## Contract rules (non-negotiable, enforced server-side)

1. **Low claim count → forced `NOT_VERIFIABLE`.** If `n_verifiable_claims < 3`, the response `verdict` is forced to `NOT_VERIFIABLE` regardless of what the per-claim labels say. A bare percentage is never shown for `n_verifiable_claims <= 2` — there isn't enough signal to report one meaningfully.
2. **No evidence, no guess.** If `evidence_source: none` and no web-search provider key is configured, the response `verdict` is `NOT_VERIFIABLE` with reason `no_evidence_configured`. The service never silently answers from the underlying LLM's own world knowledge.
3. **`SUPPORTED` requires a verified quote.** A claim can only carry `label: SUPPORTED` if `quote_verified: true`. If the quote-grounding check fails, the claim is downgraded to `label: NOT_ENOUGH_INFO` server-side, flagged `quote_unverified`, before the response is returned. A `SUPPORTED` label with an unverified quote must never reach a caller.

These three rules are the direct fix for the defect that made v1 unusable: a model was allowed to fall back to its own general knowledge and have that labeled "supported."

---

## Web search provider decision (Phase 0.2)

**Decision: Tavily**, for the `web` evidence source (used when `evidence_source: web` and `evidence` is not supplied directly).

Reasoning:
- Tavily offers a genuinely perpetual free tier (1,000 searches/month, no credit card required), whereas Serper's free access is a one-time trial credit with no ongoing free tier.
- Tavily is purpose-built for LLM/RAG pipelines: it returns cleaned, extracted page content ready to use as evidence chunks, rather than raw Google SERP HTML that would need additional scraping and parsing (Serper's output).
- This keeps Phase 3's `WebSearchEvidence` implementation simpler and avoids adding an HTML-scraping dependency to the evidence pipeline.

Configuration: `TAVILY_API_KEY`, optional. Its absence means the `web` evidence source is simply unavailable — requests with `evidence_source: web` and no key configured behave the same as `evidence_source: none` (verdict `NOT_VERIFIABLE`, reason `no_evidence_configured`). This must never crash the request or silently fall back to a different evidence path.

This decision covers Phase 0.2 only (provider selection). Implementation is Phase 3.2.
