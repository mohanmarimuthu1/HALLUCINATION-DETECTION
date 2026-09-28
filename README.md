# Hallucination Detection

This repo is mid-migration. The active project is **HALLUDETECT v2**
(`src/halludetect/`), a standalone hallucination-verification API service.
It supersedes the original Streamlit self-RAG demo at the repo root
(`app.py`, `detection/`, `rag/`, `knowledge_base/`), which is kept for
reference until the migration finishes but is not where new work happens.

See `docs/contract.md` for the frozen API contract this service implements.

## HALLUDETECT v2

Verifies whether a given answer is grounded in supplied or fetched
evidence. It never lets a verifier fall back on an LLM's own world
knowledge and call that "supported" - no evidence available always means
`NOT_VERIFIABLE`, never a guess. See `docs/contract.md` for the full,
versioned request/response contract.

### Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # or `source .venv/bin/activate` on Linux/Mac
pip install -e ".[dev]"
cp .env.example .env
```

Fill in `.env`:
- `OPENROUTER_API_KEY` - the default backend is OpenRouter's free-model
  pool; without a key, every request that needs an LLM call fails.
- `CLIENT_API_KEYS` - one or more caller-facing keys you invent yourself
  (not from a vendor), comma-separated. Without at least one, every
  `/v1/verify` request is rejected with 401.

Everything else in `.env.example` is optional and degrades gracefully if
left blank (a missing web-search key just disables `evidence_source: web`,
for example - it never crashes a request).

### Run the tests

```bash
pytest
```

The full suite runs offline - no network access and no live API keys
required.

### Run the service

```bash
uvicorn halludetect.api.main:app --reload
```

```bash
curl -X POST http://localhost:8000/v1/verify \
  -H "Authorization: Bearer <one of your CLIENT_API_KEYS>" \
  -H "Content-Type: application/json" \
  -d '{
        "answer": "The Eiffel Tower is 330 metres tall, was completed in 1889, and stands in Paris.",
        "evidence": ["The Eiffel Tower is on the Champ de Mars in Paris, France. It is 330 metres tall.",
                     "Construction began in 1887 and it was completed in 1889."],
        "evidence_source": "none"
      }'
```

An answer needs at least 3 factual claims for an overall verdict; fewer
returns `NOT_VERIFIABLE` with `reason: insufficient_verifiable_claims`.
No evidence returns `NOT_VERIFIABLE` with `reason: no_evidence_configured`.

`GET /healthz` is unauthenticated (liveness probe). `POST /v1/verify`
requires a valid `Authorization: Bearer <key>` and is rate-limited per key.

### Run the UI

A thin Streamlit client for the API above. It has no detection logic of
its own - everything it shows comes from `/v1/verify`.

```bash
pip install -e ".[ui]"
uvicorn halludetect.api.main:app            # terminal 1
streamlit run src/halludetect/ui/app.py     # terminal 2, http://localhost:8501
```

Run locally it uses the first of `CLIENT_API_KEYS` from the same `.env`,
so there is nothing extra to configure. Set `HALLUDETECT_API_URL` /
`HALLUDETECT_API_KEY` to point it at a service running elsewhere.

### Bring your own key

By default every request is served through this deployment's own
OpenRouter free-model pool (`model_prefs.provider: openrouter`, the
default). A caller can instead route a single request through their own
Gemini, OpenAI, or Anthropic key via `model_prefs`:

```bash
curl -X POST http://localhost:8000/v1/verify \
  -H "Authorization: Bearer <one of your CLIENT_API_KEYS>" \
  -H "Content-Type: application/json" \
  -d '{
        "answer": "The Eiffel Tower is 330 metres tall, was completed in 1889, and stands in Paris.",
        "evidence": ["The Eiffel Tower is on the Champ de Mars in Paris, France. It is 330 metres tall.",
                     "Construction began in 1887 and it was completed in 1889."],
        "evidence_source": "none",
        "model_prefs": {
          "provider": "gemini",
          "user_api_key": "<caller-supplied Gemini API key>",
          "pinned_model": "gemini-1.5-flash"
        }
      }'
```

- `model_prefs.provider` - `openrouter` (default) | `gemini` | `openai` |
  `anthropic` | `custom`. `custom` needs the deployment's own
  `CUSTOM_PROVIDER_BASE_URL` set in `.env` first - it's a
  deployment-level endpoint, not something a caller can point anywhere
  per-request.
- `model_prefs.user_api_key` - the caller's own provider key. Falls back
  to this deployment's own key for that provider (if configured) when
  omitted. Never logged and never echoed back in the response - only
  hashed into the result-cache key, so two callers with different keys
  never share a cached answer (`docs/contract.md`).
- `model_prefs.pinned_model` - forces a specific model id, bypassing
  OpenRouter's free-pool rotation. Required for `custom`, optional
  everywhere else (each provider falls back to its own default model).
- `model_prefs.allow_free_pool` - set `false` to require a pinned/paid
  model instead of ever falling back to the shared free pool.

### Layout

```
src/halludetect/
  settings.py     pydantic-settings, SecretStr per provider/client key
  logging.py      structlog JSON logging, per-request request_id
  llm/            LLMProvider per backend (OpenRouter free pool, Gemini,
                  OpenAI, Anthropic, generic OpenAI-compatible), router
                  with health tracking + circuit breaker
  evidence/       EvidenceSource per mode (caller-supplied, web search,
                  none, a caller's own retriever)
  detect/         claim extraction, verification, quote-grounding,
                  calibrated fusion - the pipeline itself
  api/            FastAPI app: /v1/verify, /healthz, auth, rate limiting
docs/
  contract.md     the frozen API contract (source of truth)
  openapi.yaml    machine-readable mirror of the same contract
tests/            fully offline - monkeypatched HTTP, no live keys
```

## Legacy v1 app (reference only)

The original Streamlit demo: RAG over a local knowledge base with a
claim-extraction/fact-verification pass on top. Superseded by v2 above -
its detector had structural defects (substring-matched verdicts, no quote
grounding, a self-learning loop that fed hallucinated answers back into
its own knowledge base) that v2's design specifically avoids.

```bash
pip install -r requirements.txt
python init_legacy_app.py   # first-time setup: builds the vector store
streamlit run app.py        # http://localhost:8501
```

Configuration lives in `config.py` / `.env` (`GOOGLE_API_KEY`,
`OPENROUTER_API_KEY_1`, `OPENROUTER_API_KEY_2`). These are different names
from v2's: setting `OPENROUTER_API_KEY` configures v2 only, and the legacy
app will not see it.

Do not use its output as a verdict. When its model calls fail it does not
report an error: `detection/fact_verifier.py` substitutes hardcoded
fallback results, so with no working key it still renders a score
("PARTIALLY SUPPORTED, 60%") built from no model output at all. Its
verification prompt also tells the model to use its own general
knowledge when the evidence is silent. The same detector backs `demo.py`
and `evaluate.py`, so numbers from either inherit both problems.
