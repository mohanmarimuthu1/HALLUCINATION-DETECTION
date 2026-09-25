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
        "answer": "The Eiffel Tower is in Paris.",
        "evidence": ["The Eiffel Tower is located in Paris, France."],
        "evidence_source": "none"
      }'
```

`GET /healthz` is unauthenticated (liveness probe). `POST /v1/verify`
requires a valid `Authorization: Bearer <key>` and is rate-limited per key.

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
`OPENROUTER_API_KEY_1`, `OPENROUTER_API_KEY_2`).
