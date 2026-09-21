# Process Log

Tracks what has been done, what is pending, and the next step. Update this
file at the end of every work session — do not let it drift from reality.

## Status: Phase 0 complete (spec lock)

## What is done

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

Everything from Phase 1 onward in `plan.md` — the actual v2 rewrite
(`src/halludetect/...`) has not started. The current root-level code
(`app.py`, `config.py`, `detection/`, `rag/`, `knowledge_base/`) is the
**legacy v1 app** described in `plan.md`'s Context section; it is not yet
superseded and still runs, but it is not where new work should go. Phase 1
builds the new `src/` layout from scratch alongside it.

Known outstanding issue not yet fixed in the legacy app: the hardcoded API
keys committed in prior git history are still exposed in git log/GitHub even
though `config.py` no longer contains them on disk. They should be treated
as compromised.

## How Phase 0 was done

Read `plan.md`'s API contract section (frozen request/response shape) and
transcribed it verbatim into `docs/contract.md` as prose + tables, then
mirrored the same shape into `docs/openapi.yaml` using `$ref`-based
`components/schemas` so the two files cannot drift silently. No provider or
detection code was written — Phase 0 is spec-only per the plan.

## Next process

Start **Phase 1 — Core skeleton + provider abstraction** in `plan.md`:
`src/halludetect/` package layout, `pyproject.toml`, `settings.py` with
`SecretStr` per provider key, the `LLMProvider` protocol plus
Gemini/OpenAI/Anthropic/generic-OpenAI-compatible implementations, and
structlog JSON logging with a `request_id` contextvar. Do not start Phase 2
or later until Phase 1's exit criterion passes: swapping provider is a
config change, not a code change.
