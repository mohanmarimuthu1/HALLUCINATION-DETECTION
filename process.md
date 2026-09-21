# Process Log

Tracks what has been done, what is pending, and the next step. Update this
file at the end of every work session — do not let it drift from reality.

## Status: Phase 0 complete (spec lock). Phase 1 started — task 1.1 done (1 of 3 tasks).

## What is done

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
  - **Known blocker, not fixed**: `pip install -e .` currently fails.
    The repo has a root-level `setup.py` (a legacy interactive init
    script for the v1 app — it prints emoji banners — not a packaging
    script) which `setuptools` picks up and executes as a legacy build
    hook, and it crashes with `UnicodeEncodeError` on Windows' default
    `cp1252` console encoding. This blocks real `pip install`-based
    verification of `pyproject.toml` until `setup.py` is renamed (and
    `README.md`'s reference to `python setup.py` updated to match) or
    replaced with a `[project.scripts]` entry point. Left untouched for
    now since it's a legacy-app file, not part of Phase 1's task list —
    flagging so it isn't mistaken for "done."
  - Tasks 1.2 (`LLMProvider` protocol + Gemini/OpenAI/Anthropic/generic
    implementations) and 1.3 (structlog JSON logging with a
    `request_id` contextvar) are **not started**.

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

Rest of Phase 1 (1.2 provider abstraction, 1.3 structlog logging), then
Phases 2-8 in `plan.md` — the v2 rewrite is only just started. The current
root-level code (`app.py`, `config.py`, `detection/`, `rag/`,
`knowledge_base/`) is the **legacy v1 app** described in `plan.md`'s
Context section; it is not yet superseded and still runs, but it is not
where new work should go.

The `setup.py`/`pip install -e .` naming collision above should be fixed
before Phase 1's exit criterion can be verified end-to-end via a real
install.

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

## Next process

Finish **Phase 1 — Core skeleton + provider abstraction** in `plan.md`:
1.2 — `LLMProvider` Protocol in `src/halludetect/llm/base.py` plus
Gemini/OpenAI/Anthropic/generic-OpenAI-compatible implementations; 1.3 —
structlog JSON logging with a `request_id` contextvar. Resolve the
`setup.py` naming collision so `pip install -e .` works, since Phase 1's
exit criterion ("swapping provider is a config change, not a code
change") should be verified via a real install, not `PYTHONPATH` tricks.
Do not start Phase 2 until that exit criterion passes.
