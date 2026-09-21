# Project instructions

This repo is mid-migration. Read `plan.md` and `process.md` before doing
anything else — they are the source of truth for what phase this project is
in and what's done vs. pending. Keep `process.md` updated whenever you
complete or start work; do not let it go stale.

## What this project is

`plan.md` defines **HALLUDETECT v2**: a standalone hallucination
verification service being built from scratch under `src/halludetect/`
(does not exist yet — Phase 0 is spec-only). It supersedes the current
root-level app (`app.py`, `config.py`, `detection/`, `rag/`,
`knowledge_base/`), a Streamlit self-RAG demo whose detector had structural
defects: substring-matched verdicts, no quote grounding, and a self-learning
loop that fed its own hallucinated answers back into its knowledge base as
trusted fact.

`docs/contract.md` is the frozen API contract (`/v1/verify` request and
`AnalysisResult` response schema). It is versioned and must not change
without a version bump — treat it as authoritative over any code that
disagrees with it. `docs/openapi.yaml` is its machine-readable mirror; keep
both in sync if either changes.

## Non-negotiable rule for any detection/verification code

Never let a verifier fall back on the model's own general/world knowledge
and label that "supported" or "grounded." No evidence available means the
verdict is `NOT_VERIFIABLE` — full stop, never a guess. This was the exact
defect that made v1 unusable; every phase of `plan.md` exists to prevent it
from returning in a new shape.

## Working conventions for this repo

- **No secrets in source.** API keys load from environment variables via
  `python-dotenv` (see `config.py`, `.env.example`). Never hardcode a key,
  even temporarily "for a demo." `.env` is gitignored; keep it that way.
- **No generated/binary artifacts in git.** `chroma_db/` (vector store),
  `__pycache__/`, `*.pyc`, and debug dumps (`config_error.txt` and similar)
  do not belong in version control — they're in `.gitignore`. Do not `git
  add` them back. If a task produces scratch output (test run logs, one-off
  debug text files, model probe results), write it to the OS temp/scratch
  directory, not the repo root, and don't commit it.
- **No ad-hoc scripts left in repo root.** One-off probes like "list every
  Gemini model and see which responds" belong in the scratch directory or a
  real `tests/` file, not as a permanent root-level `test_*.py` that never
  runs in CI.
- **Small, atomic commits.** One logical change per commit (a bug fix, a
  doc addition, a dependency bump) — not one giant "updates" commit.
- **Commit authorship**: commit as the repo owner, not as a co-author or
  collaborator. Do not add `Co-Authored-By` trailers to commits in this
  repo unless explicitly asked to.
- **No AI-slop content.** Docs and comments should read like something a
  production team wrote: state the fact, the reason if non-obvious, and
  stop. No filler, no restating the diff in prose, no comments that repeat
  what the code already says.

## Verifying changes

There's no CI yet (that's Phase 6). Before calling a change done:
- `python -m py_compile <changed .py files>` at minimum.
- For the legacy app, `streamlit run app.py` should still start without
  import errors if you touched `app.py`, `config.py`, `detection/`, or
  `rag/`.
- For Phase 0 docs work, any YAML/JSON schema file must actually parse
  (`python -c "import yaml; yaml.safe_load(open(path))"` or equivalent).
