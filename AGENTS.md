# Repository instructions

Graph-based HumanEval evaluation platform (`dr_dspy`, planned rename:
`whetstone`). Generation and scoring run through graph-shaped specs, explicit
LM/prompt boundaries, and append-only terminal outcomes under
`dr_dspy.platform`. See [README.md](README.md) for the package layout and CLI
commands.

## Plans and priorities

Plans, priorities, and backlog live in **Linear** (DevInfra team, Code
Compression Paper project) — not in this repo. Docs under `docs/` are
reference material, not active task lists. When you notice follow-up or
future work, record it in Linear rather than in a doc.

A large extraction PR (#4, `composable-migration`) is open; check it before
starting structural work on package boundaries.

## Environment

- Python >= 3.12, managed with `uv`. Install dependencies with
  `uv sync --group dev` (the `dev` group carries the pinned `dspy` used by
  serialization contract tests). Run everything through `uv run ...`.
- Postgres connection comes from `DATABASE_URL`; when unset it defaults to
  peer-auth `postgresql+psycopg:///dr_dspy`. Copy `.env.example` to `.env` to
  adjust. Bare `postgresql://` URLs are normalized to `postgresql+psycopg://`
  by Alembic and the platform CLIs.
- Apply the v1 schema with `uv run alembic upgrade head`. The canonical head
  lives in [`src/dr_dspy/db/migrations/head.py`](src/dr_dspy/db/migrations/head.py);
  see [docs/v1-schema-migrations.md](docs/v1-schema-migrations.md).

## Tests and checks

See [TESTING.md](TESTING.md) for the tier model, fixtures, and conventions.

- Unit: `./scripts/ci/unit.sh` (fast; no Postgres or DBOS required).
- Lint/type: `./scripts/ci/lint.sh` (`ruff check` + `ty check`).
- Integration: `DATABASE_URL=postgresql+psycopg:///dr_dspy ./scripts/ci/integration.sh`
  (opt-in via `@pytest.mark.integration`; skips cleanly without Postgres).
- Golden gate: `uv run pytest -k golden` pins parser/scoring/digest behavior
  with committed fixtures. The golden suite lands with PR #4 and applies on
  branches that include it; a failing golden test means fix your change, not
  the fixtures.

## Conventions

- Generation, node, and score outcome rows are **append-only** (enforced by
  DB triggers); batch submit audit rows are mutable operational audit.
- Never rewrite Alembic migration history — forward revisions only. See
  [docs/v1-schema-migrations.md](docs/v1-schema-migrations.md) for the chain,
  post-freeze policy, and reset procedure for draft databases.
- Keep legacy v0 Postgres tables (`dr_dspy_eval_predictions`,
  `dr_dspy_encdec_eval_predictions`, etc.) as read-only backup unless
  explicitly asked to drop them — see
  [docs/v0-migration-completion-checklist.md](docs/v0-migration-completion-checklist.md).
