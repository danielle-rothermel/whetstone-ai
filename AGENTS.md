# Repository instructions

Graph-based HumanEval evaluation platform (`whetstone`). Generation and scoring
run through graph-shaped specs, explicit LM/prompt boundaries, DBOS workflows,
and append-only terminal outcomes under `whetstone.platform`. See
[README.md](README.md) for package layout and CLI commands.

## Plans and priorities

Plans, priorities, and backlog live in **Linear** (DevInfra team, Code
Compression Paper project), not in this repo. Docs under `docs/` are reference
material, not active task lists. Record follow-up work in Linear rather than in
repository docs.

## Environment

- Python >= 3.13, managed with `uv`. Install dependencies with
  `uv sync --group dev` and run commands through `uv run ...`.
- Postgres connection comes from `DATABASE_URL`; when unset it defaults to
  peer-auth `postgresql+psycopg:///dr_dspy`. Copy `.env.example` to `.env` to
  adjust. Bare `postgresql://` URLs are normalized to `postgresql+psycopg://`
  by Alembic and the platform CLIs.
- Apply the schema with `uv run alembic upgrade head`. Schema tables live in
  [`src/whetstone/db/schema.py`](src/whetstone/db/schema.py).

## Tests and checks

See [TESTING.md](TESTING.md) for the tier model, fixtures, and conventions.

- Unit: `./scripts/ci/unit.sh` (fast; no Postgres or DBOS required).
- Lint/type: `./scripts/ci/lint.sh` (`ruff check` + `ty check`).
- Integration: `DATABASE_URL=postgresql+psycopg:///dr_dspy ./scripts/ci/integration.sh`
  (opt-in via `@pytest.mark.integration`; skips cleanly without Postgres).

## Conventions

- Generation, node, and score outcome rows are append-only; batch submit audit
  rows are mutable operational audit.
- Never rewrite Alembic revision history. Add forward revisions only.
- Keep docs forward-facing. Dated project history belongs in
  [CHANGELOG.md](CHANGELOG.md).
