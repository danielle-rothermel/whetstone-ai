# Testing

This document describes how to run tests, the tier model for platform integration
coverage, shared fixtures, and conventions for adding new tests.

## Overview

The default pytest suite is fast and does not require Postgres or a DBOS system
database. Integration tests are opt-in via the `@pytest.mark.integration` marker
and skip gracefully when PostgreSQL is unavailable.

| Command | What runs |
|---------|-----------|
| `./scripts/ci/unit.sh` | Unit tests (excludes integration marker) |
| `./scripts/ci/integration.sh` | Postgres + DBOS integration proofs (generation + scoring) |
| `./scripts/ci/lint.sh` | `ruff check` + `ty check` |
| `./scripts/ci/coverage.sh` | Unit + integration tests with combined coverage report |
| `uv run pytest tests/test_v0_reshape.py` | v0 reshape unit smoke (no database) |

## Test tiers

| Tier | Purpose | Location |
|------|---------|----------|
| **Unit** | Pure graph orchestration, record contracts, SQL compilation, reshape logic | `tests/test_*.py` (except `tests/integration/`) |
| **0 — Fixtures** | Shared Postgres schema + DBOS reset helpers | [`tests/conftest.py`](tests/conftest.py) |
| **1 — DB steps** | Generation: `load_prediction_spec_step` / `persist_generation_result_step` Postgres round-trip. Scoring: `load_scoring_target_step` / `persist_score_attempt_step` idempotency and profile uniqueness | [`tests/integration/test_platform_db_steps.py`](tests/integration/test_platform_db_steps.py), [`tests/integration/test_platform_scoring_db_steps.py`](tests/integration/test_platform_scoring_db_steps.py) |
| **2 — Workflow** | Generation: `run_prediction_graph_workflow_once` happy path with mocked LM. Scoring: `run_score_generation_workflow_once` with mocked HumanEval task load | [`tests/integration/test_platform_dbos_workflow.py`](tests/integration/test_platform_dbos_workflow.py), [`tests/integration/test_platform_scoring_dbos_workflow.py`](tests/integration/test_platform_scoring_dbos_workflow.py) |
| **3 — Recovery** | Generation: retry-exhaustion step/timestamp assertions, upstream `BLOCKED` runs, error-path idempotent replay, duplicate-start recovery, persist idempotency, persist failure surfacing. Scoring: workflow replay idempotency, task-loader memoization, orphan workflow recovery | [`tests/integration/test_platform_dbos_workflow.py`](tests/integration/test_platform_dbos_workflow.py), [`tests/integration/test_platform_scoring_dbos_workflow.py`](tests/integration/test_platform_scoring_dbos_workflow.py) |
| **3.5 — Migration smoke** | Frozen v0 samples → v1 reshape → import / workflow pass-through (only remaining v0-related tier until backfill completes) | [`tests/integration/test_v0_reshape_*.py`](tests/integration/), [`tests/test_v0_reshape.py`](tests/test_v0_reshape.py) |
| **4 — Pipeline E2E** | JSONL submit → real DBOS enqueue → in-process queue consumer → generation → scoring (mock LM + HumanEval loader only) | [`tests/integration/test_platform_pipeline_e2e.py`](tests/integration/test_platform_pipeline_e2e.py) |

Design context: [append-only eval platform design](docs/append-only-eval-records-design.md),
[platform graph workflow notes](docs/platform-graph-workflow-implementation.md).

## Layout

```
tests/
  conftest.py                 # integration fixtures (Postgres schema, reset_dbos)
  serialization_support.py    # helpers for serialization contract tests
  support/                    # shared spec/node helpers for unit + integration
    platform_integration_helpers.py
    platform_scoring_fixtures.py
    platform_workflow_fixtures.py
    jsonl_fixtures.py
    postgres_fixtures.py
  fixtures/v0_samples/        # committed JSON rows from legacy v0 tables
  integration/                # @pytest.mark.integration tests
    dbos_test_workflows.py    # minimal workflows for step-level DBOS proofs
    test_platform_db_steps.py
    test_platform_dbos_workflow.py
    test_platform_scoring_db_steps.py
    test_platform_scoring_dbos_workflow.py
    test_platform_pipeline_e2e.py
    test_v0_reshape_outcomes.py
    test_v0_reshape_specs.py
scripts/ci/                   # portable CI entrypoints (package-root cwd)
  unit.sh
  integration.sh
  coverage.sh
  lint.sh
src/dr_dspy/migration/        # v0 → v1 reshape logic (not inline in tests)
```

## Shared fixtures

Defined in [`tests/conftest.py`](tests/conftest.py):

- **`app_postgres_schema`** — creates an isolated schema, applies v1 migrations +
  append-only triggers, exposes `database_url` with `search_path` set for steps
  that open their own SQLAlchemy engines.
- **`reset_dbos`** — destroys/reconfigures DBOS, resets the system database
  (SQLite file under `tmp_path` by default), and launches the platform runtime.
- **`reset_dbos_generation_consumer`** — like `reset_dbos`, but listens to the
  platform generation queue and registers a worker before yielding (for Tier 4
  pipeline tests).

Seed helpers live in [`tests/support/postgres_fixtures.py`](tests/support/postgres_fixtures.py):

- **`seed_prediction_spec(connection, spec)`** — inserts experiment + spec rows.
- **`start_test_workflow(workflow, workflow_id, *args)`** — DBOS workflow helper.

Integration polling helpers live in
[`tests/support/platform_integration_helpers.py`](tests/support/platform_integration_helpers.py):

- **`wait_for_workflow_result(workflow_id)`** — poll DBOS until a workflow
  reaches a terminal status.

## Conventions

### Mock boundaries

- **Workflow integration tests:** mock only the LM boundary (`execute_lm_node` or
  provider caller). Do not mock DB steps under test.
- **Pipeline E2E (Tier 4):** mock LM and HumanEval task load only; use real
  JSONL submit, enqueue, queue consumption, Postgres persistence, and scoring
  workflow steps.
- **Unit orchestration tests:** may mock all steps and use `.__wrapped__` to
  verify call order without DBOS overhead.

### Anti-patterns for contract tests

- Using `run_prediction_graph_workflow.__wrapped__` when the goal is to prove
  DBOS memoization, replay, or step registration.
- Using `_RecordingConnection` when the goal is a real Postgres round-trip.
- Putting migration reshape logic inline in test files (belongs in
  `src/dr_dspy/migration/`).

### v0 sample fixtures

Legacy rows live in [`tests/fixtures/v0_samples/`](tests/fixtures/v0_samples/) as
committed JSON. CI does not require live v0 tables. Delete fixtures and Tier 3.5
tests after backfill validation — see
[`docs/v0-migration-completion-checklist.md`](docs/v0-migration-completion-checklist.md).

### Adding new tests

1. Pick the tier (unit vs integration vs migration smoke).
2. Reuse helpers in `tests/support/` before adding new factories.
3. Integration tests must skip cleanly when Postgres is unavailable.
4. Append a dated entry to the changelog below when test infrastructure, tiers,
   fixtures, markers, or CI invocation changes materially.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | App Postgres URL (defaults to `postgresql+psycopg:///dr_dspy`) |
| `DBOS_SYSTEM_DATABASE_URL` | Optional; integration tests use a per-test SQLite file when unset |

Integration tests compose `app_postgres_schema` with `reset_dbos` when DBOS
workflows are under test. Tier 4 pipeline tests use `reset_dbos_generation_consumer`
instead.

## Coverage

Combined coverage runs unit tests first, then appends integration test
coverage. The threshold is enforced on the merged report because migration
backfill/downgrade proofs and submit idempotency live in the integration tier.

| Command | Notes |
|---------|-------|
| `./scripts/ci/coverage.sh` | Requires Postgres (`DATABASE_URL`) for integration append |
| `uv sync --group dev` | Installs `coverage` and `pytest-cov` dev dependencies |

Current `fail_under` threshold: **88%** in [`pyproject.toml`](pyproject.toml)
(`[tool.coverage.report]`). Combined unit + integration coverage is typically
**~94%**; unit-only coverage alone is not sufficient for DB/migration gaps.

CI runs a dedicated **Coverage** job (see below) that executes
`./scripts/ci/coverage.sh` with a Postgres 16 service.

## CI

GitHub Actions workflow: [`.github/workflows/whetstone_tests.yml`](.github/workflows/whetstone_tests.yml).
Jobs run from the standalone repository root.

| Job | Script | Notes |
|-----|--------|-------|
| lint | `./scripts/ci/lint.sh` | ruff + ty |
| unit | `./scripts/ci/unit.sh` | unit tests |
| integration | `./scripts/ci/integration.sh` | Postgres 16 service; integration tests |
| coverage | `./scripts/ci/coverage.sh` | Postgres 16 service; combined coverage gate |

Local equivalents (from the package root):

```bash
./scripts/ci/lint.sh
./scripts/ci/unit.sh
./scripts/ci/coverage.sh
DATABASE_URL=postgresql+psycopg:///dr_dspy ./scripts/ci/integration.sh
```

Install git hooks (runs the same lint script on commit and push):

```bash
uv run pre-commit install
```

DSPy resolves from the pinned PyPI dependency in `pyproject.toml` and `uv.lock`.

[`Dockerfile.ci`](Dockerfile.ci) builds a unit-test image for future Depot
wiring after the org repo is created.

## Changelog

### 2026-06-30 — Remove unused `lm.utils` symbols

- Deleted v0-only helpers (`LmEventBuffer`, `response_text`, `stable_json`,
  `usage_metadata_from_response`, `ModelConfig`); boundary uses
  `content_to_text` and `provider_cost_from_response` only.
- Cleared outstanding testing backlog from planning docs; Tier 4 E2E marked done.

### 2026-06-30 — Coverage gap closure

- Added `./scripts/ci/coverage.sh` and a **Coverage** CI job with an 88%
  combined threshold (`coverage` + `pytest-cov` dev dependencies).
- Removed unused `db/io.py` `insert_*_on_conflict_do_nothing` helpers; added
  Postgres submit idempotency integration tests.
- Added validator, humaneval, serialization, migration backfill/downgrade, and
  Alembic env offline SQL smoke tests.

### 2026-06-30 — High-risk coverage: prompts, import inference, worker CLI, pipeline E2E

- Added unit tests for `platform/prompts.py` error paths, `humaneval/import_inference.py`,
  and `platform/worker.py` CLI wiring (`score-one`, `submit-jsonl`, `worker`, `rescore`).
- Added Tier 4 pipeline integration test:
  JSONL submit → DBOS enqueue → queue consumer → generation → scoring.
- Added `reset_dbos_generation_consumer` fixture, `wait_for_workflow_result` helper,
  and `tests/support/jsonl_fixtures.py`.

### 2026-06-30 — Standalone repository CI

- Added `.github/workflows/whetstone_tests.yml` for root-based lint, unit, and
  integration jobs.
- Updated `Dockerfile.ci` for the standalone root layout.
- Removed the fork-only `ensure_pypi_dspy.sh` workflow step.

### 2026-06-30 — CI scripts and GitHub workflow

- Added `scripts/ci/{unit,integration,lint}.sh`.
- Added the original fork-scoped GitHub workflow before extraction.
- Pinned `dspy==3.3.0b1`.
- Fixed `tests/test_serialization.py` import path.
- Added `Dockerfile.ci` for future Depot wiring.

### 2026-06-30 — v0 runtime archive

- Removed v0 experiment/harness/legacy LM runtime code; platform DBOS bootstrap
  now lives under `src/dr_dspy/platform/`.
- Kept `migration/v0_reshape.py` and Tier 3.5 fixtures for backfill.
- Added [`docs/v0-migration-completion-checklist.md`](docs/v0-migration-completion-checklist.md).

### 2026-06-30 — Platform integration + v0 migration smoke tiers

- Added tiered integration test model (Tiers 0–3 and 3.5).
- Added `tests/conftest.py` shared fixtures and `@pytest.mark.integration`.
- Added `src/dr_dspy/migration/v0_reshape.py` and frozen v0 JSON fixtures.
- Fixed optional JSONB columns in `persist_generation_result` to insert SQL
  `NULL` instead of JSON `null` for Postgres check constraints.
- Added `TESTING.md` as canonical testing documentation.
