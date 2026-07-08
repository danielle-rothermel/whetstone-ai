# Testing

This document describes how to run tests, the platform integration tiers, shared
fixtures, and conventions for adding coverage.

## Overview

The default pytest suite is fast and does not require Postgres or a DBOS system
database. Integration tests are opt-in via the `@pytest.mark.integration` marker
and skip gracefully when PostgreSQL is unavailable.

| Command | What runs |
|---------|-----------|
| `./scripts/ci/unit.sh` | Unit tests excluding the integration marker |
| `./scripts/ci/integration.sh` | Postgres + DBOS integration proofs |
| `./scripts/ci/lint.sh` | `ruff check` + `ty check` |

Install dev dependencies, including `dspy` for serialization contract tests,
with `uv sync --group dev`. CI scripts assume the dev group is synced.

## Test Tiers

| Tier | Purpose | Location |
|------|---------|----------|
| Unit | Pure orchestration, record contracts, SQL compilation, helper logic | `tests/test_*.py` except `tests/integration/` |
| 0 - Fixtures | Shared Postgres schema and DBOS reset helpers | `tests/conftest.py` |
| 1 - DB steps | Generation and scoring DB round-trips, idempotency, and uniqueness checks | `tests/integration/test_platform_db_steps.py`, `tests/integration/test_platform_scoring_db_steps.py` |
| 2 - Workflow | Generation and scoring workflow happy paths with mocked external boundaries | `tests/integration/test_platform_dbos_workflow.py`, `tests/integration/test_platform_scoring_dbos_workflow.py` |
| 3 - Recovery | Retry exhaustion, replay idempotency, duplicate-start recovery, and orphan workflow recovery | `tests/integration/test_platform_dbos_workflow.py`, `tests/integration/test_platform_scoring_dbos_workflow.py` |
| 4 - Pipeline E2E | JSONL submit to DBOS enqueue to generation to scoring | `tests/integration/test_platform_pipeline_e2e.py` |

Workflow CLI details are in [README.md](README.md#platform-workflow).

## Layout

```text
tests/
  conftest.py
  serialization_support.py
  support/
    jsonl_fixtures.py
    platform_integration_helpers.py
    platform_scoring_fixtures.py
    platform_workflow_fixtures.py
    postgres_fixtures.py
  integration/
    dbos_test_workflows.py
    test_platform_db_steps.py
    test_platform_dbos_workflow.py
    test_platform_pipeline_e2e.py
    test_platform_scoring_db_steps.py
    test_platform_scoring_dbos_workflow.py
scripts/ci/
  integration.sh
  lint.sh
  unit.sh
```

## Shared Fixtures

Defined in `tests/conftest.py`:

- `app_postgres_schema` creates an isolated schema, applies the application
  schema, adopts the dr-platform lineage, and exposes a `database_url` with
  `search_path` set.
- `reset_dbos` destroys and reconfigures DBOS, resets the system database, and
  launches the platform runtime.
- `reset_dbos_generation_consumer` starts the platform runtime, registers a
  generation worker, and listens to the platform queue for pipeline tests.

Seed helpers live in `tests/support/postgres_fixtures.py`:

- `seed_prediction_spec(connection, spec)` inserts experiment and spec rows.
- `start_test_workflow(workflow, workflow_id, *args)` starts a DBOS workflow.

Integration polling helpers live in
`tests/support/platform_integration_helpers.py`:

- `wait_for_workflow_result(workflow_id)` polls DBOS until a workflow reaches a
  terminal status.

## Conventions

### Mock Boundaries

- Workflow integration tests mock only the LM boundary or HumanEval task load.
  Do not mock DB steps under test.
- Pipeline E2E tests mock LM and HumanEval task load only; use real JSONL
  submit, enqueue, queue consumption, Postgres persistence, and scoring
  workflow steps.
- Unit orchestration tests may mock all steps and use `.__wrapped__` to verify
  call order without DBOS overhead.

### Anti-Patterns

- Using `run_prediction_graph_workflow.__wrapped__` when the goal is to prove
  DBOS memoization, replay, or step registration.
- Using `_RecordingConnection` when the goal is a real Postgres round-trip.
- Adding one-off factories when an existing helper in `tests/support/` fits the
  scenario.

### Adding New Tests

1. Pick the tier: unit, workflow integration, recovery integration, or pipeline
   E2E.
2. Reuse helpers in `tests/support/` before adding new factories.
3. Integration tests must skip cleanly when Postgres is unavailable.
4. Keep each test focused on one behavior or contract.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | App Postgres URL; defaults to `postgresql+psycopg:///dr_dspy` |
| `DBOS_SYSTEM_DATABASE_URL` | Optional DBOS system database URL; integration tests use a per-test SQLite file when unset |

## CI

GitHub Actions workflow: `.github/workflows/whetstone_tests.yml`. Jobs run from
the standalone repository root.

| Job | Script | Notes |
|-----|--------|-------|
| lint | `./scripts/ci/lint.sh` | ruff + ty |
| unit | `./scripts/ci/unit.sh` | unit tests |
| integration | `./scripts/ci/integration.sh` | Postgres 16 service; integration tests |

Local equivalents:

```bash
./scripts/ci/lint.sh
./scripts/ci/unit.sh
DATABASE_URL=postgresql+psycopg:///dr_dspy ./scripts/ci/integration.sh
```

Install git hooks:

```bash
uv run pre-commit install
```
