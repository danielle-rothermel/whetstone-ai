# Testing

This document describes how to run tests, the tier model for platform integration
coverage, shared fixtures, and conventions for adding new tests.

## Overview

The default pytest suite is fast and does not require Postgres or a DBOS system
database. Integration tests are opt-in via the `@pytest.mark.integration` marker
and skip gracefully when PostgreSQL is unavailable.

| Command | What runs |
|---------|-----------|
| `./scripts/ci/unit.sh` | Unit tests (294 tests; excludes integration marker) |
| `./scripts/ci/integration.sh` | Postgres + DBOS integration proofs (27 tests) |
| `./scripts/ci/lint.sh` | `ruff check` + `ty check` |
| `uv run pytest tests/test_v0_reshape.py` | v0 reshape unit smoke (no database) |

## Test tiers

| Tier | Purpose | Location |
|------|---------|----------|
| **Unit** | Pure graph orchestration, record contracts, SQL compilation, reshape logic | `tests/test_*.py` (except `tests/integration/`) |
| **0 — Fixtures** | Shared Postgres schema + DBOS reset helpers | [`tests/conftest.py`](tests/conftest.py) |
| **1 — DB steps** | `load_prediction_spec_step` / `persist_generation_result_step` Postgres round-trip | [`tests/integration/test_platform_db_steps.py`](tests/integration/test_platform_db_steps.py) |
| **2 — Workflow** | `run_prediction_graph_workflow_once` happy path with mocked LM | [`tests/integration/test_platform_dbos_workflow.py`](tests/integration/test_platform_dbos_workflow.py) |
| **3 — Recovery** | Retry-exhaustion step/timestamp assertions, upstream `BLOCKED` runs, error-path idempotent replay, duplicate-start recovery, persist idempotency, persist failure surfacing | [`tests/integration/test_platform_dbos_workflow.py`](tests/integration/test_platform_dbos_workflow.py) |
| **3.5 — Migration smoke** | Frozen v0 samples → v1 reshape → import / workflow pass-through | [`tests/integration/test_v0_reshape_*.py`](tests/integration/), [`tests/test_v0_reshape.py`](tests/test_v0_reshape.py) |

Design context: [append-only eval platform design](docs/append-only-eval-records-design.md),
[platform graph workflow notes](docs/platform-graph-workflow-implementation.md).

## Layout

```
tests/
  conftest.py                 # integration fixtures (Postgres schema, reset_dbos)
  serialization_support.py    # helpers for serialization contract tests
  support/                    # shared spec/node helpers for unit + integration
    platform_integration_helpers.py
    platform_workflow_fixtures.py
    postgres_fixtures.py
  fixtures/v0_samples/        # committed JSON rows from legacy v0 tables
  integration/                # @pytest.mark.integration tests
    dbos_test_workflows.py    # minimal workflows for step-level DBOS proofs
    test_platform_db_steps.py
    test_platform_dbos_workflow.py
    test_v0_reshape_outcomes.py
    test_v0_reshape_specs.py
scripts/ci/                   # portable CI entrypoints (package-root cwd)
  unit.sh
  integration.sh
  lint.sh
  ensure_pypi_dspy.sh         # reinstall PyPI dspy wheel in fork monorepo
src/dr_dspy/migration/        # v0 → v1 reshape logic (not inline in tests)
```

## Shared fixtures

Defined in [`tests/conftest.py`](tests/conftest.py):

- **`app_postgres_schema`** — creates an isolated schema, applies v1 migrations +
  append-only triggers, exposes `database_url` with `search_path` set for steps
  that open their own SQLAlchemy engines.
- **`reset_dbos`** — destroys/reconfigures DBOS, resets the system database
  (SQLite file under `tmp_path` by default), and launches the platform runtime.

Seed helpers live in [`tests/support/postgres_fixtures.py`](tests/support/postgres_fixtures.py):

- **`seed_prediction_spec(connection, spec)`** — inserts experiment + spec rows.
- **`start_test_workflow(workflow, workflow_id, *args)`** — DBOS workflow helper.

## Conventions

### Mock boundaries

- **Workflow integration tests:** mock only the LM boundary (`execute_lm_node` or
  provider caller). Do not mock DB steps under test.
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
committed JSON. CI does not require live v0 tables. Refresh samples from a local
database with an ad-hoc script when legacy schema rows change materially.

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
workflows are under test.

## CI

GitHub Actions workflow: [`.github/workflows/dr_dspy_tests.yml`](../.github/workflows/dr_dspy_tests.yml)
(path-scoped to `dr-dspy/**`). While the package still lives inside the DSPy
fork, jobs use `working-directory: dr-dspy`; after
[repo extraction](docs/repo-split-and-naming-plan.md), drop that prefix.

| Job | Script | Notes |
|-----|--------|-------|
| lint | `./scripts/ci/lint.sh` | ruff + ty |
| unit | `./scripts/ci/unit.sh` | 294 unit tests |
| integration | `./scripts/ci/integration.sh` | Postgres 16 service; 27 tests |

Local equivalents (from the package root):

```bash
./scripts/ci/lint.sh
./scripts/ci/unit.sh
DATABASE_URL=postgresql+psycopg:///dr_dspy ./scripts/ci/integration.sh
```

The fork monorepo still vendors DSPy at the workspace root. CI runs
[`scripts/ci/ensure_pypi_dspy.sh`](scripts/ci/ensure_pypi_dspy.sh) to reinstall
`dspy==3.3.0b1` from PyPI and prove the post-extraction dependency cut. Drop
that script after `git filter-repo`.

[`Dockerfile.ci`](Dockerfile.ci) builds a unit-test image for future Depot
wiring after the org repo is created.

Upstream DSPy workflows ([`run_tests.yml`](../.github/workflows/run_tests.yml))
skip when a PR touches only `dr-dspy/**`.

## Changelog

### 2026-06-30 — CI scripts and GitHub workflow

- Added `scripts/ci/{unit,integration,lint,ensure_pypi_dspy}.sh`.
- Added `.github/workflows/dr_dspy_tests.yml` (lint, unit, integration).
- Pinned `dspy==3.3.0b1`; CI reinstalls the PyPI wheel in the fork monorepo.
- Fixed `tests/test_serialization.py` import path.
- Added `Dockerfile.ci` for future Depot wiring.

### 2026-06-30 — Platform integration + v0 migration smoke tiers

- Added tiered integration test model (Tiers 0–3 and 3.5).
- Added `tests/conftest.py` shared fixtures and `@pytest.mark.integration`.
- Added `src/dr_dspy/migration/v0_reshape.py` and frozen v0 JSON fixtures.
- Fixed optional JSONB columns in `persist_generation_result` to insert SQL
  `NULL` instead of JSON `null` for Postgres check constraints.
- Added `TESTING.md` as canonical testing documentation.
