#!/usr/bin/env bash
set -euo pipefail

uv sync --locked --group dev
uv run pytest \
  tests/coordination/test_evaluation_claims.py \
  tests/coordination/test_evaluation_service.py \
  tests/coordination/test_proposal_provider.py \
  tests/core/effects/test_postgres.py \
  tests/core/effects/test_sqlite.py \
  tests/evaluation/drivers/test_internal.py \
  tests/evaluation/test_engine.py \
  tests/execution/test_fanout.py \
  tests/execution/test_partials.py \
  tests/execution/test_prompt_cache.py \
  tests/optimization/gepa/test_upstream_oracle.py \
  tests/optimization/tools/test_contracts.py \
  tests/optimization/tools/test_evaluator.py \
  tests/optimization/tools/test_facade.py \
  tests/optimization/tools/test_postgres.py \
  tests/optimization/tools/test_sqlite.py \
  -q \
  -m "process_integration and not postgres_integration and not sqlite_time_integration and not sqlite_contention"
