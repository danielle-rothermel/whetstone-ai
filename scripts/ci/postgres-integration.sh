#!/usr/bin/env bash
set -euo pipefail

uv sync --locked --group dev
uv run pytest \
  tests/coordination/test_proposal_provider.py \
  tests/core/effects/test_postgres.py \
  tests/optimization/gepa/test_effect_runtime.py \
  tests/optimization/gepa/test_runner.py \
  tests/optimization/tools/test_conformance.py \
  tests/optimization/tools/test_postgres.py \
  -q -m "postgres_integration"
