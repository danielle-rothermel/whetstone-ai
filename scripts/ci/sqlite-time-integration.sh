#!/usr/bin/env bash
set -euo pipefail

uv sync --locked --group dev
uv run pytest \
  tests/coordination/test_evaluation_claims.py \
  tests/core/effects/test_authority.py \
  tests/core/effects/test_sqlite.py \
  tests/optimization/test_harness_proposal.py \
  tests/optimization/tools/test_refusals.py \
  -q -m "sqlite_time_integration and not sqlite_contention"
