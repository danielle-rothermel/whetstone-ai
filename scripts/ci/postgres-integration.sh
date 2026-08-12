#!/usr/bin/env bash
set -euo pipefail

uv sync --locked --group dev
uv run pytest \
  tests/coordination/test_proposal_provider.py \
  tests/core/effects/test_postgres.py \
  tests/optimization/tools/test_conformance.py \
  tests/optimization/tools/test_postgres.py \
  -q -m "postgres_integration"
exec "$(dirname "$0")/gepa-dbos-pathway.sh"
