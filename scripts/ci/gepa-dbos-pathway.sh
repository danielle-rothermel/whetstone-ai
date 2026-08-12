#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${WHETSTONE_TEST_POSTGRES_DSN:-}" ]]; then
  echo "WHETSTONE_TEST_POSTGRES_DSN is required for GEPA DBOS pathway tests" >&2
  exit 1
fi

uv sync --locked --group dev
uv run pytest tests/pathways/gepa/ -q -m pathway
