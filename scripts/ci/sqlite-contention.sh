#!/usr/bin/env bash
set -euo pipefail

uv sync --locked --group dev
uv run pytest \
  tests/core/effects/test_sqlite.py \
  tests/optimization/test_sqlite_contention_fork.py \
  tests/optimization/tools/test_sqlite.py \
  -q -m "sqlite_contention"
