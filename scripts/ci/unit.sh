#!/usr/bin/env bash
set -euo pipefail

uv sync --group dev
uv run pytest tests/ -q -n 4 --dist loadgroup \
  --ignore tests/orchestration/test_concurrency.py \
  --ignore tests/orchestration/test_retry_gate.py \
  --ignore tests/orchestration/test_executor_dbos.py
