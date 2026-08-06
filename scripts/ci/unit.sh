#!/usr/bin/env bash
set -euo pipefail

uv sync --locked --group dev
uv run pytest tests/ -q -n auto \
  -m "not process_integration and not postgres_integration and not sqlite_time_integration and not sqlite_contention"
