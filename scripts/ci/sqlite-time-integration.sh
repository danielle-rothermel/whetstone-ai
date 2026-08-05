#!/usr/bin/env bash
set -euo pipefail

uv sync --locked --group dev
uv run pytest tests/ -q -m "sqlite_time_integration"
