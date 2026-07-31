#!/usr/bin/env bash
set -euo pipefail

uv sync --locked --group dev
uv run pytest tests/ -q -m "not process_integration"
