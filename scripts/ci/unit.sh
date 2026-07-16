#!/usr/bin/env bash
set -euo pipefail

uv sync --group dev
uv run pytest tests/ -q
