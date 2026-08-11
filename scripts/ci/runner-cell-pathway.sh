#!/usr/bin/env bash
set -euo pipefail

uv sync --locked --group dev
uv run pytest tests/pathways/runner/ -q -m pathway
