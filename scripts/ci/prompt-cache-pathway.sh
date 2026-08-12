#!/usr/bin/env bash
set -euo pipefail

uv sync --locked --group dev
uv run pytest tests/pathways/execution/test_prompt_cache_pathway.py -q -m pathway
