#!/usr/bin/env bash
set -euo pipefail

# The slow tier: tests deselected from the default fast run (see the `slow`
# marker + addopts in pyproject.toml). Currently the full-N c18h PrOntoQA
# pool regeneration that pins the committed pool shape. Runs as its own CI
# job so deselected-by-default never means never-run.
uv sync --group dev
uv run pytest tests/ -m slow -q
