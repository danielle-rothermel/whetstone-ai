#!/usr/bin/env bash
set -euo pipefail

uv sync --locked --group dev
uv run pytest tests/pathways/execution/test_fanout_containment_pathway.py -q -m pathway
