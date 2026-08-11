#!/usr/bin/env bash
set -euo pipefail

uv sync --locked --group dev
uv run pytest tests/pathways/sqlite_contention/ -q -m pathway
uv run pytest tests/optimization/tools/test_sqlite.py -q \
  -k test_spawned_sqlite_capacity_race_is_atomic
