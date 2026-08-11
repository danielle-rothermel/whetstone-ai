#!/usr/bin/env bash
set -euo pipefail

uv sync --locked --group dev
uv run pytest tests/pathways/sqlite_time/ -q -m pathway
uv run pytest tests/core/effects/test_sqlite.py -q \
  -k test_sqlite_heartbeat_keeps_real_time_work_publishable
