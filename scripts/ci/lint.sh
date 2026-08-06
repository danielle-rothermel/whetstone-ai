#!/usr/bin/env bash
set -euo pipefail

uv run ruff format --check .
uv run ruff check
uv run ruff check --select TID251 tests
uv run ty check
./scripts/ci/defs.sh
