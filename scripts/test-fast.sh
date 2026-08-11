#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run pytest --ignore=tests/pathways -m 'not slow' "$@"
