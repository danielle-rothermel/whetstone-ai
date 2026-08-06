#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"
cd -- "${repo_root}"

uv sync --locked --group dev
./scripts/ci/lint.sh --structural-only
# Specific tests own this fast gate through the precheck marker; exhaustive
# and integration coverage remains in the CI lanes.
uv run pytest -q -n auto \
  -m "precheck and not process_integration and not postgres_integration and not sqlite_time_integration and not sqlite_contention"
