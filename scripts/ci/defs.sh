#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
cd -- "${repo_root}"

uvx tombi@1.2.5 lint --offline .defs/terms.toml .defs/contracts.toml
uv run python scripts/ci/check_defs.py
