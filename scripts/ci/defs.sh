#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
cd -- "${repo_root}"

if (( $# > 1 )) || (( $# == 1 )) && [[ "$1" != "--structural-only" ]]; then
  echo "usage: $0 [--structural-only]" >&2
  exit 2
fi

uvx tombi@1.2.5 lint --offline .defs/terms.toml .defs/contracts.toml
if (( $# == 1 )); then
  uv run python scripts/ci/check_defs.py "$1"
else
  uv run python scripts/ci/check_defs.py
fi
