#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
cd -- "${repo_root}"

uv sync --locked --group dev

"${script_dir}/runner-cell-pathway.sh"
"${script_dir}/evaluation-restart-pathway.sh"
"${script_dir}/fanout-containment-pathway.sh"
"${script_dir}/prompt-cache-pathway.sh"
"${script_dir}/preview-anchor-pathway.sh"
"${script_dir}/sqlite-time-pathway.sh"
"${script_dir}/sqlite-contention-pathway.sh"

if [[ -n "${WHETSTONE_TEST_POSTGRES_DSN:-}" ]]; then
  "${script_dir}/gepa-dbos-pathway.sh"
else
  echo "Skipping GEPA DBOS pathway (WHETSTONE_TEST_POSTGRES_DSN unset)"
fi
