#!/usr/bin/env bash
set -euo pipefail

exec "$(dirname "$0")/sqlite-contention-pathway.sh" "$@"
