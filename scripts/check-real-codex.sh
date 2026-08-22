#!/usr/bin/env bash
#
# Run the real-Codex ladder against a live Codex CLI session.
#
# This is a MANUAL check. It drives the real Codex CLI and spends real
# Codex agent turns on a logged-in subscription session, so it is never
# part of `mise run check` and never runs automatically in CI.
#
# The task model stays fake throughout: evaluations run on the reference
# transport, so the ladder spends Codex turns and no eval-provider credit.
# No OPENAI_API_KEY is set or required, and nothing here reads credential
# material -- the runner's own staging copies ~/.codex/auth.json into each
# run's scratch CODEX_HOME.
#
# Usage:
#   scripts/check-real-codex.sh                 # whole ladder, stop at first break
#   scripts/check-real-codex.sh -k rung3        # one rung
#   WHETSTONE_REAL_CODEX_BINARY=/path/to/codex scripts/check-real-codex.sh
#
# Prerequisites: macOS (the sandbox is sandbox-exec only), the Codex CLI on
# PATH or at /opt/homebrew/bin/codex, and a logged-in session
# (`codex login`).

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Run outputs live outside the repository, one directory per invocation.
timestamp="$(date +%Y%m%d-%H%M%S)"
output_root="${WHETSTONE_REAL_CODEX_OUTPUT_DIR:-$HOME/drotherm/data/whetstone-ai/real-codex}"
output_dir="$output_root/$timestamp"
mkdir -p "$output_dir"

log="$output_dir/pytest.log"
report="$output_dir/rungs.txt"

echo "real-Codex ladder"
echo "  repo:    $repo_root"
echo "  output:  $output_dir"
echo "  codex:   ${WHETSTONE_REAL_CODEX_BINARY:-/opt/homebrew/bin/codex}"
echo

# -x: the ladder is ordered by cost and by what each rung presupposes, so a
# broken lower rung makes every higher one uninterpretable.
set +e
WHETSTONE_REAL_CODEX=1 uv run --extra platform pytest \
    tests/real_codex/test_real_codex_ladder.py \
    -m real_codex \
    -x -v -p no:cacheprovider \
    "$@" 2>&1 | tee "$log"
status="${PIPESTATUS[0]}"
set -e

# One line per rung, in ladder order, from pytest's own verbose output.
{
    echo "real-Codex ladder — $timestamp"
    echo "codex version: $("${WHETSTONE_REAL_CODEX_BINARY:-/opt/homebrew/bin/codex}" --version 2>/dev/null || echo unknown)"
    echo "commit:        $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo
    printf '%-8s %-6s %s\n' RUNG RESULT TEST
    grep -oE 'test_rung[0-9a-c]+[a-z0-9_]*(\[[^]]*\])? (PASSED|FAILED|ERROR|SKIPPED)' "$log" \
        | sed -E 's/^test_(rung[0-9a-c]+)([a-z0-9_]*(\[[^]]*\])?) (.*)$/\1|\4|test_\1\2/' \
        | awk -F'|' '{printf "%-8s %-6s %s\n", $1, $2, $3}' \
        || echo "(no rung results parsed; see pytest.log)"
    echo
    if [ "$status" -eq 0 ]; then
        echo "RESULT: all rungs passed"
    else
        echo "RESULT: ladder stopped (exit $status) — see pytest.log"
    fi
} | tee "$report"

echo
echo "transcript + rung table: $output_dir"
exit "$status"
