#!/usr/bin/env bash
set -euo pipefail

: "${DR_PLATFORM_TEST_DATABASE_URL:?DBOS tests require an explicit Postgres URL}"
case "$DR_PLATFORM_TEST_DATABASE_URL" in
  postgresql://*|postgresql+psycopg://*) ;;
  *)
    echo "DR_PLATFORM_TEST_DATABASE_URL must be a Postgres URL" >&2
    exit 2
    ;;
esac
export WHETSTONE_REQUIRE_POSTGRES_TESTS=1

uv sync --group dev
uv run pytest \
  tests/orchestration/test_concurrency.py \
  tests/orchestration/test_executor_dbos.py \
  tests/orchestration/test_retry_gate.py \
  -q
