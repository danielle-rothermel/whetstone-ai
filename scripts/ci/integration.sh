#!/usr/bin/env bash
set -euo pipefail

uv sync --group dev

# Integration never touches the configured long-lived database. It uses that
# connection only to create a uniquely named disposable database.
database_name="whetstone_integration_${RANDOM}_${RANDOM}_$$"
lifecycle_args=()
application_database_url="postgresql+psycopg:///$database_name"
if [[ -n "${DATABASE_URL:-}" ]]; then
  lifecycle_args=("--maintenance-db=${DATABASE_URL/postgresql+psycopg:/postgresql:}")
  application_database_url="${DATABASE_URL%/*}/$database_name"
  application_database_url="${application_database_url/postgresql:/postgresql+psycopg:}"
fi
cleanup() {
  dropdb "${lifecycle_args[@]}" --if-exists "$database_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
createdb "${lifecycle_args[@]}" "$database_name"
export DATABASE_URL="$application_database_url"
uv run alembic upgrade head

collected="$(uv run pytest --collect-only -m integration tests/integration/ -q)"
if ! grep -Eq '[1-9][0-9]* test(s)? collected' <<<"$collected"; then
  printf '%s\n' "$collected" >&2
  echo "integration suite collected zero tests" >&2
  exit 1
fi
uv run pytest -m integration tests/integration/ -q
