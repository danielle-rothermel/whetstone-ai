#!/usr/bin/env bash
set -euo pipefail

uv sync --group dev

# Integration never touches the developer's long-lived database.  The local
# peer-auth cluster is used only to create a uniquely named disposable DB.
database_name="whetstone_integration_${RANDOM}_${RANDOM}_$$"
cleanup() {
  dropdb --if-exists "$database_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
createdb "$database_name"
export DATABASE_URL="postgresql+psycopg:///$database_name"
uv run alembic upgrade head

collected="$(uv run pytest --collect-only -m integration tests/integration/ -q)"
if ! grep -Eq '[1-9][0-9]* test(s)? collected' <<<"$collected"; then
  printf '%s\n' "$collected" >&2
  echo "integration suite collected zero tests" >&2
  exit 1
fi
uv run pytest -m integration tests/integration/ -q
