#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"

readonly postgres_socket_dir="${WHETSTONE_LOCAL_POSTGRES_SOCKET_DIR:-/tmp}"
readonly postgres_host="127.0.0.1"
readonly postgres_port="${WHETSTONE_LOCAL_POSTGRES_PORT:-5432}"
readonly postgres_admin_user="${WHETSTONE_LOCAL_POSTGRES_ADMIN_USER:-$(id -un)}"
readonly postgres_admin_database="postgres"
readonly test_role="whetstone_test"
readonly test_password="whetstone_test"
readonly test_database_prefix="whetstone_test_"

if [[ ! "${postgres_socket_dir}" = /* ]]; then
  echo "local PostgreSQL socket directory must be an absolute path" >&2
  exit 1
fi
if [[ ! "${postgres_port}" =~ ^[0-9]+$ ]]; then
  echo "local PostgreSQL port must be numeric" >&2
  exit 1
fi
if [[ -z "${postgres_admin_user}" ]]; then
  echo "local PostgreSQL administrator must be non-empty" >&2
  exit 1
fi

for command_name in psql uuidgen; do
  if ! command -v "${command_name}" >/dev/null; then
    echo "required command is unavailable: ${command_name}" >&2
    exit 1
  fi
done

admin_psql=(
  psql
  --no-psqlrc
  --host="${postgres_socket_dir}"
  --port="${postgres_port}"
  --username="${postgres_admin_user}"
  --dbname="${postgres_admin_database}"
  --set=ON_ERROR_STOP=1
)

server_version="$("${admin_psql[@]}" --tuples-only --no-align \
  --command="SHOW server_version_num")"
if [[ ! "${server_version}" =~ ^17[0-9]{4}$ ]]; then
  echo "local PostgreSQL must match CI major version 17" >&2
  exit 1
fi

listen_addresses="$("${admin_psql[@]}" --tuples-only --no-align \
  --command="SHOW listen_addresses")"
IFS=',' read -r -a configured_addresses <<< "${listen_addresses}"
for configured_address in "${configured_addresses[@]}"; do
  configured_address="${configured_address//[[:space:]]/}"
  case "${configured_address}" in
    "" | localhost | 127.0.0.1 | ::1) ;;
    *)
      echo "refusing to provision test credentials on a non-loopback server" >&2
      exit 1
      ;;
  esac
done

role_state="$("${admin_psql[@]}" --tuples-only --no-align --command="
  SELECT concat_ws('|', rolsuper, rolcreatedb, rolcreaterole, rolcanlogin)
  FROM pg_catalog.pg_roles
  WHERE rolname = '${test_role}'
")"
case "${role_state}" in
  "")
    "${admin_psql[@]}" --command="
      CREATE ROLE ${test_role}
      LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION
      PASSWORD '${test_password}'
    " >/dev/null
    ;;
  f\|f\|f\|t) ;;
  *)
    echo "refusing to use ${test_role}: role privileges are unexpected" >&2
    exit 1
    ;;
esac

"${admin_psql[@]}" --command="
  ALTER ROLE ${test_role} PASSWORD '${test_password}'
" >/dev/null

database_suffix="$(uuidgen | tr -d '-' | tr 'A-F' 'a-f')"
test_database="${test_database_prefix}${database_suffix}"
readonly test_database

if [[ ! "${test_database}" =~ ^whetstone_test_[0-9a-f]{32}$ ]]; then
  echo "generated test database name is invalid" >&2
  exit 1
fi

existing_database="$(
  "${admin_psql[@]}" --tuples-only --no-align \
    --set=test_database="${test_database}" <<'SQL'
SELECT datname
FROM pg_catalog.pg_database
WHERE datname = :'test_database';
SQL
)"
if [[ -n "${existing_database}" ]]; then
  echo "refusing to overwrite existing database ${test_database}" >&2
  exit 1
fi

database_created=false
cleanup() {
  local run_status=$?
  local database_owner

  if [[ "${database_created}" != true ]]; then
    return "${run_status}"
  fi
  if [[ ! "${test_database}" =~ ^whetstone_test_[0-9a-f]{32}$ ]]; then
    echo "refusing to drop an unexpectedly named database" >&2
    return 1
  fi

  database_owner="$(
    "${admin_psql[@]}" --tuples-only --no-align \
      --set=test_database="${test_database}" <<'SQL'
SELECT pg_catalog.pg_get_userbyid(datdba)
FROM pg_catalog.pg_database
WHERE datname = :'test_database';
SQL
  )"
  if [[ "${database_owner}" != "${test_role}" ]]; then
    echo "refusing to drop ${test_database}: owner is unexpected" >&2
    return 1
  fi

  if ! "${admin_psql[@]}" --set=test_database="${test_database}" \
    >/dev/null <<'SQL'
DROP DATABASE :"test_database";
SQL
  then
    echo "failed to drop ${test_database}; it remains for inspection" >&2
    return 1
  fi
  echo "Removed disposable PostgreSQL database ${test_database}"
  return "${run_status}"
}
trap cleanup EXIT

"${admin_psql[@]}" \
  --set=test_database="${test_database}" \
  --set=test_role="${test_role}" \
  >/dev/null <<'SQL'
CREATE DATABASE :"test_database" OWNER :"test_role";
SQL
database_created=true

export WHETSTONE_TEST_POSTGRES_DSN="postgresql://${test_role}:${test_password}@${postgres_host}:${postgres_port}/${test_database}"

connection_identity="$(psql --no-psqlrc \
  --dbname="${WHETSTONE_TEST_POSTGRES_DSN}" \
  --set=ON_ERROR_STOP=1 \
  --tuples-only --no-align \
  --command="SELECT current_database() || '|' || current_user")"
if [[ "${connection_identity}" != "${test_database}|${test_role}" ]]; then
  echo "test connection reached an unexpected database or role" >&2
  exit 1
fi

echo "Running PostgreSQL integration tests in ${test_database}"
(
  cd -- "${repo_root}"
  ./scripts/ci/postgres-integration.sh
)
