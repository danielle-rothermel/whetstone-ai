#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
tmp_parent="$(cd -- "${TMPDIR:-/tmp}" && pwd -P)"
created_tmp_root="$(
  mktemp -d "${tmp_parent}/whetstone-wheel-smoke.XXXXXX"
)"
validated_tmp_root="$(cd -- "${created_tmp_root:?}" && pwd -P)"
validated_tmp_parent="$(dirname -- "${validated_tmp_root:?}")"

if [[ ! -d "${validated_tmp_root}" || -L "${created_tmp_root}" ]]; then
  echo "wheel smoke temporary root is not a real directory" >&2
  exit 1
fi
if [[ "${validated_tmp_parent}" != "${tmp_parent}" ]]; then
  echo "wheel smoke temporary root escaped its parent" >&2
  exit 1
fi
if [[ "$(basename -- "${validated_tmp_root}")" != whetstone-wheel-smoke.* ]]; then
  echo "wheel smoke temporary root has an unexpected name" >&2
  exit 1
fi

cleanup() {
  local cleanup_name
  local cleanup_parent
  local cleanup_target="${validated_tmp_root:?}"

  if [[ ! -d "${cleanup_target}" || -L "${cleanup_target}" ]]; then
    echo "refusing to clean an invalid wheel smoke temporary root" >&2
    return 1
  fi
  cleanup_parent="$(dirname -- "${cleanup_target}")"
  cleanup_name="$(basename -- "${cleanup_target}")"
  if [[ "${cleanup_parent}" != "${validated_tmp_parent}" ]]; then
    echo "refusing to clean a temporary root outside its parent" >&2
    return 1
  fi
  if [[ "${cleanup_name}" != whetstone-wheel-smoke.* ]]; then
    echo "refusing to clean an unexpectedly named temporary root" >&2
    return 1
  fi
  rm -rf -- "${cleanup_target:?}"
}
trap cleanup EXIT

wheel_dir="${validated_tmp_root}/wheel"
venv_dir="${validated_tmp_root}/venv"
smoke_dir="${validated_tmp_root}/smoke"
mkdir -p -- "${wheel_dir}" "${smoke_dir}"

uv build --wheel --out-dir "${wheel_dir}" "${repo_root}"

shopt -s nullglob
wheels=("${wheel_dir}"/*.whl)
shopt -u nullglob
if (( ${#wheels[@]} != 1 )); then
  echo "expected exactly one built wheel, found ${#wheels[@]}" >&2
  exit 1
fi

uv venv --python 3.13 "${venv_dir}"
venv_python="${venv_dir}/bin/python"
"${venv_python}" -c \
  'import sys; assert sys.version_info[:2] == (3, 13), sys.version'

VIRTUAL_ENV="${venv_dir}" uv sync \
  --project "${repo_root}" \
  --active \
  --locked \
  --no-dev \
  --no-install-project
uv pip install --python "${venv_python}" --no-deps "${wheels[0]}"
uv pip check --python "${venv_python}"

(
  cd -- "${smoke_dir}"
  unset PYTHONPATH
  "${venv_python}" -I \
    "${repo_root}/scripts/ci/installed_package_smoke.py" \
    "${repo_root}"
)
