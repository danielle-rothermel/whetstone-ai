#!/usr/bin/env bash
set -euo pipefail

: "${DATABASE_URL:=postgresql+psycopg://postgres:postgres@localhost:5432/dr_dspy_test}"
export DATABASE_URL

uv run pytest -m integration tests/integration/ -q
