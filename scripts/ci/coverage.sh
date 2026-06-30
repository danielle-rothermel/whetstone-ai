#!/usr/bin/env bash
set -euo pipefail

: "${DATABASE_URL:=postgresql+psycopg://postgres:postgres@localhost:5432/dr_dspy_test}"
export DATABASE_URL

uv sync --group dev
uv run pytest tests/ -m "not integration" \
  --cov=dr_dspy \
  --cov-report=term-missing \
  --cov-fail-under=88

uv run pytest -m integration tests/integration/ \
  --cov=dr_dspy \
  --cov-append \
  --cov-report=term-missing \
  --cov-fail-under=88
