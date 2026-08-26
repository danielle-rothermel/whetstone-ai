#!/usr/bin/env bash

set -euo pipefail

uv run ruff check .
uv run pytest
uv run pytest -m integration tests/integration/
