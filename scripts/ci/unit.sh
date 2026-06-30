#!/usr/bin/env bash
set -euo pipefail

uv run pytest tests/ -m "not integration" -q
