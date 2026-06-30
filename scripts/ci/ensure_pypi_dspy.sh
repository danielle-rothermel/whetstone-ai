#!/usr/bin/env bash
set -euo pipefail

# In the fork monorepo, dr-dspy shares a workspace with vendored dspy. Pin
# dspy==3.3.0b1 in pyproject.toml; CI reinstalls the PyPI wheel to prove the
# post-extraction dependency cut. After filter-repo, drop this reinstall step.
uv pip install --reinstall "dspy==3.3.0b1"
