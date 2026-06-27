from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def encdec_configured() -> None:
    """Configure the enc-dec experiment by loading its script module."""
    script = (
        REPO_ROOT / "scripts" / "humaneval_dspy_eval_only_encdec_dbos_v0.py"
    )
    spec = importlib.util.spec_from_file_location("encdec_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
