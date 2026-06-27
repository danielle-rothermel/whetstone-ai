from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

EVAL_DBOS_HARNESS_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "humaneval_dspy_eval_only_dbos_v0.py"
)


def _load_script_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def eval_dbos_harness() -> ModuleType:
    return _load_script_module(
        "humaneval_dspy_eval_only_dbos_v0",
        EVAL_DBOS_HARNESS_PATH,
    )
