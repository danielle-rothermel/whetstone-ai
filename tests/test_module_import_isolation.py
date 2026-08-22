"""Every public entrypoint must import as the first whetstone import.

A module that only imports successfully after some other whetstone module
has been imported makes import order load-bearing for consumers. Each
case runs in a fresh subprocess so no earlier import can mask a cycle.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

#: Public entrypoints consumers import directly. Each must stand alone.
FIRST_IMPORT_MODULES = (
    "whetstone.coordination.runtime_bootstrap",
    "whetstone.eval.drivers",
    "whetstone.experiment.binding",
    "whetstone.optim.codex",
    "whetstone.optim.copro.control",
    "whetstone.optim.gepa.control",
    "whetstone.optim.miprov2.control",
    "whetstone.platform.cli",
    "whetstone.provider.llm_call",
)


@pytest.mark.parametrize("module", FIRST_IMPORT_MODULES)
def test_module_imports_as_the_first_whetstone_import(module: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"{module} must import as the first whetstone import; "
        f"got:\n{completed.stderr}"
    )
