from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "src" / "whetstone"


@pytest.mark.parametrize(
    "relative",
    [
        "runner/optimizers.py",
        "runner/eval_run.py",
        "runner/execution_mode.py",
        "runner/power.py",
        "runner/statistics.py",
    ],
)
def test_removed_runner_modules_do_not_exist(relative: str) -> None:
    assert not (SOURCE / relative).exists()


def test_removed_symbols_and_memory_backend_are_absent_from_source() -> None:
    source = "\n".join(
        path.read_text() for path in sorted(SOURCE.rglob("*.py"))
    )
    forbidden = (
        "whetstone.runner.optimizers",
        "whetstone.runner.eval_run",
        "whetstone.runner.execution_mode",
        "from whetstone.runner.power import",
        "from whetstone.runner.statistics import",
        "run_optimize",
        "CodexProposerTransport",
        "stub_evaluator",
        "MemoryBackend",
    )
    for value in forbidden:
        assert value not in source


def test_environments_never_import_runner() -> None:
    for path in (SOURCE / "envs").rglob("*.py"):
        assert "whetstone.runner" not in path.read_text()
