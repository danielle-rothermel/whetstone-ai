from __future__ import annotations

import sys
from pathlib import Path

import pytest

import whetstone.optimization.codex.runner as runner_module
from tests.optimization.codex.support import experiment
from whetstone.envs.factory import EnvExperiment


@pytest.fixture(scope="session")
def codex_experiment() -> EnvExperiment:
    return experiment()


@pytest.fixture
def declared_seatbelt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    sandbox_exec = tmp_path / "sandbox-exec"
    sandbox_exec.touch()
    monkeypatch.setattr(runner_module, "_MACOS_SANDBOX_EXEC", sandbox_exec)
    monkeypatch.setattr(sys, "platform", "darwin")
    return sandbox_exec
