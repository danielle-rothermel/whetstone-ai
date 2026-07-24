from __future__ import annotations

import os

import pytest

from whetstone.runner.cell import run_cell
from whetstone.runner.cli import _load_factory
from whetstone.runner.optimization_run import Optimizer


@pytest.mark.skipif(
    os.environ.get("WHETSTONE_LIVE_CODEX") != "1",
    reason="set WHETSTONE_LIVE_CODEX=1 for the paid Codex smoke",
)
def test_live_codex_cell_is_explicit_and_separate() -> None:
    factory_path = os.environ.get("WHETSTONE_LIVE_CELL_FACTORY")
    if not factory_path:
        raise ValueError(
            "WHETSTONE_LIVE_CELL_FACTORY must name a typed Codex cell factory"
        )
    config = _load_factory(factory_path)()
    if config.optimization.optimizer is not Optimizer.CODEX:
        raise ValueError("live Codex smoke factory returned another optimizer")

    outcome = run_cell(config)

    assert outcome.record.optimizer == Optimizer.CODEX.value
