from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

import whetstone.runner.dryrun as dryrun
from whetstone.runner.cell import CellConfig, CellOutcome
from whetstone.runner.optimization_run import Optimizer


@dataclass(frozen=True)
class _DryMarker:
    optimizer: Optimizer


@pytest.mark.parametrize("optimizer", list(Optimizer))
def test_all_optimizers_use_the_same_dry_cell_seam(
    monkeypatch,
    optimizer: Optimizer,
) -> None:
    marker = _DryMarker(optimizer)
    observed: list[_DryMarker] = []

    def fake_run(config):
        observed.append(config)
        return cast(CellOutcome, config)

    monkeypatch.setattr(dryrun, "run_cell", fake_run)

    result = dryrun.run_dry_cell(cast(CellConfig, marker))

    assert result is marker
    assert observed == [marker]
