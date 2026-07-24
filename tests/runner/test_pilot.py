from __future__ import annotations

from typing import cast

import whetstone.runner.ed1_pilot as ed1_pilot
import whetstone.runner.pilot as pilot
from whetstone.runner.cell import CellConfig, CellOutcome


def test_pilot_delegates_every_cell_to_run_cell(monkeypatch) -> None:
    configs = (cast(CellConfig, object()), cast(CellConfig, object()))
    outcomes = (
        cast(CellOutcome, object()),
        cast(CellOutcome, object()),
    )
    remaining = iter(outcomes)
    observed: list[CellConfig] = []

    def fake_run(config):
        observed.append(config)
        return next(remaining)

    monkeypatch.setattr(pilot, "run_cell", fake_run)

    report = pilot.run_pilot(configs)

    assert report.outcomes == outcomes
    assert observed == list(configs)


def test_ed1_pilot_rejects_other_environments() -> None:
    class Config:
        env = "c18"

    try:
        ed1_pilot.run_ed1_pilot((cast(CellConfig, Config()),))
    except ValueError as exc:
        assert "ed1/ed1m" in str(exc)
    else:
        raise AssertionError("non-ED1 pilot was accepted")
