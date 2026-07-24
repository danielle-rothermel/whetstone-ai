"""Pilot orchestration over the canonical cell entry point."""

from __future__ import annotations

from dataclasses import dataclass

from whetstone.runner.cell import CellConfig, CellOutcome, run_cell


@dataclass(frozen=True, slots=True)
class PilotReport:
    outcomes: tuple[CellOutcome, ...]

    @property
    def completed(self) -> int:
        return sum(not outcome.skipped for outcome in self.outcomes)

    @property
    def skipped(self) -> int:
        return sum(outcome.skipped for outcome in self.outcomes)


def run_pilot(configs: tuple[CellConfig, ...]) -> PilotReport:
    """Run a bounded pilot through the exact live cell seam."""
    return PilotReport(tuple(run_cell(config) for config in configs))


__all__ = ["PilotReport", "run_pilot"]
