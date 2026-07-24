"""Dry execution delegates to the same typed cell seam as live execution."""

from whetstone.runner.cell import CellConfig, CellOutcome, run_cell


def run_dry_cell(config: CellConfig) -> CellOutcome:
    """Run a cell whose injected provider/process boundaries are scripted."""
    return run_cell(config)


__all__ = ["run_dry_cell"]
