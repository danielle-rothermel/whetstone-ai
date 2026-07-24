"""ED1-family pilot selection over canonical cells."""

from whetstone.runner.cell import CellConfig
from whetstone.runner.pilot import PilotReport, run_pilot


def run_ed1_pilot(configs: tuple[CellConfig, ...]) -> PilotReport:
    if any(config.env not in {"ed1", "ed1m"} for config in configs):
        raise ValueError("ED1 pilot accepts only ed1/ed1m cells")
    return run_pilot(configs)


__all__ = ["run_ed1_pilot"]
