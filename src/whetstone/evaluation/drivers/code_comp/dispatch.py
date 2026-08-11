from __future__ import annotations

from typing import Any

from whetstone.envs.code_comp.modes.direct import D1Experiment
from whetstone.envs.code_comp.modes.encdec import Ed1Experiment
from whetstone.envs.factory import EnvExperiment

__all__ = ["run_code_comp_eval"]


def run_code_comp_eval(experiment: EnvExperiment, /, **kwargs: Any) -> object:
    """Run one code-compression evaluation for the experiment's mode."""

    if isinstance(experiment, D1Experiment):
        from whetstone.evaluation.drivers.code_comp.direct import run_d1_eval

        return run_d1_eval(experiment, **kwargs)
    if isinstance(experiment, Ed1Experiment):
        from whetstone.evaluation.drivers.code_comp.encdec import run_ed1_eval

        return run_ed1_eval(experiment, **kwargs)
    raise TypeError(
        "experiment is not a code_comp direct or encdec/mutant experiment"
    )
