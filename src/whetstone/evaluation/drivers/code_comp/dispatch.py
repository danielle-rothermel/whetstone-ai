from __future__ import annotations

from typing import Any

from whetstone.envs.code_comp.modes.direct import DirectExperiment
from whetstone.envs.code_comp.modes.encdec import EncDecExperiment
from whetstone.envs.code_comp.registry import CodeCompMode, code_comp_mode_for
from whetstone.envs.factory import EnvExperiment
from whetstone.evaluation.drivers.code_comp.direct import DirectEvalResult
from whetstone.evaluation.drivers.code_comp.encdec import EncDecEvalResult

__all__ = ["run_code_comp_eval"]


def run_code_comp_eval(
    experiment: EnvExperiment, /, **kwargs: Any
) -> DirectEvalResult | EncDecEvalResult:
    """Run one code-compression evaluation for the experiment's mode."""

    mode = code_comp_mode_for(experiment)
    if mode is CodeCompMode.DIRECT:
        from whetstone.evaluation.drivers.code_comp.direct import (
            run_direct_eval,
        )

        assert isinstance(experiment, DirectExperiment)
        return run_direct_eval(experiment, **kwargs)
    if mode in {CodeCompMode.ENCDEC, CodeCompMode.ENCDEC_MUTANT}:
        from whetstone.evaluation.drivers.code_comp.encdec import (
            run_encdec_eval,
        )

        assert isinstance(experiment, EncDecExperiment)
        return run_encdec_eval(experiment, **kwargs)
    raise TypeError(
        "experiment is not a code_comp direct or encdec/mutant experiment"
    )
