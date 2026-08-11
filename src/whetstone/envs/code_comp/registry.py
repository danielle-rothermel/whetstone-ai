from __future__ import annotations

from enum import UNIQUE, StrEnum, verify
from typing import Any

from whetstone.envs.factory import EnvExperiment

__all__ = [
    "CodeCompMode",
    "build_code_comp_experiment",
    "code_comp_mode_for",
]


@verify(UNIQUE)
class CodeCompMode(StrEnum):
    """HumanEval code-compression experiment modes."""

    DIRECT = "direct"
    ENCDEC = "encdec"
    ENCDEC_MUTANT = "encdec_mutant"


def build_code_comp_experiment(
    mode: CodeCompMode,
    /,
    **kwargs: Any,
) -> EnvExperiment:
    """Build one code-compression experiment for the selected mode."""

    if mode is CodeCompMode.DIRECT:
        from whetstone.envs.code_comp.modes.direct import (
            build_direct_experiment,
        )

        return build_direct_experiment(**kwargs)
    if mode is CodeCompMode.ENCDEC:
        from whetstone.envs.code_comp.modes.encdec import (
            build_encdec_experiment,
        )

        return build_encdec_experiment(**kwargs)
    if mode is CodeCompMode.ENCDEC_MUTANT:
        from whetstone.envs.code_comp.modes.mutant import (
            build_mutant_experiment,
        )

        return build_mutant_experiment(**kwargs)
    raise ValueError(f"unsupported code_comp mode {mode!r}")


def code_comp_mode_for(experiment: EnvExperiment) -> CodeCompMode:
    """Resolve the mode for one built code-compression experiment."""

    from whetstone.envs.code_comp.modes.direct import DirectExperiment
    from whetstone.envs.code_comp.modes.encdec import EncDecExperiment
    from whetstone.envs.code_comp.modes.mutant import MutantExperiment

    if isinstance(experiment, MutantExperiment):
        return CodeCompMode.ENCDEC_MUTANT
    if isinstance(experiment, EncDecExperiment):
        return CodeCompMode.ENCDEC
    if isinstance(experiment, DirectExperiment):
        return CodeCompMode.DIRECT
    raise TypeError(
        "experiment is not a code_comp direct, encdec, or mutant experiment"
    )
