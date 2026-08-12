from __future__ import annotations

from typing import Any

from whetstone.envs.code_comp.config import (
    CodeCompExperimentConfig,
    default_code_comp_config,
)
from whetstone.envs.code_comp.constants import CODE_COMP_ENV_NAME
from whetstone.envs.code_comp.experiment import (
    CodeCompExperiment,
    DirectExperiment,
    EncDecExperiment,
    MutantExperiment,
)
from whetstone.envs.code_comp.mode import (
    CodeCompMode,
    code_comp_identity_prefix,
)
from whetstone.envs.factory import EnvExperiment

__all__ = [
    "CODE_COMP_ENV_NAME",
    "CodeCompMode",
    "build_code_comp_experiment",
    "build_code_comp_experiment_from_mode",
    "code_comp_identity_prefix",
    "code_comp_mode_for",
]


def build_code_comp_experiment(
    first: CodeCompExperimentConfig | CodeCompMode,
    /,
    **kwargs: Any,
) -> CodeCompExperiment:
    """Build one code-compression experiment from config or legacy kwargs."""
    if isinstance(first, CodeCompExperimentConfig):
        if kwargs:
            raise TypeError(
                "build_code_comp_experiment(config) accepts no extra kwargs"
            )
        return first.build_experiment()
    return build_code_comp_experiment_from_mode(first, **kwargs)


def build_code_comp_experiment_from_mode(
    mode: CodeCompMode,
    /,
    **kwargs: Any,
) -> CodeCompExperiment:
    """Legacy kwargs entrypoint; prefer the config-first builder."""
    scorer = kwargs.pop("scorer", None)
    return default_code_comp_config(mode, **kwargs).build_experiment(
        scorer=scorer
    )


def code_comp_mode_for(experiment: EnvExperiment) -> CodeCompMode:
    """Resolve the mode for one built code-compression experiment."""
    if isinstance(experiment, MutantExperiment):
        return CodeCompMode.ENCDEC_MUTANT
    if isinstance(experiment, EncDecExperiment):
        return CodeCompMode.ENCDEC
    if isinstance(experiment, DirectExperiment):
        return CodeCompMode.DIRECT
    raise TypeError(
        "experiment is not a code_comp direct, encdec, or mutant experiment"
    )
