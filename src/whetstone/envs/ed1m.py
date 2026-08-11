"""Legacy import path.

Implementation lives in whetstone.envs.code_comp.
"""

from whetstone.envs.code_comp.modes.mutant import (
    ED1M_CANONICAL_MODEL,
    ED1M_ENV_NAME,
    ED1M_FIDELITY_NAME,
    MUTANT_ENV_NAME,
    MutantExperiment,
    build_ed1m_procedure_config,
    build_ed1m_reward_policy,
    build_mutant_experiment,
    score_ed1m_row,
)

Ed1mExperiment = MutantExperiment
build_ed1m_experiment = build_mutant_experiment

__all__ = [
    "ED1M_CANONICAL_MODEL",
    "ED1M_ENV_NAME",
    "ED1M_FIDELITY_NAME",
    "MUTANT_ENV_NAME",
    "Ed1mExperiment",
    "MutantExperiment",
    "build_ed1m_experiment",
    "build_ed1m_procedure_config",
    "build_ed1m_reward_policy",
    "build_mutant_experiment",
    "score_ed1m_row",
]
