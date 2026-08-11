"""Legacy import path.

Implementation lives in whetstone.envs.code_comp.
"""

from whetstone.envs.code_comp.modes.direct import (
    D1_CANONICAL_MODEL,
    D1_INPUT_ARMS,
    D1_RENAMED_ARM,
    D1_SUBMISSION_SCORE_NAME,
    D1_WRAPPER_BODY_CEILING,
    D1_WRAPPER_BODY_NAIVE,
    D1RolloutDefinition,
    DirectExperiment,
    build_d1_procedure_config,
    build_d1_reward_policy,
    build_d1_rollout_definition,
    build_direct_experiment,
    d1_ceiling_candidate,
    d1_graph_definition,
    d1_initial_candidate,
)
from whetstone.envs.code_comp.rollout.direct import (
    D1_DEFAULT_RENAME_TOKEN,
    D1_ENV_NAME,
    D1_WRAPPER_FRAME,
    DIRECT_ENV_NAME,
    render_d1_frame,
)

D1Experiment = DirectExperiment
build_d1_experiment = build_direct_experiment

__all__ = [
    "D1_CANONICAL_MODEL",
    "D1_DEFAULT_RENAME_TOKEN",
    "D1_ENV_NAME",
    "D1_INPUT_ARMS",
    "D1_RENAMED_ARM",
    "D1_SUBMISSION_SCORE_NAME",
    "D1_WRAPPER_BODY_CEILING",
    "D1_WRAPPER_BODY_NAIVE",
    "D1_WRAPPER_FRAME",
    "DIRECT_ENV_NAME",
    "D1Experiment",
    "D1RolloutDefinition",
    "DirectExperiment",
    "build_d1_experiment",
    "build_d1_procedure_config",
    "build_d1_reward_policy",
    "build_d1_rollout_definition",
    "build_direct_experiment",
    "d1_ceiling_candidate",
    "d1_graph_definition",
    "d1_initial_candidate",
    "render_d1_frame",
]
