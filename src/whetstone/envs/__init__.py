"""Whetstone adapters binding the five whetstone-envs task families.

This package is the thin adapter the whetstone-envs PLAN's "Integration
handoff" describes: it declares each candidate's generator + oracle + prompts
as a single LLM Call Node + single terminal Eval Node Rollout Definition,
wraps the env ``TaskPool`` splits as the internal / official Evaluation
Contexts (held-out untouched), and wires the env oracle's 0/1 result into
Metric Facts + a ``env_exact_match`` Score. It owns the execution-contract
concerns (Rollout Definition, Eval Configs, Reward Policy) that deliberately
do not live in whetstone-envs.

The single cross-env entry point is
:func:`~whetstone.envs.factory.build_env_experiment`; the transport-injected
internal-eval loop is
:func:`~whetstone.envs.internal_eval.run_internal_eval`.
"""

from __future__ import annotations

from whetstone.envs.factory import EnvExperiment, build_env_experiment
from whetstone.envs.internal_eval import (
    InternalEvalResult,
    run_internal_eval,
)
from whetstone.envs.oracle_operator import (
    ENV_EXACT_MATCH_NAME,
    ENV_EXACT_MATCH_UNIT,
    env_exact_match_fact,
    env_exact_match_score,
)
from whetstone.envs.procedure import env_procedure_config
from whetstone.envs.registry import (
    ENV_NAMES,
    EnvSpec,
    UnknownEnvError,
    env_spec,
)
from whetstone.envs.reward import build_reward_policy
from whetstone.envs.rollout_definition import (
    EnvRolloutDefinition,
    build_rollout_definition,
    ceiling_candidate,
    initial_candidate,
)
from whetstone.envs.sampling import (
    Completeness,
    EnvEvalConfigs,
    build_eval_configs,
)
from whetstone.envs.task import EnvTask

__all__ = [
    "ENV_EXACT_MATCH_NAME",
    "ENV_EXACT_MATCH_UNIT",
    "ENV_NAMES",
    "Completeness",
    "EnvEvalConfigs",
    "EnvExperiment",
    "EnvRolloutDefinition",
    "EnvSpec",
    "EnvTask",
    "InternalEvalResult",
    "UnknownEnvError",
    "build_env_experiment",
    "build_eval_configs",
    "build_reward_policy",
    "build_rollout_definition",
    "ceiling_candidate",
    "env_exact_match_fact",
    "env_exact_match_score",
    "env_procedure_config",
    "env_spec",
    "initial_candidate",
    "run_internal_eval",
]
