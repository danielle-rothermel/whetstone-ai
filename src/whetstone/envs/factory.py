"""``build_env_experiment`` -- the single cross-env factory the validation
runner uses.

:func:`build_env_experiment` binds one env's five deliverables into one
value: the Rollout Definition graph, the Initial (naive) and reference
(ceiling) Candidates, the internal + official Eval Configs (sharing one
Evaluation Procedure Config identity), and the Reward Policy. It is the
single entry point the 5x5 validation runner calls per env. The
internal-eval loop that drives a candidate through an injected transport
lives in :mod:`whetstone.envs.internal_eval`.
"""

from __future__ import annotations

from dataclasses import dataclass

from whetstone.envs.procedure import env_procedure_config
from whetstone.envs.registry import DEFAULT_REPEATS, env_spec
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
from whetstone.optimization.reward import RewardPolicy
from whetstone.optimization.schema import Candidate


@dataclass(frozen=True, slots=True)
class EnvExperiment:
    """The five bound deliverables for one env, returned by the factory.

    ``rollout_definition``, ``initial_candidate``, ``ceiling_candidate``,
    ``eval_configs`` (internal + official, shared Procedure identity), and
    ``reward_policy`` -- everything the validation runner needs to run one
    env's cell without re-deriving any identity.
    """

    env_name: str
    rollout_definition: EnvRolloutDefinition
    initial_candidate: Candidate
    ceiling_candidate: Candidate
    eval_configs: EnvEvalConfigs
    reward_policy: RewardPolicy

    def as_dict(self) -> dict[str, object]:
        """The factory's contract shape (the keys the runner reads)."""
        return {
            "rollout_definition": self.rollout_definition,
            "initial_candidate": self.initial_candidate,
            "ceiling_candidate": self.ceiling_candidate,
            "eval_configs": self.eval_configs,
            "reward_policy": self.reward_policy,
        }


def build_env_experiment(
    env_name: str,
    *,
    model: str,
    pool_n_per_stratum: int | None = None,
    completeness: Completeness = Completeness.PROPAGATE,
    repeats: int = DEFAULT_REPEATS,
    split_sizes: tuple[int, int, int] | None = None,
) -> EnvExperiment:
    """Build one env's complete experiment: the single validation entry point.

    Parameters
    ----------
    env_name:
        One of the five bound envs (``c22`` .. ``c23``).
    model:
        The task model route for the LLM Call Node's Provider Call Config.
    pool_n_per_stratum:
        Override the env's spec-default pool size (tests use a tiny pool).
    completeness:
        The Aggregation Config completeness policy (default propagate).
    repeats:
        The Repeat Plan repeat count (default the spec-default 3).
    split_sizes:
        Override the env's committed spec-default pool split (tests pass a
        tiny ``(internal, official, held_out)`` split for a small pool).

    The internal and official Eval Configs share the Rollout Definition's
    Evaluation Procedure Config identity, so ``graph_hash`` is stable across
    the two while their ``eval_config_hash`` values differ.
    """
    env = env_spec(env_name)
    rollout_definition = build_rollout_definition(env, model=model)
    procedure = env_procedure_config(env)
    pool = env.generate_pool(n_per_stratum=pool_n_per_stratum)
    eval_configs = build_eval_configs(
        env,
        pool=pool,
        procedure=procedure,
        completeness=completeness,
        repeats=repeats,
        split_sizes=split_sizes,
    )
    # The Rollout Definition's Procedure identity is the one both Eval Configs
    # fold in -- assert the partition holds at construction so a divergence is
    # caught here, not at execution.
    if rollout_definition.procedure_config_hash != (
        eval_configs.procedure_config_hash
    ):
        raise AssertionError(
            "Rollout Definition and Eval Configs disagree on the Evaluation "
            "Procedure Config identity"
        )
    return EnvExperiment(
        env_name=env.name,
        rollout_definition=rollout_definition,
        initial_candidate=initial_candidate(env),
        ceiling_candidate=ceiling_candidate(env),
        eval_configs=eval_configs,
        reward_policy=build_reward_policy(env),
    )


__all__ = [
    "EnvExperiment",
    "build_env_experiment",
]
