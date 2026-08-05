"""``build_env_experiment`` -- the single cross-environment factory.

:func:`build_env_experiment` binds one env's five deliverables into one
value: the Rollout Definition graph, the Initial (naive) and reference
(ceiling) Candidates, the internal + official Eval Configs (sharing one
Evaluation Procedure Config identity), and the Reward Policy. It is the
single entry point for constructing a family contract. The
internal-eval loop that drives a candidate through an injected transport
lives in :mod:`whetstone.envs.internal_eval`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from dr_providers import ProviderCallConfig

from whetstone.code_eval.aggregate import CompletenessPolicy
from whetstone.envs.procedure import env_procedure_config
from whetstone.envs.registry import DEFAULT_REPEATS, env_spec
from whetstone.envs.reward import build_reward_policy
from whetstone.envs.rollout_definition import (
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


class RolloutDefinitionLike(Protocol):
    """The structural Rollout Definition contract evaluation reads.

    Both the QA ``EnvRolloutDefinition`` (2-node) and the ed1
    ``EncDecRolloutDefinition`` (3-node) satisfy it, so evaluation reads
    ``graph_hash`` / ``provider_call_config`` / ``procedure_config_hash``
    uniformly across env kinds without a concrete-type coupling.
    """

    @property
    def graph_hash(self) -> str: ...

    @property
    def provider_call_config(self) -> ProviderCallConfig: ...

    @property
    def procedure_config_hash(self) -> str: ...


@dataclass(frozen=True, slots=True)
class EnvExperiment:
    """The five bound deliverables for one env, returned by the factory.

    ``rollout_definition``, ``initial_candidate``, ``ceiling_candidate``,
    ``eval_configs`` (internal + official, shared Procedure identity), and
    ``reward_policy`` -- everything evaluation needs without re-deriving an
    identity.
    """

    env_name: str
    rollout_definition: RolloutDefinitionLike
    initial_candidate: Candidate
    ceiling_candidate: Candidate
    eval_configs: EnvEvalConfigs
    reward_policy: RewardPolicy
    #: The declared completeness policy the aggregation reduction MUST use --
    #: the SAME policy folded into the official Eval Config identity, so the
    #: runtime reduction and the config hash never disagree on missing-data
    #: behaviour.
    completeness_policy: CompletenessPolicy = field(
        default_factory=CompletenessPolicy
    )

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
    max_skip_fraction: float = 0.0,
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
    max_skip_fraction:
        The declared completeness tolerance for a SKIP policy: the maximum
        fraction of skipped (missing/failed/invalid) rows still certified as a
        value; beyond it the official arm is forced incomplete. Identity-
        bearing (folds into ``eval_config_hash``). Inert under PROPAGATE.
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
        max_skip_fraction=max_skip_fraction,
        repeats=repeats,
        split_sizes=split_sizes,
    )
    completeness_policy = completeness.to_policy(
        max_skip_fraction=max_skip_fraction
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
        completeness_policy=completeness_policy,
    )


__all__ = [
    "EnvExperiment",
    "build_env_experiment",
]
