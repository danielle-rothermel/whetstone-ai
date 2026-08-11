from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from dr_providers import ProviderCallConfig

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
from whetstone.evaluation.aggregate import CompletenessPolicy
from whetstone.experiment.candidate import Candidate
from whetstone.experiment.reward import RewardPolicy


class RolloutDefinitionLike(Protocol):
    """The structural Rollout Definition contract evaluation reads.

    Both the QA ``EnvRolloutDefinition`` (2-node) and the code_comp
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
    """Identity-bound components required to evaluate one environment.

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
