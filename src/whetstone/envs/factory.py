from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from dr_providers import ProviderCallConfig

from whetstone.envs.generation_graph import (
    build_generation_graph,
    ceiling_candidate,
    initial_candidate,
)
from whetstone.envs.procedure import env_procedure_config
from whetstone.envs.registry import DEFAULT_NUM_SAMPLES, env_spec
from whetstone.envs.reward import build_reward_policy
from whetstone.envs.sampling import (
    Completeness,
    EnvEvalConfigs,
    build_eval_configs,
)
from whetstone.evaluation.aggregate import CompletenessPolicy
from whetstone.experiment.candidate import Candidate
from whetstone.experiment.reward import RewardPolicy


class GenerationGraphLike(Protocol):
    """The structural Generation Graph contract evaluation reads.

    Both the QA ``EnvGenerationGraph`` (2-node) and the code_comp
    ``EncDecGenerationGraph`` (3-node) satisfy it, so evaluation reads
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

    ``generation_graph``, ``initial_candidate``, ``ceiling_candidate``,
    ``eval_configs`` (internal + official, shared Procedure identity), and
    ``reward_policy`` -- everything evaluation needs without re-deriving an
    identity.
    """

    env_name: str
    generation_graph: GenerationGraphLike
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
            "generation_graph": self.generation_graph,
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
    num_samples: int = DEFAULT_NUM_SAMPLES,
    split_sizes: tuple[int, int, int] | None = None,
) -> EnvExperiment:
    env = env_spec(env_name)
    generation_graph = build_generation_graph(env, model=model)
    procedure = env_procedure_config(env)
    pool = env.generate_pool(n_per_stratum=pool_n_per_stratum)
    eval_configs = build_eval_configs(
        env,
        pool=pool,
        procedure=procedure,
        completeness=completeness,
        max_skip_fraction=max_skip_fraction,
        num_samples=num_samples,
        split_sizes=split_sizes,
    )
    completeness_policy = completeness.to_policy(
        max_skip_fraction=max_skip_fraction
    )
    # The Generation Graph's Procedure identity is the one both Eval Configs
    # fold in -- assert the partition holds at construction so a divergence is
    # caught here, not at execution.
    if generation_graph.procedure_config_hash != (
        eval_configs.procedure_config_hash
    ):
        raise AssertionError(
            "Generation Graph and Eval Configs disagree on the Evaluation "
            "Procedure Config identity"
        )
    return EnvExperiment(
        env_name=env.name,
        generation_graph=generation_graph,
        initial_candidate=initial_candidate(env),
        ceiling_candidate=ceiling_candidate(env),
        eval_configs=eval_configs,
        reward_policy=build_reward_policy(env),
        completeness_policy=completeness_policy,
    )


__all__ = [
    "EnvExperiment",
    "GenerationGraphLike",
    "build_env_experiment",
]
