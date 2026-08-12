from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from dr_graph import GraphConfig
from dr_providers import ProviderCallConfig

from whetstone.evaluation.aggregate import CompletenessPolicy
from whetstone.experiment.candidate import Candidate
from whetstone.experiment.reward import RewardPolicy
from whetstone.experiment.sampling import EvalConfigs


class GenerationGraphLike(Protocol):
    """The structural generation graph contract evaluation reads."""

    @property
    def graph_hash(self) -> str: ...

    @property
    def graph_config(self) -> GraphConfig: ...

    @property
    def provider_call_config(self) -> ProviderCallConfig: ...

    @property
    def procedure_config_hash(self) -> str: ...


@dataclass(frozen=True, slots=True)
class Experiment:
    """Identity-bound components required to evaluate one environment."""

    env_name: str
    generation_graph: GenerationGraphLike
    initial_candidate: Candidate
    ceiling_candidate: Candidate
    eval_configs: EvalConfigs
    reward_policy: RewardPolicy
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


__all__ = [
    "Experiment",
    "GenerationGraphLike",
]
