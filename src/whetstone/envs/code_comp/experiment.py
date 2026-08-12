from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dr_code.humaneval import HumanEvalTask
from whetstone_envs.core import Instance

from whetstone.envs.code_comp.constants import (
    CODE_COMP_DATASET_REVISION,
    CODE_COMP_ENV_NAME,
)
from whetstone.envs.code_comp.generation_graph.encdec import (
    EncDecGenerationGraph,
)
from whetstone.envs.code_comp.mutant.dataset import MutantRecord
from whetstone.envs.code_comp.reward.blended import (
    BoundedCompressionMetricConfig,
)
from whetstone.envs.code_comp.submission_result import CodeSubmissionResult
from whetstone.envs.factory import EnvExperiment

if TYPE_CHECKING:
    from whetstone.envs.code_comp.config import CodeCompExperimentConfig


@dataclass(frozen=True, slots=True)
class CodeCompExperiment(EnvExperiment):
    """Built code_comp experiment whose sole config input is ``config``."""

    config: CodeCompExperimentConfig = field(kw_only=True)
    scorer: Callable[..., CodeSubmissionResult] | None = field(
        default=None,
        kw_only=True,
    )


@dataclass(frozen=True, slots=True)
class DirectExperiment(CodeCompExperiment):
    """Direct-generation code_comp experiment."""

    humaneval_by_id: dict[str, HumanEvalTask] = field(
        default_factory=dict,
        kw_only=True,
    )

    def humaneval_for(self, instance: Instance) -> HumanEvalTask:
        return self.humaneval_by_id[str(instance.id)]

    @property
    def input_arm(self) -> str:
        assert self.config.direct is not None
        return self.config.direct.input_arm

    @property
    def rename_token(self) -> str:
        assert self.config.direct is not None
        return self.config.direct.rename_token

    @property
    def dataset_revision(self) -> str:
        return CODE_COMP_DATASET_REVISION


@dataclass(frozen=True, slots=True)
class EncDecExperiment(CodeCompExperiment):
    """Encoder-decoder code_comp experiment."""

    encdec_generation_graph: EncDecGenerationGraph | None = field(
        default=None,
        kw_only=True,
    )
    dataset_revision: str = field(default="", kw_only=True)
    blend_config: BoundedCompressionMetricConfig | None = field(
        default=None,
        kw_only=True,
    )

    @property
    def budget_ratio(self) -> float | None:
        if self.config.encdec is not None:
            return self.config.encdec.budget_ratio
        if self.config.mutant is not None:
            return self.config.mutant.budget_ratio
        return None


@dataclass(frozen=True, slots=True)
class MutantExperiment(EncDecExperiment):
    """Mutant-oracle encdec experiment."""

    mutants: dict[str, MutantRecord] = field(
        default_factory=dict,
        kw_only=True,
    )


def validate_encdec_blend(experiment: EncDecExperiment) -> None:
    """Require a blend config for canonical encdec experiments."""
    if isinstance(experiment, MutantExperiment):
        return
    if (
        experiment.env_name == CODE_COMP_ENV_NAME
        and experiment.blend_config is None
    ):
        raise ValueError("encdec requires a bounded compression blend config")


__all__ = [
    "CodeCompExperiment",
    "DirectExperiment",
    "EncDecExperiment",
    "MutantExperiment",
    "validate_encdec_blend",
]
