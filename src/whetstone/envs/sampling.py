from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from whetstone_envs.core import Instance

from whetstone.core.roles import EvaluationRole
from whetstone.evaluation import (
    AggregationConfig,
    EvalConfig,
    EvalDefinition,
    EvaluationProcedureConfig,
    SamplePlan,
    SamplingConfig,
    SamplingDefinition,
    TaskSet,
)
from whetstone.evaluation.aggregate import (
    CompletenessPolicy,
    EvaluationMatrixPlan,
    RowPolicy,
)

_DEFINITION_VERSION = "1"

#: Default repeat count per task in a Sample Plan.
DEFAULT_NUM_SAMPLES = 3

#: The split roles this adapter samples from. ``held_out`` is deliberately
#: absent: no Sampling Config references it.
INTERNAL_EVAL = "internal_eval"
OFFICIAL = "official"


def validate_evaluation_role_for_split(
    *, split_role: str, evaluation_role: EvaluationRole
) -> None:
    """Require the exact evidence role owned by a sampling split."""
    if split_role == INTERNAL_EVAL:
        expected = EvaluationRole.INTERNAL
    elif split_role == OFFICIAL:
        expected = EvaluationRole.OFFICIAL
    else:
        raise ValueError(f"unknown evaluation split role {split_role!r}")
    if evaluation_role is not expected:
        raise ValueError(
            f"evaluation binding role {evaluation_role.value!r} does not "
            f"match split role {split_role!r}"
        )


class Completeness(StrEnum):
    """The Aggregation Config completeness policy over planned rows."""

    PROPAGATE = "propagate"
    SKIP = "skip"

    def to_policy(
        self, *, max_skip_fraction: float = 0.0
    ) -> CompletenessPolicy:
        """The :class:`CompletenessPolicy` this enum + tolerance denotes."""
        row_policy = (
            RowPolicy.PROPAGATE
            if self is Completeness.PROPAGATE
            else RowPolicy.SKIP
        )
        return CompletenessPolicy(
            row_policy=row_policy, max_skip_fraction=max_skip_fraction
        )


@dataclass(frozen=True, slots=True)
class EnvSplitSampling:
    """The sampling artifacts for one split (internal_eval or official)."""

    split_role: str
    tasks: tuple[Instance, ...]
    task_set: TaskSet
    sample_plan: SamplePlan
    sampling_config: SamplingConfig
    procedure_config: EvaluationProcedureConfig
    aggregation_config: AggregationConfig
    eval_config: EvalConfig

    @property
    def completeness_policy(self) -> CompletenessPolicy:
        """The runtime completeness policy bound by ``aggregation_config``."""
        assignment = dict(self.aggregation_config.assignment)
        return CompletenessPolicy(
            row_policy=RowPolicy(str(assignment["missing_data"])),
            max_skip_fraction=float(
                str(assignment.get("max_skip_fraction", "0.0000"))
            ),
        )

    @property
    def evaluation_matrix_plan(self) -> EvaluationMatrixPlan:
        """The exact aggregate plan composed by this split binding."""
        return EvaluationMatrixPlan(
            eval_config=self.eval_config,
            sampling_config=self.sampling_config,
            task_set=self.task_set,
            sample_plan=self.sample_plan,
            aggregation_config=self.aggregation_config,
        )


@dataclass(frozen=True, slots=True)
class EnvEvalConfigs:
    """The internal + official Eval Configs and their shared Procedure."""

    env_name: str
    procedure_config_hash: str
    internal: EnvSplitSampling
    official: EnvSplitSampling
    held_out_task_hashes: tuple[str, ...]

    def eval_config_for(self, split_role: str) -> EvalConfig:
        if split_role == INTERNAL_EVAL:
            return self.internal.eval_config
        if split_role == OFFICIAL:
            return self.official.eval_config
        raise KeyError(f"no eval config for split role {split_role!r}")


def derive_split_sampling(
    *,
    namespace: str,
    dataset_revision: str,
    split_role: str,
    tasks: tuple[Instance, ...],
    task_hash_of: Callable[[Instance], str],
    procedure: EvaluationProcedureConfig,
    aggregation: AggregationConfig,
    num_samples: int,
) -> EnvSplitSampling:
    """Derive one exact sampling and EvalConfig contract."""
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1; got {num_samples}")
    task_hashes = tuple(task_hash_of(task) for task in tasks)
    task_set = TaskSet(
        manifest_id=f"{namespace}.{split_role}",
        version=_DEFINITION_VERSION,
        dataset_revision=dataset_revision,
        task_hashes=task_hashes,
    )
    sample_plan = SamplePlan(
        plan_id=f"{namespace}.{split_role}",
        version=_DEFINITION_VERSION,
        task_hashes=task_hashes,
        num_samples=num_samples,
    )
    sampling = SamplingDefinition(
        definition_id=f"{namespace}.{split_role}.sampling",
        version=_DEFINITION_VERSION,
    ).materialize(
        {
            "task_set_hash": task_set.identity_hash(),
            "sample_plan_hash": sample_plan.identity_hash(),
        }
    )
    eval_config = EvalDefinition(
        definition_id=f"{namespace}.eval",
        version=_DEFINITION_VERSION,
    ).materialize(
        sampling=sampling,
        evaluation_procedure=procedure,
        aggregation=aggregation,
    )
    return EnvSplitSampling(
        split_role=split_role,
        tasks=tasks,
        task_set=task_set,
        sample_plan=sample_plan,
        sampling_config=sampling,
        procedure_config=procedure,
        aggregation_config=aggregation,
        eval_config=eval_config,
    )


class HeldOutReferencedError(AssertionError):
    """A Sampling Config referenced a held-out task identity."""


class SplitOverlapError(AssertionError):
    """The internal and official Task Sets share a task identity."""


__all__ = [
    "DEFAULT_NUM_SAMPLES",
    "INTERNAL_EVAL",
    "OFFICIAL",
    "Completeness",
    "EnvEvalConfigs",
    "EnvSplitSampling",
    "HeldOutReferencedError",
    "SplitOverlapError",
    "derive_split_sampling",
    "validate_evaluation_role_for_split",
]
