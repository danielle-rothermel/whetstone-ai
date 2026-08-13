from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from whetstone.core.roles import EvalRole
from whetstone.eval import (
    AggregationConfig,
    EvalConfig,
    EvalDefinition,
    EvalProcedureConfig,
    SeedPlan,
    SamplingConfig,
    SamplingDefinition,
    TaskSet,
)
from whetstone.eval.aggregate import (
    CompletenessPolicy,
    EvalMatrixPlan,
    RowPolicy,
)

_DEFINITION_VERSION = "1"

DEFAULT_NUM_SEEDS = 3

INTERNAL_EVAL = "internal_eval"
OFFICIAL = "official"


@runtime_checkable
class SamplingTaskLike(Protocol):
    """Minimal task identity contract for split derivation."""

    @property
    def task_id(self) -> str: ...


def evaluation_role_for_split(split_role: str) -> EvalRole:
    """Return the exact evidence role owned by a sampling split."""
    if split_role == INTERNAL_EVAL:
        return EvalRole.INTERNAL
    if split_role == OFFICIAL:
        return EvalRole.OFFICIAL
    raise ValueError(f"unknown evaluation split role {split_role!r}")


def validate_evaluation_role_for_split(
    *, split_role: str, evaluation_role: EvalRole
) -> None:
    """Require the exact evidence role owned by a sampling split."""
    expected = evaluation_role_for_split(split_role)
    if evaluation_role is not expected:
        raise ValueError(
            f"evaluation role {evaluation_role.value!r} does not "
            f"match split role {split_role!r}"
        )


class Completeness(StrEnum):
    """The aggregation config completeness policy over planned rows."""

    PROPAGATE = "propagate"
    SKIP = "skip"

    def to_policy(
        self, *, max_skip_fraction: float = 0.0
    ) -> CompletenessPolicy:
        row_policy = (
            RowPolicy.PROPAGATE
            if self is Completeness.PROPAGATE
            else RowPolicy.SKIP
        )
        return CompletenessPolicy(
            row_policy=row_policy, max_skip_fraction=max_skip_fraction
        )


@dataclass(frozen=True, slots=True)
class EvalSplit:
    """The sampling artifacts for one split (internal_eval or official)."""

    split_role: str
    tasks: tuple[SamplingTaskLike, ...]
    task_set: TaskSet
    seed_plan: SeedPlan
    sampling_config: SamplingConfig
    procedure_config: EvalProcedureConfig
    aggregation_config: AggregationConfig
    eval_config: EvalConfig

    @property
    def completeness_policy(self) -> CompletenessPolicy:
        assignment = dict(self.aggregation_config.assignment)
        return CompletenessPolicy(
            row_policy=RowPolicy(str(assignment["missing_data"])),
            max_skip_fraction=float(
                str(assignment.get("max_skip_fraction", "0.0000"))
            ),
        )

    @property
    def evaluation_matrix_plan(self) -> EvalMatrixPlan:
        return EvalMatrixPlan(
            eval_config=self.eval_config,
            sampling_config=self.sampling_config,
            task_set=self.task_set,
            seed_plan=self.seed_plan,
            aggregation_config=self.aggregation_config,
        )


@dataclass(frozen=True, slots=True)
class EvalConfigs:
    """The internal + official eval configs and their shared procedure."""

    env_name: str
    procedure_config_hash: str
    internal: EvalSplit
    official: EvalSplit
    held_out_task_hashes: tuple[str, ...]

    def eval_config_for(self, split_role: str) -> EvalConfig:
        if split_role == INTERNAL_EVAL:
            return self.internal.eval_config
        if split_role == OFFICIAL:
            return self.official.eval_config
        raise KeyError(f"no eval config for split role {split_role!r}")


def derive_eval_split(
    *,
    namespace: str,
    dataset_revision: str,
    split_role: str,
    tasks: tuple[SamplingTaskLike, ...],
    task_hash_of: Callable[[SamplingTaskLike], str],
    procedure: EvalProcedureConfig,
    aggregation: AggregationConfig,
    num_seeds: int,
) -> EvalSplit:
    """Derive one exact sampling and EvalConfig contract."""
    if num_seeds < 1:
        raise ValueError(f"num_seeds must be at least 1; got {num_seeds}")
    task_hashes = tuple(task_hash_of(task) for task in tasks)
    task_set = TaskSet(
        manifest_id=f"{namespace}.{split_role}",
        version=_DEFINITION_VERSION,
        dataset_revision=dataset_revision,
        task_hashes=task_hashes,
    )
    seed_plan = SeedPlan(
        plan_id=f"{namespace}.{split_role}",
        version=_DEFINITION_VERSION,
        task_hashes=task_hashes,
        num_seeds=num_seeds,
    )
    sampling = SamplingDefinition(
        definition_id=f"{namespace}.{split_role}.sampling",
        version=_DEFINITION_VERSION,
    ).materialize(
        {
            "task_set_hash": task_set.identity_hash(),
            "seed_plan_hash": seed_plan.identity_hash(),
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
    return EvalSplit(
        split_role=split_role,
        tasks=tasks,
        task_set=task_set,
        seed_plan=seed_plan,
        sampling_config=sampling,
        procedure_config=procedure,
        aggregation_config=aggregation,
        eval_config=eval_config,
    )


class HeldOutReferencedError(AssertionError):
    """A sampling config referenced a held-out task identity."""


class SplitOverlapError(AssertionError):
    """The internal and official task sets share a task identity."""


__all__ = [
    "DEFAULT_NUM_SEEDS",
    "INTERNAL_EVAL",
    "OFFICIAL",
    "Completeness",
    "EvalConfigs",
    "EvalSplit",
    "HeldOutReferencedError",
    "SamplingTaskLike",
    "SplitOverlapError",
    "derive_eval_split",
    "validate_evaluation_role_for_split",
]
