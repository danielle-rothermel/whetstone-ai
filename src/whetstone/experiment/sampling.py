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
HELD_OUT = "held_out"

#: Every split role, in the order an experiment consumes them: the optimizer's
#: own search signal, then selection across runs, then the reporting split that
#: is touched exactly once.
SPLIT_ROLES: tuple[str, ...] = (INTERNAL_EVAL, OFFICIAL, HELD_OUT)

_ROLE_BY_SPLIT: dict[str, EvalRole] = {
    INTERNAL_EVAL: EvalRole.INTERNAL,
    OFFICIAL: EvalRole.OFFICIAL,
    HELD_OUT: EvalRole.HELD_OUT,
}


@runtime_checkable
class SamplingTaskLike(Protocol):
    """Minimal task identity contract for split derivation."""

    @property
    def task_id(self) -> str: ...


def evaluation_role_for_split(split_role: str) -> EvalRole:
    """Return the exact evidence role owned by a sampling split."""
    role = _ROLE_BY_SPLIT.get(split_role)
    if role is None:
        raise ValueError(f"unknown evaluation split role {split_role!r}")
    return role


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
    """The sampling artifacts for one split role in ``SPLIT_ROLES``."""

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
    """The experiment's eval splits and their shared evaluation procedure.

    ``internal`` and ``official`` are required. ``held_out`` is optional: an
    experiment that never reports on a frozen split (toy runs, envs smoke
    experiments) leaves it ``None``. When it is present it is a full
    :class:`EvalSplit` derived by its own ``derive_eval_split`` call, so its
    task identity, seed plan, and Eval Config are content-addressed on the
    same terms as the other two and can be intersected against them by
    :func:`assert_split_disjointness`.
    """

    env_name: str
    procedure_config_hash: str
    internal: EvalSplit
    official: EvalSplit
    held_out: EvalSplit | None = None

    def __post_init__(self) -> None:
        for expected, split in (
            (INTERNAL_EVAL, self.internal),
            (OFFICIAL, self.official),
            (HELD_OUT, self.held_out),
        ):
            if split is not None and split.split_role != expected:
                raise ValueError(
                    f"eval configs field for {expected!r} carries split "
                    f"role {split.split_role!r}"
                )

    @property
    def held_out_task_hashes(self) -> tuple[str, ...]:
        """The held-out task identities, empty when there is no held-out split."""
        if self.held_out is None:
            return ()
        return self.held_out.task_set.task_hashes

    def splits(self) -> dict[str, EvalSplit]:
        """The present splits keyed by split role, in ``SPLIT_ROLES`` order."""
        present = {INTERNAL_EVAL: self.internal, OFFICIAL: self.official}
        if self.held_out is not None:
            present[HELD_OUT] = self.held_out
        return present

    def split_for(self, split_role: str) -> EvalSplit:
        try:
            return self.splits()[split_role]
        except KeyError:
            raise KeyError(
                f"no eval split for split role {split_role!r}"
            ) from None

    def eval_config_for(self, split_role: str) -> EvalConfig:
        return self.split_for(split_role).eval_config


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
    """Two eval splits share a task identity."""


def assert_split_disjointness(configs: EvalConfigs) -> frozenset[str]:
    """Assert every present split owns a disjoint set of task hashes.

    This is the mechanical leakage check: it intersects the content-addressed
    task-hash sets of every split in ``configs`` pairwise. A held-out hash
    reaching the internal or official split raises
    :class:`HeldOutReferencedError`; any other overlap raises
    :class:`SplitOverlapError`. Returns the union of all task hashes so a
    caller can record the study's total task identity in one place.
    """
    # Within-split uniqueness is already a TaskSet validation, so this only
    # has to rule out cross-split sharing.
    splits = configs.splits()
    hashes = {
        role: frozenset(split.task_set.task_hashes)
        for role, split in splits.items()
    }
    roles = [role for role in SPLIT_ROLES if role in splits]
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            shared = hashes[left] & hashes[right]
            if not shared:
                continue
            sample = ", ".join(sorted(shared)[:3])
            if HELD_OUT in (left, right):
                raise HeldOutReferencedError(
                    f"splits {left!r} and {right!r} share "
                    f"{len(shared)} held-out task identities: {sample}"
                )
            raise SplitOverlapError(
                f"splits {left!r} and {right!r} share "
                f"{len(shared)} task identities: {sample}"
            )
    return frozenset().union(*hashes.values()) if hashes else frozenset()


__all__ = [
    "DEFAULT_NUM_SEEDS",
    "HELD_OUT",
    "INTERNAL_EVAL",
    "OFFICIAL",
    "SPLIT_ROLES",
    "Completeness",
    "EvalConfigs",
    "EvalSplit",
    "HeldOutReferencedError",
    "SamplingTaskLike",
    "SplitOverlapError",
    "assert_split_disjointness",
    "derive_eval_split",
    "evaluation_role_for_split",
    "validate_evaluation_role_for_split",
]
