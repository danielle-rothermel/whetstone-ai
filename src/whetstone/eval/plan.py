from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from pydantic import field_validator, model_validator

from whetstone.eval.config import (
    SCHEMA_SEED_PLAN,
    SCHEMA_TASK_SET,
    SCHEMA_TASK_TRIAL_ID,
    _FrozenModel,
    identity_hash_for,
)


class SelectionRule(_FrozenModel):
    kind: str
    params: tuple[tuple[str, str], ...] = ()


class TaskSet(_FrozenModel):
    manifest_id: str
    version: str
    dataset_revision: str
    task_hashes: tuple[str, ...] = ()
    selection_rule: SelectionRule | None = None

    @model_validator(mode="after")
    def validate_selector(self) -> Self:
        has_list = bool(self.task_hashes)
        has_rule = self.selection_rule is not None
        if has_list == has_rule:
            raise ValueError(
                "TaskSet carries exactly one of task_hashes or selection_rule"
            )
        if len(set(self.task_hashes)) != len(self.task_hashes):
            raise ValueError("task_hashes must be unique")
        return self

    def identity_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "manifest_id": self.manifest_id,
            "version": self.version,
            "dataset_revision": self.dataset_revision,
        }
        if self.selection_rule is not None:
            payload["selection_rule"] = {
                "kind": self.selection_rule.kind,
                "params": [list(pair) for pair in self.selection_rule.params],
            }
        else:
            payload["task_hashes"] = list(self.task_hashes)
        return payload

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=SCHEMA_TASK_SET,
            payload=self.identity_payload(),
        )


@dataclass(frozen=True, slots=True)
class TaskTrialProvenanceRow:
    task_hash: str
    seed_index: int
    rng_seed: int | None = None

    def __post_init__(self) -> None:
        if self.seed_index < 0:
            raise ValueError("seed_index must be non-negative")


class TaskTrialId(_FrozenModel):
    task_hash: str
    seed_index: int
    rng_seed: int | None = None

    @field_validator("seed_index")
    @classmethod
    def validate_seed_index(cls, value: int) -> int:
        if value < 0:
            raise ValueError("seed_index must be non-negative")
        return value

    def identity_payload(self) -> dict[str, object]:
        return {"task_hash": self.task_hash, "seed_index": self.seed_index}

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=SCHEMA_TASK_TRIAL_ID,
            payload=self.identity_payload(),
        )


class TaskTrial(_FrozenModel):
    task_trial_id: TaskTrialId


class SeedPlan(_FrozenModel):
    plan_id: str
    version: str
    task_hashes: tuple[str, ...]
    num_seeds: int
    rng_seeds: tuple[tuple[str, int], ...] = ()

    @field_validator("num_seeds")
    @classmethod
    def validate_num_seeds(cls, value: int) -> int:
        if value < 1:
            raise ValueError("num_seeds must be at least 1")
        return value

    @model_validator(mode="after")
    def validate_slots(self) -> Self:
        if not self.task_hashes:
            raise ValueError("task_hashes must not be empty")
        if len(set(self.task_hashes)) != len(self.task_hashes):
            raise ValueError("task_hashes must be unique")
        valid_keys = {
            f"{task_hash}#{index}"
            for task_hash in self.task_hashes
            for index in range(self.num_seeds)
        }
        seed_keys = [key for key, _seed in self.rng_seeds]
        if len(seed_keys) != len(set(seed_keys)):
            raise ValueError("repeat seed keys must be unique")
        unknown = set(seed_keys) - valid_keys
        if unknown:
            raise ValueError(
                "repeat seeds reference unknown slots: "
                + ", ".join(sorted(unknown))
            )
        return self

    def task_trials(self) -> tuple[TaskTrial, ...]:
        seeds = dict(self.rng_seeds)
        return tuple(
            TaskTrial(
                task_trial_id=TaskTrialId(
                    task_hash=task_hash,
                    seed_index=index,
                    rng_seed=seeds.get(f"{task_hash}#{index}"),
                )
            )
            for task_hash in self.task_hashes
            for index in range(self.num_seeds)
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "version": self.version,
            "task_hashes": list(self.task_hashes),
            "num_seeds": self.num_seeds,
        }

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=SCHEMA_SEED_PLAN,
            payload=self.identity_payload(),
        )


def seed_plan_from_provenance(
    rows: tuple[TaskTrialProvenanceRow, ...],
    *,
    plan_id: str,
    version: str,
) -> SeedPlan:
    task_order: list[str] = []
    indices_by_task: dict[str, set[int]] = {}
    seeds: dict[tuple[str, int], int] = {}
    for row in rows:
        if row.task_hash not in indices_by_task:
            task_order.append(row.task_hash)
            indices_by_task[row.task_hash] = set()
        indices = indices_by_task[row.task_hash]
        if row.seed_index in indices:
            raise ValueError(
                "duplicate (task_hash, seed_index) in provenance rows"
            )
        indices.add(row.seed_index)
        if row.rng_seed is not None:
            seeds[(row.task_hash, row.seed_index)] = row.rng_seed

    if not task_order:
        raise ValueError("provenance rows are empty")
    counts = {len(indices) for indices in indices_by_task.values()}
    if len(counts) != 1:
        raise ValueError(
            "every task must have the same number of repeat slots"
        )
    num_seeds = counts.pop()
    for task_hash, indices in indices_by_task.items():
        if indices != set(range(num_seeds)):
            raise ValueError(
                f"task {task_hash!r} repeat indices are not contiguous "
                f"0..{num_seeds - 1}"
            )

    seed_pairs = tuple(
        (f"{task_hash}#{index}", seeds[(task_hash, index)])
        for task_hash in task_order
        for index in range(num_seeds)
        if (task_hash, index) in seeds
    )
    return SeedPlan(
        plan_id=plan_id,
        version=version,
        task_hashes=tuple(task_order),
        num_seeds=num_seeds,
        rng_seeds=seed_pairs,
    )


__all__ = [
    "SeedPlan",
    "SelectionRule",
    "TaskSet",
    "TaskTrial",
    "TaskTrialId",
    "TaskTrialProvenanceRow",
    "seed_plan_from_provenance",
]
