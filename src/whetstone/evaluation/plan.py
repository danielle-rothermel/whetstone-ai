"""Whetstone-owned task selection and repeat-plan contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from pydantic import field_validator, model_validator

from whetstone.evaluation.config import (
    SCHEMA_REPEAT_ID,
    SCHEMA_REPEAT_PLAN,
    SCHEMA_TASK_SET,
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
    task_identities: tuple[str, ...] = ()
    selection_rule: SelectionRule | None = None

    @model_validator(mode="after")
    def validate_selector(self) -> Self:
        has_list = bool(self.task_identities)
        has_rule = self.selection_rule is not None
        if has_list == has_rule:
            raise ValueError(
                "TaskSet carries exactly one of task_identities or "
                "selection_rule"
            )
        if len(set(self.task_identities)) != len(self.task_identities):
            raise ValueError("task_identities must be unique")
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
            payload["task_identities"] = list(self.task_identities)
        return payload

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=SCHEMA_TASK_SET,
            payload=self.identity_payload(),
        )


@dataclass(frozen=True, slots=True)
class RepeatProvenanceRow:
    task_identity: str
    repeat_index: int
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.repeat_index < 0:
            raise ValueError("repeat_index must be non-negative")


class RepeatId(_FrozenModel):
    task_identity: str
    index: int
    rng_seed: int | None = None

    @field_validator("index")
    @classmethod
    def validate_index(cls, value: int) -> int:
        if value < 0:
            raise ValueError("repeat index must be non-negative")
        return value

    def identity_payload(self) -> dict[str, object]:
        return {"task_identity": self.task_identity, "index": self.index}

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=SCHEMA_REPEAT_ID,
            payload=self.identity_payload(),
        )


class Repeat(_FrozenModel):
    repeat_id: RepeatId


class RepeatPlan(_FrozenModel):
    plan_id: str
    version: str
    task_identities: tuple[str, ...]
    repeat_count: int
    seeds: tuple[tuple[str, int], ...] = ()

    @field_validator("repeat_count")
    @classmethod
    def validate_repeat_count(cls, value: int) -> int:
        if value < 1:
            raise ValueError("repeat_count must be at least 1")
        return value

    @model_validator(mode="after")
    def validate_slots(self) -> Self:
        if not self.task_identities:
            raise ValueError("task_identities must not be empty")
        if len(set(self.task_identities)) != len(self.task_identities):
            raise ValueError("task_identities must be unique")
        valid_keys = {
            f"{task_identity}#{index}"
            for task_identity in self.task_identities
            for index in range(self.repeat_count)
        }
        seed_keys = [key for key, _seed in self.seeds]
        if len(seed_keys) != len(set(seed_keys)):
            raise ValueError("repeat seed keys must be unique")
        unknown = set(seed_keys) - valid_keys
        if unknown:
            raise ValueError(
                "repeat seeds reference unknown slots: "
                + ", ".join(sorted(unknown))
            )
        return self

    def repeats(self) -> tuple[Repeat, ...]:
        seeds = dict(self.seeds)
        return tuple(
            Repeat(
                repeat_id=RepeatId(
                    task_identity=task_identity,
                    index=index,
                    rng_seed=seeds.get(f"{task_identity}#{index}"),
                )
            )
            for task_identity in self.task_identities
            for index in range(self.repeat_count)
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "version": self.version,
            "task_identities": list(self.task_identities),
            "repeat_count": self.repeat_count,
        }

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=SCHEMA_REPEAT_PLAN,
            payload=self.identity_payload(),
        )


def repeat_plan_from_provenance(
    rows: tuple[RepeatProvenanceRow, ...],
    *,
    plan_id: str,
    version: str,
) -> RepeatPlan:
    task_order: list[str] = []
    indices_by_task: dict[str, set[int]] = {}
    seeds: dict[tuple[str, int], int] = {}
    for row in rows:
        if row.task_identity not in indices_by_task:
            task_order.append(row.task_identity)
            indices_by_task[row.task_identity] = set()
        indices = indices_by_task[row.task_identity]
        if row.repeat_index in indices:
            raise ValueError(
                "duplicate (task_identity, repeat_index) in provenance rows"
            )
        indices.add(row.repeat_index)
        if row.seed is not None:
            seeds[(row.task_identity, row.repeat_index)] = row.seed

    if not task_order:
        raise ValueError("provenance rows are empty")
    counts = {len(indices) for indices in indices_by_task.values()}
    if len(counts) != 1:
        raise ValueError(
            "every task must have the same number of repeat slots"
        )
    repeat_count = counts.pop()
    for task_identity, indices in indices_by_task.items():
        if indices != set(range(repeat_count)):
            raise ValueError(
                f"task {task_identity!r} repeat indices are not contiguous "
                f"0..{repeat_count - 1}"
            )

    seed_pairs = tuple(
        (f"{task_identity}#{index}", seeds[(task_identity, index)])
        for task_identity in task_order
        for index in range(repeat_count)
        if (task_identity, index) in seeds
    )
    return RepeatPlan(
        plan_id=plan_id,
        version=version,
        task_identities=tuple(task_order),
        repeat_count=repeat_count,
        seeds=seed_pairs,
    )


__all__ = [
    "Repeat",
    "RepeatId",
    "RepeatPlan",
    "RepeatProvenanceRow",
    "SelectionRule",
    "TaskSet",
    "repeat_plan_from_provenance",
]
