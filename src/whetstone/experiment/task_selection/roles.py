from __future__ import annotations

from dataclasses import dataclass
from enum import UNIQUE, StrEnum, verify

from pydantic import BaseModel, ConfigDict, model_validator


class TaskSplitManifestError(ValueError):
    """A typed failure parsing or applying a task-selection manifest."""


@verify(UNIQUE)
class TaskSplitRole(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@verify(UNIQUE)
class TaskRoleSelectionMethod(StrEnum):
    FULL_ROLE = "full_role"
    LOWEST_HISTORICAL_PASS_RATE = "lowest_historical_pass_rate"


class TaskRoleSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_content_hash: str
    pool_key: str
    role: TaskSplitRole
    task_ids: tuple[str, ...]
    selection_method: TaskRoleSelectionMethod = (
        TaskRoleSelectionMethod.FULL_ROLE
    )
    source_role_count: int | None = None
    eligible_pool_count: int | None = None
    excluded_task_ids: tuple[str, ...] = ()
    historical_pass_rates: tuple[float, ...] = ()

    @model_validator(mode="after")
    def _validate_selection(self) -> TaskRoleSelection:
        if self.historical_pass_rates and len(
            self.historical_pass_rates
        ) != len(self.task_ids):
            raise ValueError(
                "historical pass rates must align with selected task IDs"
            )
        if self.source_role_count is not None and self.source_role_count < len(
            self.task_ids
        ):
            raise ValueError(
                "source role count cannot be smaller than selection"
            )
        if (
            self.eligible_pool_count is not None
            and self.eligible_pool_count < len(self.task_ids)
        ):
            raise ValueError(
                "eligible pool count cannot be smaller than selection"
            )
        return self


@dataclass(frozen=True, slots=True)
class TaskSplitRoles:
    pool_key: str
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    content_hash: str

    @property
    def internal_ids(self) -> tuple[str, ...]:
        return self.train_ids + self.val_ids

    @property
    def official_ids(self) -> tuple[str, ...]:
        return self.test_ids

    def all_role_ids(self) -> frozenset[str]:
        return frozenset(self.train_ids + self.val_ids + self.test_ids)

    def ids_for(self, role: TaskSplitRole) -> tuple[str, ...]:
        if role is TaskSplitRole.TRAIN:
            return self.train_ids
        if role is TaskSplitRole.VALIDATION:
            return self.val_ids
        return self.test_ids


__all__ = [
    "TaskRoleSelection",
    "TaskRoleSelectionMethod",
    "TaskSplitManifestError",
    "TaskSplitRole",
    "TaskSplitRoles",
]
