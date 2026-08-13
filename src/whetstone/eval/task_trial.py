from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "TaskTrialKey",
]


@dataclass(frozen=True, slots=True)
class TaskTrialKey:
    task_index: int
    seed_index: int

    def __post_init__(self) -> None:
        if self.task_index < 0:
            raise ValueError("task_index must be non-negative")
        if self.seed_index < 0:
            raise ValueError("seed_index must be non-negative")
