from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "Generation",
    "GenerationIndex",
]


@dataclass(frozen=True, slots=True)
class GenerationIndex:
    task_index: int
    sample_index: int

    def __post_init__(self) -> None:
        if self.task_index < 0:
            raise ValueError("task_index must be non-negative")
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")


@dataclass(frozen=True, slots=True)
class Generation:
    index: GenerationIndex
