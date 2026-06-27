from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast

from datasets import load_dataset  # type: ignore[import-not-found]
from pydantic import BaseModel, ConfigDict, StrictInt

from dr_dspy.human_eval import HumanEvalTask, parse_human_eval_dataset

HumanEvalRow = Mapping[str, Any]


class HumanEvalDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> HumanEvalRow: ...


class SampledHumanEvalTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_index: StrictInt
    task: HumanEvalTask


def load_human_eval_rows(
    *,
    dataset_name: str,
    dataset_split: str,
) -> list[HumanEvalRow]:
    dataset = cast(
        HumanEvalDataset,
        load_dataset(dataset_name, split=dataset_split),
    )
    return [dataset[index] for index in range(len(dataset))]


def sample_human_eval_tasks_from_rows(
    rows: Sequence[HumanEvalRow],
    *,
    seed: int,
    sample_count: int,
) -> list[SampledHumanEvalTask]:
    tasks = parse_human_eval_dataset(rows)
    indices = list(range(len(tasks)))
    random.Random(seed).shuffle(indices)
    return [
        SampledHumanEvalTask(sample_index=sample_index, task=tasks[task_index])
        for sample_index, task_index in enumerate(indices[:sample_count])
    ]


def sample_human_eval_tasks(
    *,
    seed: int,
    sample_count: int,
    dataset_name: str,
    dataset_split: str,
) -> list[SampledHumanEvalTask]:
    return sample_human_eval_tasks_from_rows(
        load_human_eval_rows(
            dataset_name=dataset_name,
            dataset_split=dataset_split,
        ),
        seed=seed,
        sample_count=sample_count,
    )

