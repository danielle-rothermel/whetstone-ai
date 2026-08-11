from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dr_code.humaneval import HumanEvalTask, load_humaneval_plus
from whetstone_envs.core import Instance

from whetstone.envs.code_comp.constants import (
    CODE_COMP_ENV_NAME,
    CODE_COMP_STRATUM,
)
from whetstone.envs.task import Task

CODE_COMP_TASK_INSTANCE_SCHEMA = "whetstone.code_comp.task_instance"
CODE_COMP_TASK_INSTANCE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CodeCompTaskInstance:
    """A HumanEval+ task packed for the runner as a whetstone Instance."""

    instance: Instance
    humaneval_task: HumanEvalTask

    @property
    def input_code(self) -> str:
        return self.instance.prompt_inputs["input_code"]

    @property
    def gt_code_wo_comments(self) -> str:
        return self.input_code


def code_comp_task_hash(instance: Instance) -> str:
    """The stable Task identity hash for one code_comp instance."""
    return Task.from_instance(CODE_COMP_ENV_NAME, instance).task_hash()


def code_comp_task_instance_to_json(
    task: CodeCompTaskInstance,
) -> dict[str, Any]:
    """JSON-safe wire form for one packed HumanEval task."""
    return {
        "schema": CODE_COMP_TASK_INSTANCE_SCHEMA,
        "schema_version": CODE_COMP_TASK_INSTANCE_SCHEMA_VERSION,
        "instance": {
            "id": task.instance.id,
            "seed": task.instance.seed,
            "strata": list(task.instance.strata),
            "prompt_inputs": dict(task.instance.prompt_inputs),
            "gold": task.instance.gold,
        },
        "humaneval_task": {
            key: value
            for key, value in task.humaneval_task.model_dump(
                mode="json"
            ).items()
            if key in HumanEvalTask.model_fields
        },
    }


def code_comp_task_instance_from_json(
    payload: dict[str, Any] | CodeCompTaskInstance,
) -> CodeCompTaskInstance:
    """Reconstruct one packed task from its JSON wire form."""
    if isinstance(payload, CodeCompTaskInstance):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(
            f"code_comp task payload must be a mapping, got {type(payload)!r}"
        )
    if payload.get("schema") != CODE_COMP_TASK_INSTANCE_SCHEMA:
        raise ValueError(
            f"schema must be exactly {CODE_COMP_TASK_INSTANCE_SCHEMA!r}"
        )
    instance_payload = payload["instance"]
    humaneval_payload = payload["humaneval_task"]
    if not isinstance(instance_payload, dict) or not isinstance(
        humaneval_payload, dict
    ):
        raise ValueError("code_comp task payload must contain object records")
    humaneval_task = HumanEvalTask.model_validate(humaneval_payload)
    instance = Instance(
        id=str(instance_payload["id"]),
        seed=int(instance_payload["seed"]),
        strata=tuple(str(item) for item in instance_payload["strata"]),
        prompt_inputs={
            str(key): str(value)
            for key, value in instance_payload["prompt_inputs"].items()
        },
        gold=str(instance_payload["gold"]),
    )
    return CodeCompTaskInstance(
        instance=instance, humaneval_task=humaneval_task
    )


def ed1_instance_from_task(task: HumanEvalTask) -> CodeCompTaskInstance:
    """Pack one parsed HumanEval task into an ``CodeCompTaskInstance``."""
    gt_wo = task.ground_truth_code_without_comments or task.ground_truth_code
    instance = Instance(
        id=task.task_id,
        seed=0,
        strata=(CODE_COMP_STRATUM,),
        prompt_inputs={
            "input_code": gt_wo,
            "task_id": task.task_id,
            "prompt": task.prompt,
            "canonical_solution": task.canonical_solution,
            "entry_point": task.entry_point,
            "test": task.test,
        },
        gold=task.ground_truth_code,
    )
    return CodeCompTaskInstance(instance=instance, humaneval_task=task)


def humaneval_task_from_instance(instance: Instance) -> HumanEvalTask:
    """Reconstruct the parsed ``HumanEvalTask`` from an ed1 ``Instance``."""
    pi = instance.prompt_inputs
    return HumanEvalTask(
        task_id=pi["task_id"],
        prompt=pi["prompt"],
        canonical_solution=pi["canonical_solution"],
        entry_point=pi["entry_point"],
        test=pi["test"],
    )


def load_tasks(
    *,
    snapshot_path: Path | None = None,
    limit: int | None = None,
) -> tuple[CodeCompTaskInstance, ...]:
    """Load the pinned live dataset or an explicit Whetstone snapshot."""
    humaneval_tasks = load_humaneval_plus(
        prefer_snapshot=snapshot_path is not None,
        snapshot_path=snapshot_path,
    )
    if limit is not None:
        humaneval_tasks = humaneval_tasks[:limit]
    tasks: list[CodeCompTaskInstance] = []
    for plus in humaneval_tasks:
        ht = HumanEvalTask(
            task_id=plus.task_id,
            prompt=plus.prompt,
            canonical_solution=plus.canonical_solution,
            entry_point=plus.entry_point,
            test=plus.test,
        )
        tasks.append(ed1_instance_from_task(ht))
    return tuple(tasks)


__all__ = [
    "CODE_COMP_TASK_INSTANCE_SCHEMA",
    "CODE_COMP_TASK_INSTANCE_SCHEMA_VERSION",
    "CodeCompTaskInstance",
    "code_comp_task_hash",
    "code_comp_task_instance_from_json",
    "code_comp_task_instance_to_json",
    "ed1_instance_from_task",
    "humaneval_task_from_instance",
    "load_tasks",
]
