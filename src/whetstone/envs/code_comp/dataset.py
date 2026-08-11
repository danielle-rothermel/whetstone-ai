from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dr_code.humaneval import HumanEvalTask, load_humaneval_plus
from whetstone_envs.core import Instance

from whetstone.envs.code_comp.constants import CODE_COMP_STRATUM


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
    tasks = load_humaneval_plus(
        prefer_snapshot=snapshot_path is not None,
        snapshot_path=snapshot_path,
    )
    if limit is not None:
        tasks = tasks[:limit]
    instances: list[CodeCompTaskInstance] = []
    for plus in tasks:
        ht = HumanEvalTask(
            task_id=plus.task_id,
            prompt=plus.prompt,
            canonical_solution=plus.canonical_solution,
            entry_point=plus.entry_point,
            test=plus.test,
        )
        instances.append(ed1_instance_from_task(ht))
    return tuple(instances)


__all__ = [
    "CodeCompTaskInstance",
    "ed1_instance_from_task",
    "humaneval_task_from_instance",
    "load_tasks",
]
