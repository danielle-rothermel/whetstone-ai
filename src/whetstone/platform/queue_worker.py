from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from dbos import DBOS
from dr_platform import dedup_enqueue
from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, StrictStr

from whetstone.platform.graph_workflow import (
    platform_generation_workflow_id,
    run_prediction_graph_workflow,
)
from whetstone.records import stable_generation_run_id

PLATFORM_GENERATION_QUEUE_NAME = "dr-dspy-platform-generation-v1"
DEFAULT_ATTEMPT_INDEX = 0
QUEUE_CONFLICT_POLICY = "always_update"

type GenerationWorkflow = Callable[[str, str, int], str]


class EnqueuedPredictionWorkflow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    generation_run_id: StrictStr
    workflow_id: StrictStr
    enqueued: StrictBool


class EnqueuePredictionWorkflowsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queue_name: StrictStr
    enqueued_count: StrictInt
    existing_count: StrictInt
    workflows: tuple[EnqueuedPredictionWorkflow, ...]


def register_platform_generation_queue(
    *,
    worker_concurrency: int,
    queue_name: str = PLATFORM_GENERATION_QUEUE_NAME,
) -> Any:
    return DBOS.register_queue(
        queue_name,
        worker_concurrency=worker_concurrency,
        on_conflict=QUEUE_CONFLICT_POLICY,
    )


def listen_to_platform_generation_queue(
    *,
    queue_name: str = PLATFORM_GENERATION_QUEUE_NAME,
) -> None:
    DBOS.listen_queues([queue_name])


def enqueue_prediction_graph_workflows(
    *,
    database_url: str,
    prediction_ids: Sequence[str],
    attempt_index: int = DEFAULT_ATTEMPT_INDEX,
    queue_name: str = PLATFORM_GENERATION_QUEUE_NAME,
    workflow: GenerationWorkflow = run_prediction_graph_workflow,
) -> EnqueuePredictionWorkflowsResult:
    workflows: list[EnqueuedPredictionWorkflow] = []
    enqueued_count = 0
    existing_count = 0
    for prediction_id in prediction_ids:
        enqueued = enqueue_prediction_graph_workflow(
            database_url=database_url,
            prediction_id=prediction_id,
            attempt_index=attempt_index,
            queue_name=queue_name,
            workflow=workflow,
        )
        if enqueued.enqueued:
            enqueued_count += 1
        else:
            existing_count += 1
        workflows.append(enqueued)
    return EnqueuePredictionWorkflowsResult(
        queue_name=queue_name,
        enqueued_count=enqueued_count,
        existing_count=existing_count,
        workflows=tuple(workflows),
    )


def enqueue_prediction_graph_workflow(
    *,
    database_url: str,
    prediction_id: str,
    attempt_index: int = DEFAULT_ATTEMPT_INDEX,
    queue_name: str = PLATFORM_GENERATION_QUEUE_NAME,
    workflow: GenerationWorkflow = run_prediction_graph_workflow,
) -> EnqueuedPredictionWorkflow:
    generation_run_id = stable_generation_run_id(
        prediction_id=prediction_id,
        attempt_index=attempt_index,
    )
    outcome = dedup_enqueue(
        queue_name=queue_name,
        workflow_id=platform_generation_workflow_id(generation_run_id),
        workflow=workflow,
        args=(database_url, prediction_id, attempt_index),
    )
    return EnqueuedPredictionWorkflow(
        prediction_id=prediction_id,
        generation_run_id=generation_run_id,
        workflow_id=outcome.workflow_id,
        enqueued=outcome.enqueued,
    )
