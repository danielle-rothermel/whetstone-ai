from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from dbos import DBOS
from dr_code.humaneval import (
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
    HumanEvalScoringProfile,
    HumanEvalTask,
    parse_human_eval_dataset,
    resolve_humaneval_scoring_profile,
)
from pydantic import BaseModel, ConfigDict, Field, field_serializer
from sqlalchemy import create_engine

from whetstone.platform.dataset_snapshot import (
    load_humaneval_snapshot,
)
from whetstone.platform.persistence import (
    ScoreAttemptInsertResult,
    ScoreAttemptInsertStatus,
    load_generation_run,
    load_node_attempts_for_generation_run,
    load_prediction_spec,
    persist_score_attempt,
    persist_score_harness_failure,
)
from whetstone.platform.runtime import resolve_application_database_url
from whetstone.platform.scoring import score_submission_run
from whetstone.records import (
    DEFAULT_SCORE_DATASET_NAME,
    DEFAULT_SCORE_DATASET_SPLIT,
    DatasetSnapshotIdentityPayload,
    GenerationRunRecord,
    NodeAttemptRecord,
    PredictionSpecRecord,
    ScoreAttemptRecord,
    ScoreHarnessFailureRecord,
    stable_score_attempt_id,
)

PLATFORM_SCORING_WORKFLOW_NAME = "whetstone_scoring"
LOAD_SCORING_TARGET_STEP_NAME = "whetstone_load_scoring_target"
LOAD_HUMANEVAL_SCORING_INPUT_STEP_NAME = (
    "whetstone_load_humaneval_scoring_input"
)
SCORING_STARTED_AT_STEP_NAME = "whetstone_scoring_started_at"
SCORE_SUBMISSION_STEP_NAME = "whetstone_score_submission"
PERSIST_SCORE_RESULT_STEP_NAME = "whetstone_persist_score_result"
WORKFLOW_ID_PREFIX = "whetstone-scoring"


class ScoreSubmissionWorkflowResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score_attempt_id: str
    insert_status: ScoreAttemptInsertStatus


class HumanEvalScoringInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: HumanEvalTask
    dataset_snapshot: DatasetSnapshotIdentityPayload

    @field_serializer("task")
    def serialize_task(self, task: HumanEvalTask) -> dict[str, Any]:
        return humaneval_task_payload(task)


class ScheduledScoreSubmissionWorkflow(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    score_attempt_id: str
    workflow_id: str
    scheduled: bool
    recovered: bool = False
    workflow_handle: Any | None = Field(default=None, exclude=True)


@DBOS.workflow(name=PLATFORM_SCORING_WORKFLOW_NAME)
def run_score_submission_workflow(
    generation_run_id: str,
    score_attempt_index: int = 0,
    scoring_profile_id: str = HUMANEVAL_SCORING_PROFILE_ID,
    scoring_profile_version: str = HUMANEVAL_SCORING_PROFILE_VERSION,
    dataset_name: str = DEFAULT_SCORE_DATASET_NAME,
    dataset_split: str = DEFAULT_SCORE_DATASET_SPLIT,
    execution_recipe_digest: str = "",
    platform_item_id: str = "",
) -> dict[str, Any]:
    # DBOS persists this workflow's arguments.  Resolve connection state at
    # execution time so DSNs and credentials never enter replay payloads.
    resolved_snapshot_path = require_dataset_snapshot_path(
        os.environ.get("WHETSTONE_HUMANEVAL_SNAPSHOT_PATH")
    )
    target = load_scoring_target_step(generation_run_id)
    spec = PredictionSpecRecord.model_validate(target["spec"])
    generation_run = GenerationRunRecord.model_validate(
        target["generation_run"]
    )
    node_attempts = tuple(
        NodeAttemptRecord.model_validate(payload)
        for payload in target["node_attempts"]
    )
    registered_snapshot = registered_dataset_snapshot_identity(spec)
    scoring_input = HumanEvalScoringInput.model_validate(
        load_humaneval_scoring_input_step(
            dataset_name,
            dataset_split,
            resolved_snapshot_path,
            spec.task_id,
            registered_snapshot.model_dump(mode="json"),
        )
    )
    task = scoring_input.task
    dataset_snapshot_payload = scoring_input.dataset_snapshot.model_dump(
        mode="json"
    )
    scoring_profile = resolve_humaneval_scoring_profile(
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
    )
    score_attempt_id = stable_score_attempt_id(
        generation_run_id=generation_run_id,
        scoring_profile_id=scoring_profile.profile_id,
        scoring_profile_version=scoring_profile.version,
        parser_profile_id=scoring_profile.parser_profile.profile_id,
        parser_version=scoring_profile.parser_profile.version,
        attempt_index=score_attempt_index,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        execution_recipe_digest=execution_recipe_digest,
        dataset_snapshot_identity=dataset_snapshot_payload,
    )
    started_at = datetime.fromisoformat(
        scoring_started_at_step(score_attempt_id)
    )
    score_result_payload = score_submission_step(
        spec.model_dump(mode="json"),
        generation_run.model_dump(mode="json"),
        [attempt.model_dump(mode="json") for attempt in node_attempts],
        task.model_dump(mode="json"),
        scoring_profile.model_dump(mode="json"),
        score_attempt_index,
        dataset_name,
        dataset_split,
        dataset_snapshot_payload,
        started_at.isoformat(),
        execution_recipe_digest,
        platform_item_id,
    )
    insert_result = ScoreAttemptInsertResult.model_validate(
        persist_score_result_step(score_result_payload)
    )
    return ScoreSubmissionWorkflowResult(
        score_attempt_id=score_attempt_id,
        insert_status=insert_result.status,
    ).model_dump(mode="json")


@DBOS.step(name=LOAD_SCORING_TARGET_STEP_NAME)
def load_scoring_target_step(
    generation_run_id: str,
) -> dict[str, Any]:
    engine = create_engine(resolve_application_database_url())
    try:
        with engine.begin() as connection:
            generation_run = load_generation_run(
                connection,
                generation_run_id=generation_run_id,
            )
            spec = load_prediction_spec(
                connection,
                prediction_id=generation_run.prediction_id,
            )
            node_attempts = load_node_attempts_for_generation_run(
                connection,
                generation_run_id=generation_run_id,
            )
        return {
            "spec": spec.model_dump(mode="json"),
            "generation_run": generation_run.model_dump(mode="json"),
            "node_attempts": [
                attempt.model_dump(mode="json") for attempt in node_attempts
            ],
        }
    finally:
        engine.dispose()


@DBOS.step(name=LOAD_HUMANEVAL_SCORING_INPUT_STEP_NAME)
def load_humaneval_scoring_input_step(
    dataset_name: str,
    dataset_split: str,
    dataset_snapshot_path: str,
    task_id: str,
    registered_snapshot_payload: dict[str, Any],
) -> dict[str, Any]:
    registered_snapshot = DatasetSnapshotIdentityPayload.model_validate(
        registered_snapshot_payload
    )
    snapshot = load_humaneval_snapshot(
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        snapshot_path=dataset_snapshot_path,
        expected_identity=registered_snapshot,
    )
    task = {
        task.task_id: task for task in parse_human_eval_dataset(snapshot.rows)
    }.get(task_id)
    if task is None:
        raise ValueError(f"HumanEval task not found: {task_id}")
    return HumanEvalScoringInput(
        task=task,
        dataset_snapshot=snapshot.identity,
    ).model_dump(mode="json")


def registered_dataset_snapshot_identity(
    spec: PredictionSpecRecord,
) -> DatasetSnapshotIdentityPayload:
    payload = spec.task.metadata.get("dataset_snapshot")
    if payload is None:
        raise ValueError(
            "prediction spec is missing dataset snapshot identity"
        )
    return DatasetSnapshotIdentityPayload.model_validate(payload)


@DBOS.step(name=SCORING_STARTED_AT_STEP_NAME)
def scoring_started_at_step(score_attempt_id: str) -> str:
    return timestamp_now_iso()


def timestamp_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@DBOS.step(name=SCORE_SUBMISSION_STEP_NAME)
def score_submission_step(
    spec_payload: dict[str, Any],
    generation_run_payload: dict[str, Any],
    node_attempt_payloads: list[dict[str, Any]],
    task_payload: dict[str, Any],
    scoring_profile_payload: dict[str, Any],
    score_attempt_index: int,
    dataset_name: str,
    dataset_split: str,
    dataset_snapshot_payload: dict[str, Any],
    started_at: str,
    execution_recipe_digest: str,
    platform_item_id: str,
) -> dict[str, Any]:
    scoring_profile = HumanEvalScoringProfile.model_validate(
        scoring_profile_payload
    )
    record = score_submission_run(
        spec=PredictionSpecRecord.model_validate(spec_payload),
        generation_run=GenerationRunRecord.model_validate(
            generation_run_payload
        ),
        node_attempts=tuple(
            NodeAttemptRecord.model_validate(payload)
            for payload in node_attempt_payloads
        ),
        task=humaneval_task_from_payload(task_payload),
        scoring_profile=scoring_profile,
        score_attempt_index=score_attempt_index,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        dataset_snapshot=DatasetSnapshotIdentityPayload.model_validate(
            dataset_snapshot_payload
        ),
        started_at=datetime.fromisoformat(started_at),
        execution_recipe_digest=execution_recipe_digest,
        platform_item_id=platform_item_id,
    )
    return record.model_dump(mode="json")


@DBOS.step(name=PERSIST_SCORE_RESULT_STEP_NAME)
def persist_score_result_step(
    score_result_payload: dict[str, Any],
) -> dict[str, Any]:
    engine = create_engine(resolve_application_database_url())
    try:
        with engine.begin() as connection:
            if score_result_payload.get("kind") == "harness_failure":
                result = persist_score_harness_failure(
                    connection,
                    harness_failure=ScoreHarnessFailureRecord.model_validate(
                        score_result_payload
                    ),
                )
            else:
                result = persist_score_attempt(
                    connection,
                    score_attempt=ScoreAttemptRecord.model_validate(
                        score_result_payload
                    ),
                )
        return result.model_dump(mode="json")
    finally:
        engine.dispose()


def humaneval_task_payload(task: HumanEvalTask) -> dict[str, Any]:
    return task.model_dump(
        mode="json",
        exclude={
            "ground_truth_code",
            "ground_truth_code_without_comments",
        },
    )


def humaneval_task_from_payload(payload: dict[str, Any]) -> HumanEvalTask:
    cleaned = dict(payload)
    cleaned.pop("ground_truth_code", None)
    cleaned.pop("ground_truth_code_without_comments", None)
    return HumanEvalTask.model_validate(cleaned)


def require_dataset_snapshot_path(
    dataset_snapshot_path: str | None,
) -> str:
    if dataset_snapshot_path is None or not dataset_snapshot_path:
        raise ValueError("dataset_snapshot_path is required")
    return dataset_snapshot_path
