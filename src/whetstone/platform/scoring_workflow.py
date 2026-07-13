from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from dbos import DBOS, SetWorkflowID
from dr_code.humaneval import (
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
    HumanEvalScoringProfile,
    HumanEvalTask,
    parse_human_eval_dataset,
    resolve_humaneval_scoring_profile,
)
from dr_platform import WORKFLOW_START_RACE_ERRORS
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
from whetstone.platform.scoring import score_submission_run
from whetstone.platform.scoring_workflow_state import (
    ScoringWorkflowPresence,
    classify_scoring_workflow_presence,
    recover_orphan_scoring_workflow,
)
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

PLATFORM_SCORING_WORKFLOW_NAME = (
    "dr_dspy_platform_humaneval_submission_scoring_v1"
)
LOAD_SCORING_TARGET_STEP_NAME = "dr_dspy_platform_load_scoring_target_v1"
LOAD_HUMANEVAL_SCORING_INPUT_STEP_NAME = (
    "dr_dspy_platform_load_humaneval_scoring_input_v2"
)
SCORING_STARTED_AT_STEP_NAME = "dr_dspy_platform_scoring_started_at_v1"
SCORE_SUBMISSION_STEP_NAME = "dr_dspy_platform_score_submission_v1"
PERSIST_SCORE_RESULT_STEP_NAME = "dr_dspy_platform_persist_score_result_v1"
WORKFLOW_ID_PREFIX = "platform-submission-score-v1"


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
    database_url: str,
    generation_run_id: str,
    score_attempt_index: int = 0,
    scoring_profile_id: str = HUMANEVAL_SCORING_PROFILE_ID,
    scoring_profile_version: str = HUMANEVAL_SCORING_PROFILE_VERSION,
    dataset_name: str = DEFAULT_SCORE_DATASET_NAME,
    dataset_split: str = DEFAULT_SCORE_DATASET_SPLIT,
    dataset_snapshot_path: str | None = None,
) -> dict[str, Any]:
    resolved_snapshot_path = require_dataset_snapshot_path(
        dataset_snapshot_path
    )
    target = load_scoring_target_step(database_url, generation_run_id)
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
    )
    insert_result = ScoreAttemptInsertResult.model_validate(
        persist_score_result_step(database_url, score_result_payload)
    )
    return ScoreSubmissionWorkflowResult(
        score_attempt_id=score_attempt_id,
        insert_status=insert_result.status,
    ).model_dump(mode="json")


def start_score_submission_workflow(
    database_url: str,
    generation_run_id: str,
    score_attempt_index: int = 0,
    scoring_profile_id: str = HUMANEVAL_SCORING_PROFILE_ID,
    scoring_profile_version: str = HUMANEVAL_SCORING_PROFILE_VERSION,
    dataset_name: str = DEFAULT_SCORE_DATASET_NAME,
    dataset_split: str = DEFAULT_SCORE_DATASET_SPLIT,
    dataset_snapshot_path: str | None = None,
) -> str:
    score_attempt_id, _handle = _start_score_submission_workflow_handle(
        database_url=database_url,
        generation_run_id=generation_run_id,
        score_attempt_index=score_attempt_index,
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        dataset_snapshot_path=dataset_snapshot_path,
    )
    return score_attempt_id


def schedule_score_submission_workflow(
    database_url: str,
    generation_run_id: str,
    score_attempt_index: int = 0,
    scoring_profile_id: str = HUMANEVAL_SCORING_PROFILE_ID,
    scoring_profile_version: str = HUMANEVAL_SCORING_PROFILE_VERSION,
    dataset_name: str = DEFAULT_SCORE_DATASET_NAME,
    dataset_split: str = DEFAULT_SCORE_DATASET_SPLIT,
    dataset_snapshot_path: str | None = None,
    recover_orphans: bool = True,
) -> ScheduledScoreSubmissionWorkflow:
    resolved_snapshot_path = require_dataset_snapshot_path(
        dataset_snapshot_path
    )
    score_attempt_id = score_attempt_id_for_workflow(
        generation_run_id=generation_run_id,
        score_attempt_index=score_attempt_index,
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
    )
    workflow_id = platform_scoring_workflow_id(score_attempt_id)
    presence = classify_scoring_workflow_presence(
        database_url=database_url,
        score_attempt_id=score_attempt_id,
        workflow_id=workflow_id,
    )
    if presence is ScoringWorkflowPresence.COMPLETE:
        return ScheduledScoreSubmissionWorkflow(
            score_attempt_id=score_attempt_id,
            workflow_id=workflow_id,
            scheduled=False,
        )
    if presence is ScoringWorkflowPresence.IN_FLIGHT:
        return ScheduledScoreSubmissionWorkflow(
            score_attempt_id=score_attempt_id,
            workflow_id=workflow_id,
            scheduled=False,
        )
    if presence is ScoringWorkflowPresence.ORPHAN:
        if recover_orphans and recover_orphan_scoring_workflow(
            database_url=database_url,
            workflow_id=workflow_id,
            score_attempt_id=score_attempt_id,
            replay_workflow=run_score_submission_workflow,
            replay_args=(
                database_url,
                generation_run_id,
                score_attempt_index,
                scoring_profile_id,
                scoring_profile_version,
                dataset_name,
                dataset_split,
                resolved_snapshot_path,
            ),
        ):
            return ScheduledScoreSubmissionWorkflow(
                score_attempt_id=score_attempt_id,
                workflow_id=workflow_id,
                scheduled=True,
                recovered=True,
            )
        return ScheduledScoreSubmissionWorkflow(
            score_attempt_id=score_attempt_id,
            workflow_id=workflow_id,
            scheduled=False,
        )
    with SetWorkflowID(workflow_id):
        try:
            workflow_handle = DBOS.start_workflow(
                run_score_submission_workflow,
                database_url,
                generation_run_id,
                score_attempt_index,
                scoring_profile_id,
                scoring_profile_version,
                dataset_name,
                dataset_split,
                resolved_snapshot_path,
            )
        except WORKFLOW_START_RACE_ERRORS:
            return ScheduledScoreSubmissionWorkflow(
                score_attempt_id=score_attempt_id,
                workflow_id=workflow_id,
                scheduled=False,
            )
        except Exception as error:
            if _scoring_workflow_start_raced(
                workflow_id=workflow_id,
                error=error,
            ):
                return ScheduledScoreSubmissionWorkflow(
                    score_attempt_id=score_attempt_id,
                    workflow_id=workflow_id,
                    scheduled=False,
                )
            raise
    return ScheduledScoreSubmissionWorkflow(
        score_attempt_id=score_attempt_id,
        workflow_id=workflow_id,
        scheduled=True,
        workflow_handle=workflow_handle,
    )


def await_scheduled_score_workflows(handles: Sequence[Any]) -> None:
    for handle in handles:
        handle.get_result()


def run_score_submission_workflow_once(
    database_url: str,
    generation_run_id: str,
    score_attempt_index: int = 0,
    scoring_profile_id: str = HUMANEVAL_SCORING_PROFILE_ID,
    scoring_profile_version: str = HUMANEVAL_SCORING_PROFILE_VERSION,
    dataset_name: str = DEFAULT_SCORE_DATASET_NAME,
    dataset_split: str = DEFAULT_SCORE_DATASET_SPLIT,
    dataset_snapshot_path: str | None = None,
) -> ScoreSubmissionWorkflowResult:
    _score_attempt_id, handle = _start_score_submission_workflow_handle(
        database_url=database_url,
        generation_run_id=generation_run_id,
        score_attempt_index=score_attempt_index,
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        dataset_snapshot_path=dataset_snapshot_path,
    )
    result = handle.get_result()
    if not isinstance(result, dict):
        raise TypeError("platform scoring workflow returned a non-dict result")
    return ScoreSubmissionWorkflowResult.model_validate(result)


def platform_scoring_workflow_id(score_attempt_id: str) -> str:
    return f"{WORKFLOW_ID_PREFIX}:{score_attempt_id}"


def score_attempt_id_for_workflow(
    *,
    generation_run_id: str,
    score_attempt_index: int,
    scoring_profile_id: str,
    scoring_profile_version: str,
    dataset_name: str = DEFAULT_SCORE_DATASET_NAME,
    dataset_split: str = DEFAULT_SCORE_DATASET_SPLIT,
) -> str:
    scoring_profile = resolve_humaneval_scoring_profile(
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
    )
    return stable_score_attempt_id(
        generation_run_id=generation_run_id,
        scoring_profile_id=scoring_profile.profile_id,
        scoring_profile_version=scoring_profile.version,
        parser_profile_id=scoring_profile.parser_profile.profile_id,
        parser_version=scoring_profile.parser_profile.version,
        attempt_index=score_attempt_index,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
    )


def _scoring_workflow_start_raced(
    *,
    workflow_id: str,
    error: BaseException,
) -> bool:
    _ = workflow_id
    return isinstance(error, WORKFLOW_START_RACE_ERRORS)


def _start_score_submission_workflow_handle(
    *,
    database_url: str,
    generation_run_id: str,
    score_attempt_index: int,
    scoring_profile_id: str,
    scoring_profile_version: str,
    dataset_name: str,
    dataset_split: str,
    dataset_snapshot_path: str | None,
) -> tuple[str, Any]:
    resolved_snapshot_path = require_dataset_snapshot_path(
        dataset_snapshot_path
    )
    score_attempt_id = score_attempt_id_for_workflow(
        generation_run_id=generation_run_id,
        score_attempt_index=score_attempt_index,
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
    )
    with SetWorkflowID(platform_scoring_workflow_id(score_attempt_id)):
        handle = DBOS.start_workflow(
            run_score_submission_workflow,
            database_url,
            generation_run_id,
            score_attempt_index,
            scoring_profile_id,
            scoring_profile_version,
            dataset_name,
            dataset_split,
            resolved_snapshot_path,
        )
    return score_attempt_id, handle


@DBOS.step(name=LOAD_SCORING_TARGET_STEP_NAME)
def load_scoring_target_step(
    database_url: str,
    generation_run_id: str,
) -> dict[str, Any]:
    engine = create_engine(database_url)
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
        task.task_id: task
        for task in parse_human_eval_dataset(snapshot.rows)
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
    )
    return record.model_dump(mode="json")


@DBOS.step(name=PERSIST_SCORE_RESULT_STEP_NAME)
def persist_score_result_step(
    database_url: str,
    score_result_payload: dict[str, Any],
) -> dict[str, Any]:
    engine = create_engine(database_url)
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
