from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from dbos import DBOS, SetWorkflowID
from pydantic import BaseModel, ConfigDict
from sqlalchemy import create_engine

from dr_dspy.humaneval.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    PARSER_PROFILE_VERSION,
    resolve_parser_profile,
)
from dr_dspy.humaneval.sampling import load_human_eval_rows
from dr_dspy.humaneval.task import HumanEvalTask, parse_human_eval_dataset
from dr_dspy.platform.persistence import (
    ScoreAttemptInsertResult,
    ScoreAttemptInsertStatus,
    load_generation_run,
    load_node_attempts_for_generation_run,
    load_prediction_spec,
    persist_score_attempt,
)
from dr_dspy.platform.scoring import (
    DEFAULT_HUMANEVAL_TIMEOUT_SECONDS,
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
    score_generation_run,
)
from dr_dspy.records import (
    GenerationRunRecord,
    NodeAttemptRecord,
    PredictionSpecRecord,
    ScoreAttemptRecord,
    stable_score_attempt_id,
)

PLATFORM_SCORING_WORKFLOW_NAME = "dr_dspy_platform_humaneval_scoring_v1"
LOAD_SCORING_TARGET_STEP_NAME = "dr_dspy_platform_load_scoring_target_v1"
LOAD_HUMANEVAL_TASK_STEP_NAME = "dr_dspy_platform_load_humaneval_task_v1"
SCORING_STARTED_AT_STEP_NAME = "dr_dspy_platform_scoring_started_at_v1"
SCORE_GENERATION_STEP_NAME = "dr_dspy_platform_score_generation_v1"
PERSIST_SCORE_ATTEMPT_STEP_NAME = "dr_dspy_platform_persist_score_attempt_v1"
WORKFLOW_ID_PREFIX = "platform-score-v1"
DEFAULT_HUMANEVAL_DATASET_NAME = "evalplus/humanevalplus"
DEFAULT_HUMANEVAL_DATASET_SPLIT = "test"


class ScoreGenerationWorkflowResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score_attempt_id: str
    insert_status: ScoreAttemptInsertStatus


@DBOS.workflow(name=PLATFORM_SCORING_WORKFLOW_NAME)
def run_score_generation_workflow(
    database_url: str,
    generation_run_id: str,
    score_attempt_index: int = 0,
    scoring_profile_id: str = HUMANEVAL_SCORING_PROFILE_ID,
    scoring_profile_version: str = HUMANEVAL_SCORING_PROFILE_VERSION,
    parser_profile_id: str = BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    parser_version: str = PARSER_PROFILE_VERSION,
    timeout_seconds: float = DEFAULT_HUMANEVAL_TIMEOUT_SECONDS,
    dataset_name: str = DEFAULT_HUMANEVAL_DATASET_NAME,
    dataset_split: str = DEFAULT_HUMANEVAL_DATASET_SPLIT,
) -> dict[str, Any]:
    target = load_scoring_target_step(database_url, generation_run_id)
    spec = PredictionSpecRecord.model_validate(target["spec"])
    generation_run = GenerationRunRecord.model_validate(
        target["generation_run"]
    )
    node_attempts = tuple(
        NodeAttemptRecord.model_validate(payload)
        for payload in target["node_attempts"]
    )
    task = humaneval_task_from_payload(
        load_humaneval_task_step(
            dataset_name,
            dataset_split,
            spec.task_id,
        )
    )
    score_attempt_id = stable_score_attempt_id(
        generation_run_id=generation_run_id,
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
        parser_profile_id=parser_profile_id,
        parser_version=parser_version,
        attempt_index=score_attempt_index,
    )
    started_at = datetime.fromisoformat(
        scoring_started_at_step(score_attempt_id)
    )
    score_attempt_payload = score_generation_step(
        spec.model_dump(mode="json"),
        generation_run.model_dump(mode="json"),
        [attempt.model_dump(mode="json") for attempt in node_attempts],
        task.model_dump(mode="json"),
        scoring_profile_id,
        scoring_profile_version,
        parser_profile_id,
        parser_version,
        score_attempt_index,
        timeout_seconds,
        started_at.isoformat(),
    )
    insert_result = ScoreAttemptInsertResult.model_validate(
        persist_score_attempt_step(database_url, score_attempt_payload)
    )
    return ScoreGenerationWorkflowResult(
        score_attempt_id=score_attempt_id,
        insert_status=insert_result.status,
    ).model_dump(mode="json")


def start_score_generation_workflow(
    database_url: str,
    generation_run_id: str,
    score_attempt_index: int = 0,
    scoring_profile_id: str = HUMANEVAL_SCORING_PROFILE_ID,
    scoring_profile_version: str = HUMANEVAL_SCORING_PROFILE_VERSION,
    parser_profile_id: str = BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    parser_version: str = PARSER_PROFILE_VERSION,
    timeout_seconds: float = DEFAULT_HUMANEVAL_TIMEOUT_SECONDS,
    dataset_name: str = DEFAULT_HUMANEVAL_DATASET_NAME,
    dataset_split: str = DEFAULT_HUMANEVAL_DATASET_SPLIT,
) -> str:
    score_attempt_id, _handle = _start_score_generation_workflow_handle(
        database_url=database_url,
        generation_run_id=generation_run_id,
        score_attempt_index=score_attempt_index,
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
        parser_profile_id=parser_profile_id,
        parser_version=parser_version,
        timeout_seconds=timeout_seconds,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
    )
    return score_attempt_id


def run_score_generation_workflow_once(
    database_url: str,
    generation_run_id: str,
    score_attempt_index: int = 0,
    scoring_profile_id: str = HUMANEVAL_SCORING_PROFILE_ID,
    scoring_profile_version: str = HUMANEVAL_SCORING_PROFILE_VERSION,
    parser_profile_id: str = BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    parser_version: str = PARSER_PROFILE_VERSION,
    timeout_seconds: float = DEFAULT_HUMANEVAL_TIMEOUT_SECONDS,
    dataset_name: str = DEFAULT_HUMANEVAL_DATASET_NAME,
    dataset_split: str = DEFAULT_HUMANEVAL_DATASET_SPLIT,
) -> ScoreGenerationWorkflowResult:
    _score_attempt_id, handle = _start_score_generation_workflow_handle(
        database_url=database_url,
        generation_run_id=generation_run_id,
        score_attempt_index=score_attempt_index,
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
        parser_profile_id=parser_profile_id,
        parser_version=parser_version,
        timeout_seconds=timeout_seconds,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
    )
    result = handle.get_result()
    if not isinstance(result, dict):
        raise TypeError("platform scoring workflow returned a non-dict result")
    return ScoreGenerationWorkflowResult.model_validate(result)


def platform_scoring_workflow_id(score_attempt_id: str) -> str:
    return f"{WORKFLOW_ID_PREFIX}:{score_attempt_id}"


def _start_score_generation_workflow_handle(
    *,
    database_url: str,
    generation_run_id: str,
    score_attempt_index: int,
    scoring_profile_id: str,
    scoring_profile_version: str,
    parser_profile_id: str,
    parser_version: str,
    timeout_seconds: float,
    dataset_name: str,
    dataset_split: str,
) -> tuple[str, Any]:
    score_attempt_id = stable_score_attempt_id(
        generation_run_id=generation_run_id,
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
        parser_profile_id=parser_profile_id,
        parser_version=parser_version,
        attempt_index=score_attempt_index,
    )
    with SetWorkflowID(platform_scoring_workflow_id(score_attempt_id)):
        handle = DBOS.start_workflow(
            run_score_generation_workflow,
            database_url,
            generation_run_id,
            score_attempt_index,
            scoring_profile_id,
            scoring_profile_version,
            parser_profile_id,
            parser_version,
            timeout_seconds,
            dataset_name,
            dataset_split,
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


@DBOS.step(name=LOAD_HUMANEVAL_TASK_STEP_NAME)
def load_humaneval_task_step(
    dataset_name: str,
    dataset_split: str,
    task_id: str,
) -> dict[str, Any]:
    tasks = parse_human_eval_dataset(
        load_human_eval_rows(
            dataset_name=dataset_name,
            dataset_split=dataset_split,
        )
    )
    for task in tasks:
        if task.task_id == task_id:
            return humaneval_task_payload(task)
    raise ValueError(f"HumanEval task not found: {task_id}")


@DBOS.step(name=SCORING_STARTED_AT_STEP_NAME)
def scoring_started_at_step(score_attempt_id: str) -> str:
    return timestamp_now_iso()


def timestamp_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@DBOS.step(name=SCORE_GENERATION_STEP_NAME)
def score_generation_step(
    spec_payload: dict[str, Any],
    generation_run_payload: dict[str, Any],
    node_attempt_payloads: list[dict[str, Any]],
    task_payload: dict[str, Any],
    scoring_profile_id: str,
    scoring_profile_version: str,
    parser_profile_id: str,
    parser_version: str,
    score_attempt_index: int,
    timeout_seconds: float,
    started_at: str,
) -> dict[str, Any]:
    parser_profile = resolve_parser_profile(
        parser_profile_id=parser_profile_id,
        parser_version=parser_version,
    )
    record = score_generation_run(
        spec=PredictionSpecRecord.model_validate(spec_payload),
        generation_run=GenerationRunRecord.model_validate(
            generation_run_payload
        ),
        node_attempts=tuple(
            NodeAttemptRecord.model_validate(payload)
            for payload in node_attempt_payloads
        ),
        task=humaneval_task_from_payload(task_payload),
        parser_profile=parser_profile,
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
        score_attempt_index=score_attempt_index,
        timeout_seconds=timeout_seconds,
        started_at=datetime.fromisoformat(started_at),
    )
    return record.model_dump(mode="json")


@DBOS.step(name=PERSIST_SCORE_ATTEMPT_STEP_NAME)
def persist_score_attempt_step(
    database_url: str,
    score_attempt_payload: dict[str, Any],
) -> dict[str, Any]:
    score_attempt = ScoreAttemptRecord.model_validate(score_attempt_payload)
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            result = persist_score_attempt(
                connection,
                score_attempt=score_attempt,
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
