"""Shared helpers for platform DBOS integration tests."""

from __future__ import annotations

import time
from datetime import datetime

from dbos import DBOS
from dr_platform import (
    DBOS_ACTIVE_WORKFLOW_STATUSES,
    DBOS_FAILED_WORKFLOW_STATUSES,
    DbosWorkflowStatus,
)
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Row

from tests.support.postgres_fixtures import seed_prediction_spec
from whetstone.platform.scoring_workflow_state import (
    dbos_workflow_status_value,
)
from whetstone.records import (
    GenerationRunRecord,
    NodeAttemptRecord,
    PredictionSpecRecord,
)


def seed_spec(database_url: str, spec: PredictionSpecRecord) -> None:
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            seed_prediction_spec(connection, spec)
    finally:
        engine.dispose()


class WorkflowRunSnapshot:
    run_status: str
    attempt_status: str | None
    attempt_started_at: datetime | None
    attempt_completed_at: datetime | None
    node_count: int
    attempt_node_id: str | None

    def __init__(self, row: Row[tuple[object, ...]], node_count: int) -> None:
        self.run_status = str(row[0])
        self.attempt_status = str(row[1]) if row[1] is not None else None
        self.attempt_started_at = (
            row[2] if isinstance(row[2], datetime) else None
        )
        self.attempt_completed_at = (
            row[3] if isinstance(row[3], datetime) else None
        )
        self.attempt_node_id = str(row[4]) if row[4] is not None else None
        self.node_count = node_count


def fetch_workflow_run_snapshot(
    database_url: str,
    generation_run_id: str,
) -> WorkflowRunSnapshot:
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT gr.status, na.status, na.started_at, "
                    "na.completed_at, na.node_id "
                    "FROM dr_dspy_generation_runs gr "
                    "LEFT JOIN dr_dspy_node_attempts na "
                    "ON na.generation_run_id = gr.generation_run_id "
                    "WHERE gr.generation_run_id = :generation_run_id "
                    "ORDER BY na.node_id NULLS LAST "
                    "LIMIT 1"
                ),
                {"generation_run_id": generation_run_id},
            ).one()
            node_count = connection.execute(
                text(
                    "SELECT COUNT(*) FROM dr_dspy_node_attempts "
                    "WHERE generation_run_id = :generation_run_id"
                ),
                {"generation_run_id": generation_run_id},
            ).scalar_one()
    finally:
        engine.dispose()
    return WorkflowRunSnapshot(row, int(node_count))


def wait_for_workflow_result(
    workflow_id: str,
    *,
    timeout_s: float = 30.0,
    poll_interval_s: float = 0.1,
) -> object:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status_value = dbos_workflow_status_value(
            DBOS.get_workflow_status(workflow_id)
        )
        if status_value == DbosWorkflowStatus.SUCCESS.value:
            return DBOS.retrieve_workflow(workflow_id).get_result()
        if status_value in DBOS_FAILED_WORKFLOW_STATUSES:
            raise RuntimeError(
                f"workflow {workflow_id} failed with status {status_value}"
            )
        if (
            status_value not in DBOS_ACTIVE_WORKFLOW_STATUSES
            and status_value is not None
        ):
            return DBOS.retrieve_workflow(workflow_id).get_result()
        time.sleep(poll_interval_s)
    raise TimeoutError(
        f"timed out waiting for workflow {workflow_id} after {timeout_s}s"
    )


def fetch_batch_submit_operation_status(
    database_url: str,
    operation_key: str,
) -> str:
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            return str(
                connection.execute(
                    text(
                        "SELECT status FROM dr_dspy_batch_submit_operations "
                        "WHERE operation_key = :operation_key"
                    ),
                    {"operation_key": operation_key},
                ).scalar_one()
            )
    finally:
        engine.dispose()


def count_generation_runs(
    database_url: str,
    generation_run_id: str,
) -> int:
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            return int(
                connection.execute(
                    text(
                        "SELECT COUNT(*) FROM dr_dspy_generation_runs "
                        "WHERE generation_run_id = :generation_run_id"
                    ),
                    {"generation_run_id": generation_run_id},
                ).scalar_one()
            )
    finally:
        engine.dispose()


def fetch_node_attempts(
    database_url: str,
    generation_run_id: str,
) -> list[tuple[str, str]]:
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT node_id, status FROM dr_dspy_node_attempts "
                    "WHERE generation_run_id = :generation_run_id "
                    "ORDER BY node_id"
                ),
                {"generation_run_id": generation_run_id},
            ).all()
    finally:
        engine.dispose()
    return [(str(node_id), str(status)) for node_id, status in rows]


def seed_generation_run_with_nodes(
    connection: Connection,
    *,
    generation_run: GenerationRunRecord,
    node_attempts: tuple[NodeAttemptRecord, ...] = (),
) -> None:
    from whetstone.platform.persistence import persist_generation_result

    persist_generation_result(
        connection,
        generation_run=generation_run,
        node_attempts=node_attempts,
    )


def seed_scoring_target(
    database_url: str,
    *,
    spec: PredictionSpecRecord,
    generation_run: GenerationRunRecord,
    node_attempts: tuple[NodeAttemptRecord, ...] = (),
) -> None:
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            seed_prediction_spec(connection, spec)
            seed_generation_run_with_nodes(
                connection,
                generation_run=generation_run,
                node_attempts=node_attempts,
            )
    finally:
        engine.dispose()


class ScoreAttemptSnapshot:
    status: str
    score: float | None
    insert_count: int

    def __init__(
        self,
        row: Row[tuple[object, ...]],
        insert_count: int,
    ) -> None:
        self.status = str(row[0])
        self.score = float(row[1]) if row[1] is not None else None
        self.insert_count = insert_count


def count_score_attempts(
    database_url: str,
    score_attempt_id: str,
) -> int:
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            return int(
                connection.execute(
                    text(
                        "SELECT COUNT(*) FROM dr_dspy_score_attempts "
                        "WHERE score_attempt_id = :score_attempt_id"
                    ),
                    {"score_attempt_id": score_attempt_id},
                ).scalar_one()
            )
    finally:
        engine.dispose()


def fetch_score_attempt_snapshot(
    database_url: str,
    score_attempt_id: str,
) -> ScoreAttemptSnapshot | None:
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT status, score FROM dr_dspy_score_attempts "
                    "WHERE score_attempt_id = :score_attempt_id"
                ),
                {"score_attempt_id": score_attempt_id},
            ).one_or_none()
            if row is None:
                return None
            insert_count = connection.execute(
                text(
                    "SELECT COUNT(*) FROM dr_dspy_score_attempts "
                    "WHERE score_attempt_id = :score_attempt_id"
                ),
                {"score_attempt_id": score_attempt_id},
            ).scalar_one()
    finally:
        engine.dispose()
    return ScoreAttemptSnapshot(row, int(insert_count))
