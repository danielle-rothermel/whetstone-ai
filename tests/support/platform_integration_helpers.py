"""Shared helpers for platform DBOS integration tests."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Row

from dr_dspy.records import PredictionSpecRecord
from tests.support.postgres_fixtures import seed_prediction_spec


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
