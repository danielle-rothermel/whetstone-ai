"""Postgres seed helpers shared by integration tests."""

from __future__ import annotations

from typing import Any

from dbos import DBOS, SetWorkflowID
from sqlalchemy.engine import Connection

from dr_dspy.db import io as db_io
from dr_dspy.records import ExperimentRecord, PredictionSpecRecord


def seed_experiment(
    connection: Connection,
    *,
    experiment_name: str = "exp",
) -> None:
    record = ExperimentRecord(
        experiment_name=experiment_name,
        config_metadata={"seed": "seed"},
    )
    connection.execute(db_io.insert_experiment(record))


def seed_prediction_spec(
    connection: Connection,
    spec: PredictionSpecRecord,
    *,
    seed_experiment_row: bool = True,
) -> None:
    if seed_experiment_row:
        seed_experiment(connection, experiment_name=spec.experiment_name)
    connection.execute(db_io.insert_prediction_spec(spec))


def start_test_workflow(workflow: Any, workflow_id: str, *args: Any) -> Any:
    with SetWorkflowID(workflow_id):
        handle = DBOS.start_workflow(workflow, *args)
    return handle.get_result()
