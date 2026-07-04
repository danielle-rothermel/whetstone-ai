"""Postgres seed helpers shared by integration tests."""

from __future__ import annotations

from typing import Any

from dbos import DBOS, SetWorkflowID
from sqlalchemy.engine import Connection

from whetstone.db import io as db_io
from whetstone.records import (
    BatchSubmitItemRecord,
    BatchSubmitOperationRecord,
    ExperimentRecord,
    PredictionSpecRecord,
)


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


def seed_batch_submit_operation(
    connection: Connection,
    operation: BatchSubmitOperationRecord,
) -> None:
    connection.execute(db_io.insert_batch_submit_operation(operation))


def seed_batch_submit_item(
    connection: Connection,
    item: BatchSubmitItemRecord,
) -> None:
    connection.execute(db_io.insert_batch_submit_item(item))


def start_test_workflow(workflow: Any, workflow_id: str, *args: Any) -> Any:
    with SetWorkflowID(workflow_id):
        handle = DBOS.start_workflow(workflow, *args)
    return handle.get_result()
