from __future__ import annotations

import uuid
from datetime import UTC, timedelta

import pytest
from sqlalchemy import create_engine, text

from dr_dspy.graph import GraphSpec
from dr_dspy.platform.graph_workflow import execute_prediction_graph
from dr_dspy.records import PredictionSpecRecord
from tests.integration.dbos_test_workflows import (
    integration_load_spec_workflow,
    integration_persist_result_workflow,
)
from tests.support.platform_workflow_fixtures import (
    NOW,
    direct_node,
    prediction_spec,
    step_success,
)
from tests.support.postgres_fixtures import (
    seed_prediction_spec,
    start_test_workflow,
)

pytestmark = pytest.mark.integration


def test_load_prediction_spec_step_round_trips_through_dbos(
    app_postgres_schema,
    reset_dbos,
) -> None:
    graph = GraphSpec(nodes=(direct_node(),), terminal_node_id="direct")
    spec = prediction_spec(graph)
    engine = create_engine(app_postgres_schema.database_url)
    try:
        with engine.begin() as connection:
            seed_prediction_spec(connection, spec)
    finally:
        engine.dispose()

    workflow_id = f"test-load-spec:{uuid.uuid4().hex}"
    payload = start_test_workflow(
        integration_load_spec_workflow,
        workflow_id,
        app_postgres_schema.database_url,
        spec.prediction_id,
    )
    loaded = PredictionSpecRecord.model_validate(payload)

    loaded_dump = loaded.model_dump(mode="json", exclude={"created_at"})
    spec_dump = spec.model_dump(mode="json", exclude={"created_at"})
    assert loaded_dump == spec_dump
    assert loaded.created_at.astimezone(UTC) == spec.created_at.astimezone(UTC)


def test_persist_generation_result_step_writes_rows_and_is_idempotent(
    app_postgres_schema,
    reset_dbos,
) -> None:
    graph = GraphSpec(nodes=(direct_node(),), terminal_node_id="direct")
    spec = prediction_spec(graph)
    generation_run_id = "integration-run-1"
    later = NOW + timedelta(seconds=5)
    execution = execute_prediction_graph(
        spec=spec,
        attempt_index=0,
        generation_run_id=generation_run_id,
        started_at=NOW,
        completed_at=later,
        run_node_step=lambda step_spec, node, inputs: step_success(
            node,
            "ok",
            completed_at=NOW + timedelta(seconds=1),
        ),
    )
    engine = create_engine(app_postgres_schema.database_url)
    try:
        with engine.begin() as connection:
            seed_prediction_spec(connection, spec)
    finally:
        engine.dispose()

    spec_payload = spec.model_dump(mode="json")
    graph_payload = execution.graph_result.model_dump(mode="json")
    node_payloads = [
        step_result.model_dump(mode="json")
        for step_result in execution.node_step_results
    ]
    args = (
        app_postgres_schema.database_url,
        spec_payload,
        generation_run_id,
        0,
        graph_payload,
        node_payloads,
        NOW.isoformat(),
        later.isoformat(),
    )

    for attempt in range(2):
        workflow_id = f"test-persist:{uuid.uuid4().hex}:{attempt}"
        start_test_workflow(
            integration_persist_result_workflow,
            workflow_id,
            *args,
        )

    engine = create_engine(app_postgres_schema.database_url)
    try:
        with engine.connect() as connection:
            run_count = connection.execute(
                text(
                    "SELECT COUNT(*) FROM dr_dspy_generation_runs "
                    "WHERE generation_run_id = :generation_run_id"
                ),
                {"generation_run_id": generation_run_id},
            ).scalar_one()
            attempt_count = connection.execute(
                text(
                    "SELECT COUNT(*) FROM dr_dspy_node_attempts "
                    "WHERE generation_run_id = :generation_run_id"
                ),
                {"generation_run_id": generation_run_id},
            ).scalar_one()
            status = connection.execute(
                text(
                    "SELECT status FROM dr_dspy_generation_runs "
                    "WHERE generation_run_id = :generation_run_id"
                ),
                {"generation_run_id": generation_run_id},
            ).scalar_one()
    finally:
        engine.dispose()

    assert run_count == 1
    assert attempt_count == 1
    assert status == "success"
