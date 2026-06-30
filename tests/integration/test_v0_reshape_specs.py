from __future__ import annotations

import uuid
from typing import Any

import pytest
from sqlalchemy import create_engine, text

from dr_dspy.migration.v0_reshape import (
    reshape_v0_direct_row,
    reshape_v0_encdec_row,
)
from dr_dspy.platform import graph_workflow
from dr_dspy.platform.graph_workflow import run_prediction_graph_workflow_once
from dr_dspy.platform.node_execution import NodeStepResult
from dr_dspy.records import GenerationRunStatus, PredictionSpecRecord
from tests.integration.dbos_test_workflows import (
    integration_load_spec_workflow,
)
from tests.integration.v0_sample_loader import load_v0_sample
from tests.support.platform_workflow_fixtures import step_success
from tests.support.postgres_fixtures import (
    seed_prediction_spec,
    start_test_workflow,
)

pytestmark = pytest.mark.integration

ALL_FIXTURES = (
    ("direct_success.json", reshape_v0_direct_row),
    ("direct_generation_error.json", reshape_v0_direct_row),
    ("encdec_success.json", reshape_v0_encdec_row),
    ("encdec_encoder_failure.json", reshape_v0_encdec_row),
    ("encdec_extraction_edge.json", reshape_v0_encdec_row),
)


def _mock_lm_for_reshaped_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_execute_lm_node(
        *,
        spec: Any,
        node: Any,
        node_inputs: dict[str, Any],
        client_factory: Any = None,
        provider_caller: Any = None,
        raise_retryable: bool = False,
    ) -> NodeStepResult:
        if node.id == "encoder":
            return step_success(node, "reshaped description")
        if node.id == "decoder":
            description = node_inputs.get("description", "")
            return step_success(node, f"def run(): return {description!r}")
        return step_success(node, f"reshaped {node_inputs.get('prompt', '')}")

    monkeypatch.setattr(
        graph_workflow,
        "execute_lm_node",
        fake_execute_lm_node,
    )


@pytest.mark.parametrize(("fixture_name", "reshape"), ALL_FIXTURES)
def test_reshaped_spec_loads_through_dbos_step(
    app_postgres_schema,
    reset_dbos,
    fixture_name: str,
    reshape: Any,
) -> None:
    row = load_v0_sample(fixture_name)
    result = reshape(row)
    engine = create_engine(app_postgres_schema.database_url)
    try:
        with engine.begin() as connection:
            seed_prediction_spec(connection, result.spec)
    finally:
        engine.dispose()

    payload = start_test_workflow(
        integration_load_spec_workflow,
        f"test-v0-load:{uuid.uuid4().hex}",
        app_postgres_schema.database_url,
        result.spec.prediction_id,
    )
    loaded = PredictionSpecRecord.model_validate(payload)

    assert loaded.prediction_id == result.spec.prediction_id
    assert loaded.graph.layout == result.spec.graph.layout


@pytest.mark.parametrize(("fixture_name", "reshape"), ALL_FIXTURES)
def test_reshaped_spec_runs_through_platform_workflow_with_mocked_lm(
    app_postgres_schema,
    reset_dbos,
    monkeypatch: pytest.MonkeyPatch,
    fixture_name: str,
    reshape: Any,
) -> None:
    row = load_v0_sample(fixture_name)
    result = reshape(row)
    engine = create_engine(app_postgres_schema.database_url)
    try:
        with engine.begin() as connection:
            seed_prediction_spec(connection, result.spec)
    finally:
        engine.dispose()

    _mock_lm_for_reshaped_specs(monkeypatch)
    generation_run_id = run_prediction_graph_workflow_once(
        app_postgres_schema.database_url,
        result.spec.prediction_id,
        attempt_index=1,
    )

    engine = create_engine(app_postgres_schema.database_url)
    try:
        with engine.connect() as connection:
            status = connection.execute(
                text(
                    "SELECT status FROM dr_dspy_generation_runs "
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
    finally:
        engine.dispose()

    assert status == GenerationRunStatus.SUCCESS.value
    assert attempt_count >= 1
