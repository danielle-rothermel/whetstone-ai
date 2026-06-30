from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from dr_dspy.migration.v0_reshape import (
    V0_SOURCE_METADATA_KEY,
    reshape_v0_direct_row,
    reshape_v0_encdec_row,
)
from dr_dspy.platform.persistence import persist_generation_result
from tests.integration.v0_sample_loader import load_v0_sample
from tests.support.postgres_fixtures import seed_prediction_spec

pytestmark = pytest.mark.integration

DIRECT_TERMINAL_FIXTURES = (
    "direct_success.json",
    "direct_generation_error.json",
)
ENC_DEC_TERMINAL_FIXTURES = (
    "encdec_success.json",
    "encdec_encoder_failure.json",
    "encdec_extraction_edge.json",
)


@pytest.mark.parametrize("fixture_name", DIRECT_TERMINAL_FIXTURES)
def test_v0_direct_outcome_import_persists_idempotently(
    app_postgres_schema,
    fixture_name: str,
) -> None:
    row = load_v0_sample(fixture_name)
    result = reshape_v0_direct_row(row)
    assert result.generation_run is not None
    assert result.node_attempts

    engine = create_engine(app_postgres_schema.database_url)
    try:
        with engine.begin() as connection:
            seed_prediction_spec(connection, result.spec)
            for _ in range(2):
                persist_generation_result(
                    connection,
                    generation_run=result.generation_run,
                    node_attempts=result.node_attempts,
                )
            run_count = connection.execute(
                text(
                    "SELECT COUNT(*) FROM dr_dspy_generation_runs "
                    "WHERE generation_run_id = :generation_run_id"
                ),
                {
                    "generation_run_id": (
                        result.generation_run.generation_run_id
                    ),
                },
            ).scalar_one()
            attempt_count = connection.execute(
                text(
                    "SELECT COUNT(*) FROM dr_dspy_node_attempts "
                    "WHERE generation_run_id = :generation_run_id"
                ),
                {
                    "generation_run_id": (
                        result.generation_run.generation_run_id
                    ),
                },
            ).scalar_one()
            summary = connection.execute(
                text(
                    "SELECT summary FROM dr_dspy_generation_runs "
                    "WHERE generation_run_id = :generation_run_id"
                ),
                {
                    "generation_run_id": (
                        result.generation_run.generation_run_id
                    ),
                },
            ).scalar_one()
    finally:
        engine.dispose()

    assert run_count == 1
    assert attempt_count == len(result.node_attempts)
    source = summary["metadata"][V0_SOURCE_METADATA_KEY]
    assert source["v0_prediction_id"] == row["prediction_id"]


@pytest.mark.parametrize("fixture_name", ENC_DEC_TERMINAL_FIXTURES)
def test_v0_encdec_outcome_import_persists_idempotently(
    app_postgres_schema,
    fixture_name: str,
) -> None:
    row = load_v0_sample(fixture_name)
    result = reshape_v0_encdec_row(row)
    assert result.generation_run is not None
    assert result.node_attempts

    engine = create_engine(app_postgres_schema.database_url)
    try:
        with engine.begin() as connection:
            seed_prediction_spec(connection, result.spec)
            persist_generation_result(
                connection,
                generation_run=result.generation_run,
                node_attempts=result.node_attempts,
            )
            status = connection.execute(
                text(
                    "SELECT status FROM dr_dspy_generation_runs "
                    "WHERE generation_run_id = :generation_run_id"
                ),
                {
                    "generation_run_id": (
                        result.generation_run.generation_run_id
                    ),
                },
            ).scalar_one()
            node_ids = [
                row_value[0]
                for row_value in connection.execute(
                    text(
                        "SELECT node_id FROM dr_dspy_node_attempts "
                        "WHERE generation_run_id = :generation_run_id "
                        "ORDER BY node_id"
                    ),
                    {
                        "generation_run_id": (
                            result.generation_run.generation_run_id
                        ),
                    },
                )
            ]
    finally:
        engine.dispose()

    assert status == result.generation_run.status.value
    expected_node_ids = sorted(
        attempt.node_id for attempt in result.node_attempts
    )
    assert node_ids == expected_node_ids
