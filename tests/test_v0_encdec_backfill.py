from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from dr_dspy.migration.v0_encdec_backfill import (
    V0_ENC_DEC_TABLE,
    backfill_v0_encdec_rows,
    insert_reshaped_encdec,
    terminal_generation_statuses,
    terminal_rows_select_sql,
    with_target_experiment,
)
from dr_dspy.migration.v0_reshape import reshape_v0_encdec_row
from tests.integration.v0_sample_loader import load_v0_sample


def test_terminal_generation_statuses_match_reshape_contract() -> None:
    assert terminal_generation_statuses() == frozenset(
        {"generated", "generation_error"}
    )


def test_terminal_rows_select_sql_filters_terminal_statuses_only() -> None:
    sql = terminal_rows_select_sql(limit=5)
    assert V0_ENC_DEC_TABLE in sql
    assert "'generated'" in sql
    assert "'generation_error'" in sql
    assert "ORDER BY generation_status ASC, prediction_id ASC" in sql
    assert "LIMIT :limit" in sql


def test_with_target_experiment_override_does_not_mutate_input() -> None:
    row = load_v0_sample("encdec_success.json")
    original_name = row["experiment_name"]
    mapped = with_target_experiment(row, "v0_encdec_backfill_smoke_20260630")

    assert row["experiment_name"] == original_name
    assert mapped["experiment_name"] == "v0_encdec_backfill_smoke_20260630"
    result = reshape_v0_encdec_row(mapped)
    assert result.spec.experiment_name == "v0_encdec_backfill_smoke_20260630"


@pytest.mark.integration
def test_backfill_v0_encdec_rows_dry_run_does_not_write(
    app_postgres_schema,
) -> None:
    row = load_v0_sample("encdec_success.json")
    engine = create_engine(app_postgres_schema.database_url)
    try:
        with engine.connect() as connection:
            before_specs = connection.execute(
                text("SELECT COUNT(*) FROM dr_dspy_prediction_specs")
            ).scalar_one()
            outcome = backfill_v0_encdec_rows(
                connection,
                [row],
                dry_run=True,
                target_experiment_name="v0_encdec_backfill_dry_run",
            )
            after_specs = connection.execute(
                text("SELECT COUNT(*) FROM dr_dspy_prediction_specs")
            ).scalar_one()

        assert outcome.dry_run is True
        assert outcome.selected_v0_rows == 1
        assert outcome.reshaped_specs == 1
        assert outcome.reshape_failures == 0
        assert outcome.specs_inserted == 0
        assert before_specs == after_specs
    finally:
        engine.dispose()


@pytest.mark.integration
def test_insert_reshaped_encdec_is_idempotent(app_postgres_schema) -> None:
    row = load_v0_sample("encdec_success.json")
    reshaped = reshape_v0_encdec_row(
        with_target_experiment(row, "v0_encdec_backfill_idempotent")
    )
    assert reshaped.generation_run is not None

    engine = create_engine(app_postgres_schema.database_url)
    try:
        with engine.begin() as connection:
            experiments_seen: set[str] = set()
            first = insert_reshaped_encdec(
                connection,
                reshaped,
                experiments_seen=experiments_seen,
            )
            second = insert_reshaped_encdec(
                connection,
                reshaped,
                experiments_seen=experiments_seen,
            )

        assert first.specs_inserted == 1
        assert first.runs_inserted == 1
        assert first.node_attempts_inserted == 2
        assert second.specs_already_present == 1
        assert second.runs_already_present == 1
        assert second.node_attempts_already_present == 2

        with engine.connect() as connection:
            run_count = connection.execute(
                text(
                    "SELECT COUNT(*) FROM dr_dspy_generation_runs "
                    "WHERE generation_run_id = :generation_run_id"
                ),
                {
                    "generation_run_id": (
                        reshaped.generation_run.generation_run_id
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
                        reshaped.generation_run.generation_run_id
                    ),
                },
            ).scalar_one()

        assert run_count == 1
        assert attempt_count == 2
    finally:
        engine.dispose()
