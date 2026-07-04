from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, text

from dr_dspy.migration.v0_encdec_backfill import (
    V0_ENC_DEC_TABLE,
    V0EncdecBackfillResult,
    backfill_v0_encdec_rows,
    insert_reshaped_encdec,
    merge_backfill_results,
    run_v0_encdec_backfill,
    run_v0_encdec_backfill_chunked,
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


def test_terminal_rows_select_sql_supports_offset_paging() -> None:
    sql = terminal_rows_select_sql(limit=10, offset=20)
    assert "LIMIT :limit" in sql
    assert "OFFSET :offset" in sql


def test_merge_backfill_results_accumulates_counts() -> None:
    first = V0EncdecBackfillResult(
        dry_run=True,
        selected_v0_rows=2,
        processed_v0_rows=2,
        reshaped_specs=2,
        specs_inserted=2,
        experiments_touched=("exp-a",),
        chunks_processed=1,
    )
    second = V0EncdecBackfillResult(
        dry_run=True,
        selected_v0_rows=1,
        processed_v0_rows=1,
        reshaped_specs=1,
        specs_already_present=1,
        reshape_failures=1,
        first_reshape_error="boom",
        experiments_touched=("exp-b",),
    )
    merged = merge_backfill_results(first, second)
    assert merged.chunks_processed == 2
    assert merged.processed_v0_rows == 3
    assert merged.reshaped_specs == 3
    assert merged.specs_inserted == 2
    assert merged.specs_already_present == 1
    assert merged.reshape_failures == 1
    assert merged.first_reshape_error == "boom"
    assert merged.experiments_touched == ("exp-a", "exp-b")


def test_run_v0_encdec_backfill_chunked_processes_multiple_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = load_v0_sample("encdec_success.json")
    pages = [[row, row], [row], []]
    page_calls: list[tuple[int, int]] = []
    chunk_calls = 0

    class FakeConnection:
        def __enter__(self) -> FakeConnection:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

    class FakeEngine:
        def connect(self) -> FakeConnection:
            return FakeConnection()

        def begin(self) -> FakeConnection:
            return FakeConnection()

    def fake_count(connection: FakeConnection) -> int:
        return 0

    def fake_fetch(
        connection: FakeConnection,
        *,
        limit: int,
        offset: int,
    ) -> list[dict[str, Any]]:
        page_calls.append((limit, offset))
        return pages.pop(0)

    def fake_process_chunk(
        engine: FakeEngine,
        rows: list[dict[str, Any]],
        *,
        dry_run: bool,
        target_experiment_name: str | None,
        reshape_workers: int,
    ) -> V0EncdecBackfillResult:
        nonlocal chunk_calls
        chunk_calls += 1
        return V0EncdecBackfillResult(
            dry_run=dry_run,
            selected_v0_rows=len(rows),
            processed_v0_rows=len(rows),
            reshaped_specs=len(rows),
            reshape_workers=reshape_workers,
        )

    monkeypatch.setattr(
        "dr_dspy.migration.v0_encdec_backfill.count_non_terminal_v0_rows",
        fake_count,
    )
    monkeypatch.setattr(
        "dr_dspy.migration.v0_encdec_backfill.count_v0_encdec_terminal_rows",
        lambda connection, limit=None: 3,
    )
    monkeypatch.setattr(
        "dr_dspy.migration.v0_encdec_backfill.fetch_v0_encdec_terminal_rows_page",
        fake_fetch,
    )
    monkeypatch.setattr(
        "dr_dspy.migration.v0_encdec_backfill._process_backfill_chunk",
        fake_process_chunk,
    )

    result = run_v0_encdec_backfill_chunked(
        cast(Any, FakeEngine()),
        dry_run=True,
        chunk_size=2,
        limit=None,
        target_experiment_name=None,
        reshape_workers=4,
    )

    assert page_calls == [(2, 0), (2, 2)]
    assert chunk_calls == 2
    assert result.chunks_processed == 2
    assert result.processed_v0_rows == 3
    assert result.selected_v0_rows == 3
    assert result.reshape_workers == 4
    assert result.chunk_size == 2


def test_run_v0_encdec_backfill_chunked_limit_caps_total_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = load_v0_sample("encdec_success.json")

    class FakeConnection:
        def __enter__(self) -> FakeConnection:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

    class FakeEngine:
        def connect(self) -> FakeConnection:
            return FakeConnection()

        def begin(self) -> FakeConnection:
            return FakeConnection()

    def fake_fetch(
        connection: FakeConnection,
        *,
        limit: int,
        offset: int,
    ) -> list[dict[str, Any]]:
        return [row] * limit

    monkeypatch.setattr(
        "dr_dspy.migration.v0_encdec_backfill.count_non_terminal_v0_rows",
        lambda connection: 0,
    )
    monkeypatch.setattr(
        "dr_dspy.migration.v0_encdec_backfill.count_v0_encdec_terminal_rows",
        lambda connection, limit=None: 3,
    )
    monkeypatch.setattr(
        "dr_dspy.migration.v0_encdec_backfill.fetch_v0_encdec_terminal_rows_page",
        fake_fetch,
    )
    monkeypatch.setattr(
        "dr_dspy.migration.v0_encdec_backfill._process_backfill_chunk",
        lambda engine, rows, **kwargs: V0EncdecBackfillResult(
            dry_run=True,
            selected_v0_rows=len(rows),
            processed_v0_rows=len(rows),
            reshaped_specs=len(rows),
        ),
    )

    result = run_v0_encdec_backfill(
        cast(Any, FakeEngine()),
        dry_run=True,
        limit=3,
        chunk_size=2,
    )

    assert result.processed_v0_rows == 3
    assert result.chunks_processed == 2


def test_run_v0_encdec_backfill_chunked_dry_run_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = load_v0_sample("encdec_success.json")
    begin_calls = 0

    class FakeConnection:
        def __enter__(self) -> FakeConnection:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

    class FakeEngine:
        def connect(self) -> FakeConnection:
            return FakeConnection()

        def begin(self) -> FakeConnection:
            nonlocal begin_calls
            begin_calls += 1
            return FakeConnection()

    monkeypatch.setattr(
        "dr_dspy.migration.v0_encdec_backfill.count_non_terminal_v0_rows",
        lambda connection: 0,
    )
    monkeypatch.setattr(
        "dr_dspy.migration.v0_encdec_backfill.count_v0_encdec_terminal_rows",
        lambda connection, limit=None: 1,
    )
    monkeypatch.setattr(
        "dr_dspy.migration.v0_encdec_backfill.fetch_v0_encdec_terminal_rows_page",
        lambda connection, *, limit, offset: [row],
    )

    result = run_v0_encdec_backfill(
        cast(Any, FakeEngine()),
        dry_run=True,
        chunk_size=1,
        limit=1,
    )

    assert begin_calls == 0
    assert result.dry_run is True
    assert result.reshaped_specs == 1
    assert result.specs_inserted == 0


def test_reshape_v0_encdec_rows_uses_parallel_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = load_v0_sample("encdec_success.json")
    executor = MagicMock()
    executor.__enter__.return_value = executor
    monkeypatch.setattr(
        "dr_dspy.migration.v0_encdec_backfill.ThreadPoolExecutor",
        lambda max_workers: executor,
    )
    executor.submit.side_effect = lambda fn, *args: MagicMock(
        result=lambda: fn(*args)
    )
    monkeypatch.setattr(
        "dr_dspy.migration.v0_encdec_backfill.as_completed",
        lambda futures: futures,
    )

    from dr_dspy.migration.v0_encdec_backfill import reshape_v0_encdec_rows

    reshaped, failures, first_error = reshape_v0_encdec_rows(
        [row, row],
        target_experiment_name=None,
        reshape_workers=2,
    )

    assert len(reshaped) == 2
    assert failures == 0
    assert first_error is None
    executor.submit.assert_called()


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
