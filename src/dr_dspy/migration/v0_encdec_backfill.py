"""Live enc-dec v0 table backfill into v1 append-only records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection, Engine

from dr_dspy.db import io, schema
from dr_dspy.migration.v0_reshape import (
    V0_TERMINAL_GENERATION_STATUSES,
    V0ReshapeResult,
    reshape_v0_encdec_row,
)
from dr_dspy.platform.persistence import (
    _NODE_ATTEMPT_NULLABLE_JSONB_COLUMNS,
    _postgres_insert_values,
    idempotent_insert_generation_run,
)
from dr_dspy.platform.submission import (
    bulk_insert_prediction_specs,
    idempotent_insert_experiment,
)
from dr_dspy.records import ExperimentRecord

if TYPE_CHECKING:
    from dr_dspy.platform.progress_log import OperationProgress

V0_ENC_DEC_TABLE = "dr_dspy_encdec_eval_predictions"
V0_BACKFILL_SOURCE = "v0_encdec_backfill"


class V0EncdecInsertCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    specs_inserted: StrictInt = 0
    specs_already_present: StrictInt = 0
    runs_inserted: StrictInt = 0
    runs_already_present: StrictInt = 0
    node_attempts_inserted: StrictInt = 0
    node_attempts_already_present: StrictInt = 0


class V0EncdecBackfillResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dry_run: bool
    target_experiment_name: StrictStr | None = None
    chunk_size: StrictInt | None = None
    reshape_workers: StrictInt = 1
    chunks_processed: StrictInt = 0
    selected_v0_rows: StrictInt = 0
    processed_v0_rows: StrictInt = 0
    non_terminal_v0_rows: StrictInt = 0
    reshaped_specs: StrictInt = 0
    reshape_failures: StrictInt = 0
    first_reshape_error: StrictStr | None = None
    specs_inserted: StrictInt = 0
    specs_already_present: StrictInt = 0
    runs_inserted: StrictInt = 0
    runs_already_present: StrictInt = 0
    node_attempts_inserted: StrictInt = 0
    node_attempts_already_present: StrictInt = 0
    experiments_touched: tuple[StrictStr, ...] = Field(default_factory=tuple)


def with_target_experiment(
    row: Mapping[str, Any],
    target_experiment_name: str | None,
) -> dict[str, Any]:
    if target_experiment_name is None:
        return dict(row)
    return {**row, "experiment_name": target_experiment_name}


def terminal_generation_statuses() -> frozenset[str]:
    return V0_TERMINAL_GENERATION_STATUSES


def terminal_rows_select_sql(
    *,
    limit: int | None,
    offset: int | None = None,
) -> str:
    statuses = ", ".join(
        f"'{status}'" for status in sorted(V0_TERMINAL_GENERATION_STATUSES)
    )
    # Offset paging is acceptable here: the v0 table is read-only for backfill.
    query = (
        f"SELECT * FROM {V0_ENC_DEC_TABLE} "
        f"WHERE generation_status IN ({statuses}) "
        "ORDER BY generation_status ASC, prediction_id ASC"
    )
    if limit is not None:
        query += " LIMIT :limit"
    if offset is not None:
        query += " OFFSET :offset"
    return query


def count_non_terminal_v0_rows(connection: Connection) -> int:
    statuses = ", ".join(
        f"'{status}'" for status in sorted(V0_TERMINAL_GENERATION_STATUSES)
    )
    return int(
        connection.execute(
            text(
                f"SELECT COUNT(*) FROM {V0_ENC_DEC_TABLE} "
                f"WHERE generation_status NOT IN ({statuses}) "
                "OR generation_status IS NULL"
            )
        ).scalar_one()
    )


def count_v0_encdec_terminal_rows(
    connection: Connection,
    *,
    limit: int | None = None,
) -> int:
    statuses = ", ".join(
        f"'{status}'" for status in sorted(V0_TERMINAL_GENERATION_STATUSES)
    )
    total = int(
        connection.execute(
            text(
                f"SELECT COUNT(*) FROM {V0_ENC_DEC_TABLE} "
                f"WHERE generation_status IN ({statuses})"
            )
        ).scalar_one()
    )
    if limit is not None:
        return min(total, limit)
    return total


def fetch_v0_encdec_terminal_rows(
    connection: Connection,
    *,
    limit: int | None = None,
    offset: int | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, int] = {}
    if limit is not None:
        params["limit"] = limit
    if offset is not None:
        params["offset"] = offset
    result = connection.execute(
        text(terminal_rows_select_sql(limit=limit, offset=offset)),
        params,
    )
    return [dict(row) for row in result.mappings()]


def fetch_v0_encdec_terminal_rows_page(
    connection: Connection,
    *,
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    return fetch_v0_encdec_terminal_rows(
        connection,
        limit=limit,
        offset=offset,
    )


def validate_backfill_request(
    *,
    chunk_size: int | None,
    limit: int | None,
    reshape_workers: int,
) -> None:
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("chunk_size must be positive when provided")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided")
    if reshape_workers < 1:
        raise ValueError("reshape_workers must be positive")


def merge_backfill_results(
    accumulated: V0EncdecBackfillResult,
    chunk_result: V0EncdecBackfillResult,
) -> V0EncdecBackfillResult:
    experiments = set(accumulated.experiments_touched) | set(
        chunk_result.experiments_touched
    )
    return accumulated.model_copy(
        update={
            "chunks_processed": accumulated.chunks_processed + 1,
            "processed_v0_rows": (
                accumulated.processed_v0_rows + chunk_result.selected_v0_rows
            ),
            "reshaped_specs": (
                accumulated.reshaped_specs + chunk_result.reshaped_specs
            ),
            "reshape_failures": (
                accumulated.reshape_failures + chunk_result.reshape_failures
            ),
            "first_reshape_error": (
                accumulated.first_reshape_error
                or chunk_result.first_reshape_error
            ),
            "specs_inserted": (
                accumulated.specs_inserted + chunk_result.specs_inserted
            ),
            "specs_already_present": (
                accumulated.specs_already_present
                + chunk_result.specs_already_present
            ),
            "runs_inserted": (
                accumulated.runs_inserted + chunk_result.runs_inserted
            ),
            "runs_already_present": (
                accumulated.runs_already_present
                + chunk_result.runs_already_present
            ),
            "node_attempts_inserted": (
                accumulated.node_attempts_inserted
                + chunk_result.node_attempts_inserted
            ),
            "node_attempts_already_present": (
                accumulated.node_attempts_already_present
                + chunk_result.node_attempts_already_present
            ),
            "experiments_touched": tuple(sorted(experiments)),
        }
    )


def _reshape_single_row(
    row: Mapping[str, Any],
    target_experiment_name: str | None,
) -> tuple[V0ReshapeResult | None, str | None]:
    mapped_row = with_target_experiment(row, target_experiment_name)
    try:
        return reshape_v0_encdec_row(mapped_row), None
    except Exception as error:
        return None, str(error)


def reshape_v0_encdec_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_experiment_name: str | None,
    reshape_workers: int,
) -> tuple[list[V0ReshapeResult], int, str | None]:
    validate_backfill_request(
        chunk_size=1,
        limit=None,
        reshape_workers=reshape_workers,
    )
    if reshape_workers == 1:
        reshaped: list[V0ReshapeResult] = []
        failures = 0
        first_error: str | None = None
        for row in rows:
            result, error = _reshape_single_row(row, target_experiment_name)
            if error is not None:
                failures += 1
                first_error = first_error or error
                continue
            assert result is not None
            reshaped.append(result)
        return reshaped, failures, first_error

    reshaped = []
    failures = 0
    first_error: str | None = None
    with ThreadPoolExecutor(max_workers=reshape_workers) as executor:
        futures = {
            executor.submit(
                _reshape_single_row,
                row,
                target_experiment_name,
            ): row
            for row in rows
        }
        for future in as_completed(futures):
            result, error = future.result()
            if error is not None:
                failures += 1
                first_error = first_error or error
                continue
            assert result is not None
            reshaped.append(result)
    return reshaped, failures, first_error


def insert_reshaped_encdec(
    connection: Connection,
    result: V0ReshapeResult,
    *,
    experiments_seen: set[str],
) -> V0EncdecInsertCounts:
    counts = V0EncdecInsertCounts()
    experiment_name = result.spec.experiment_name
    if experiment_name not in experiments_seen:
        connection.execute(
            idempotent_insert_experiment(
                ExperimentRecord(
                    experiment_name=experiment_name,
                    config_metadata={"source": V0_BACKFILL_SOURCE},
                )
            )
        )
        experiments_seen.add(experiment_name)

    inserted_spec_ids = bulk_insert_prediction_specs(connection, [result.spec])
    if result.spec.prediction_id in inserted_spec_ids:
        counts = counts.model_copy(update={"specs_inserted": 1})
    else:
        counts = counts.model_copy(update={"specs_already_present": 1})

    if result.generation_run is None:
        return counts

    run_insert = connection.execute(
        idempotent_insert_generation_run(result.generation_run).returning(
            schema.generation_runs.c.generation_run_id
        )
    )
    if run_insert.first() is not None:
        counts = counts.model_copy(update={"runs_inserted": 1})
    else:
        counts = counts.model_copy(update={"runs_already_present": 1})

    for node_attempt in result.node_attempts:
        attempt_insert = connection.execute(
            insert(schema.node_attempts)
            .values(
                _postgres_insert_values(
                    io.node_attempt_row(node_attempt),
                    nullable_jsonb_columns=_NODE_ATTEMPT_NULLABLE_JSONB_COLUMNS,
                )
            )
            .on_conflict_do_nothing(index_elements=["node_attempt_id"])
            .returning(schema.node_attempts.c.node_attempt_id)
        )
        if attempt_insert.first() is not None:
            counts = counts.model_copy(
                update={
                    "node_attempts_inserted": counts.node_attempts_inserted + 1
                }
            )
        else:
            counts = counts.model_copy(
                update={
                    "node_attempts_already_present": (
                        counts.node_attempts_already_present + 1
                    )
                }
            )
    return counts


def insert_reshaped_encdec_rows(
    connection: Connection,
    reshaped_rows: Sequence[V0ReshapeResult],
) -> V0EncdecBackfillResult:
    result = V0EncdecBackfillResult(dry_run=False)
    experiments_seen: set[str] = set()
    for reshaped in reshaped_rows:
        insert_counts = insert_reshaped_encdec(
            connection,
            reshaped,
            experiments_seen=experiments_seen,
        )
        result = result.model_copy(
            update={
                "reshaped_specs": result.reshaped_specs + 1,
                "specs_inserted": (
                    result.specs_inserted + insert_counts.specs_inserted
                ),
                "specs_already_present": (
                    result.specs_already_present
                    + insert_counts.specs_already_present
                ),
                "runs_inserted": (
                    result.runs_inserted + insert_counts.runs_inserted
                ),
                "runs_already_present": (
                    result.runs_already_present
                    + insert_counts.runs_already_present
                ),
                "node_attempts_inserted": (
                    result.node_attempts_inserted
                    + insert_counts.node_attempts_inserted
                ),
                "node_attempts_already_present": (
                    result.node_attempts_already_present
                    + insert_counts.node_attempts_already_present
                ),
                "experiments_touched": tuple(sorted(experiments_seen)),
            }
        )
    return result


def backfill_v0_encdec_rows(
    connection: Connection,
    rows: Sequence[Mapping[str, Any]],
    *,
    dry_run: bool,
    target_experiment_name: str | None = None,
    reshape_workers: int = 1,
) -> V0EncdecBackfillResult:
    reshaped_rows, failures, first_error = reshape_v0_encdec_rows(
        rows,
        target_experiment_name=target_experiment_name,
        reshape_workers=reshape_workers,
    )
    result = V0EncdecBackfillResult(
        dry_run=dry_run,
        target_experiment_name=target_experiment_name,
        reshape_workers=reshape_workers,
        selected_v0_rows=len(rows),
        processed_v0_rows=len(rows),
        reshaped_specs=len(reshaped_rows),
        reshape_failures=failures,
        first_reshape_error=first_error,
    )
    if dry_run:
        return result

    insert_result = insert_reshaped_encdec_rows(connection, reshaped_rows)
    return result.model_copy(
        update={
            "specs_inserted": insert_result.specs_inserted,
            "specs_already_present": insert_result.specs_already_present,
            "runs_inserted": insert_result.runs_inserted,
            "runs_already_present": insert_result.runs_already_present,
            "node_attempts_inserted": insert_result.node_attempts_inserted,
            "node_attempts_already_present": (
                insert_result.node_attempts_already_present
            ),
            "experiments_touched": insert_result.experiments_touched,
        }
    )


def _backfill_progress_metrics(
    result: V0EncdecBackfillResult,
    *,
    phase: str,
    chunk_index: int | None = None,
    total_candidates: int | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "phase": phase,
        "processed": result.processed_v0_rows,
        "reshaped": result.reshaped_specs,
        "failures": result.reshape_failures,
        "inserted": result.specs_inserted,
        "already_present": result.specs_already_present,
    }
    if total_candidates is not None:
        metrics["total_candidates"] = total_candidates
    if chunk_index is not None:
        metrics["chunk"] = chunk_index
    if result.chunks_processed:
        metrics["chunks"] = result.chunks_processed
    if result.chunk_size is not None:
        metrics["chunk_size"] = result.chunk_size
    return metrics


def _report_backfill_chunk_progress(
    progress: OperationProgress | None,
    *,
    accumulated: V0EncdecBackfillResult,
    chunk_index: int,
    offset: int,
    chunk_row_count: int,
    total_candidates: int | None = None,
) -> None:
    if progress is None:
        return
    progress.update(
        **_backfill_progress_metrics(
            accumulated,
            phase="chunked",
            chunk_index=chunk_index,
            total_candidates=total_candidates,
        )
    )
    progress.event(
        "chunk complete",
        {
            "offset": offset,
            "chunk_rows": chunk_row_count,
            "processed": accumulated.processed_v0_rows,
            "reshaped": accumulated.reshaped_specs,
            "inserted": accumulated.specs_inserted,
            "already_present": accumulated.specs_already_present,
            "failures": accumulated.reshape_failures,
            "chunk": chunk_index,
            "chunks": accumulated.chunks_processed,
            **(
                {"total_candidates": total_candidates}
                if total_candidates is not None
                else {}
            ),
        },
    )


def _process_backfill_chunk(
    engine: Engine,
    rows: Sequence[Mapping[str, Any]],
    *,
    dry_run: bool,
    target_experiment_name: str | None,
    reshape_workers: int,
) -> V0EncdecBackfillResult:
    reshaped_rows, failures, first_error = reshape_v0_encdec_rows(
        rows,
        target_experiment_name=target_experiment_name,
        reshape_workers=reshape_workers,
    )
    result = V0EncdecBackfillResult(
        dry_run=dry_run,
        target_experiment_name=target_experiment_name,
        reshape_workers=reshape_workers,
        selected_v0_rows=len(rows),
        processed_v0_rows=len(rows),
        reshaped_specs=len(reshaped_rows),
        reshape_failures=failures,
        first_reshape_error=first_error,
    )
    if dry_run:
        return result

    with engine.begin() as connection:
        insert_result = insert_reshaped_encdec_rows(connection, reshaped_rows)
    return result.model_copy(
        update={
            "specs_inserted": insert_result.specs_inserted,
            "specs_already_present": insert_result.specs_already_present,
            "runs_inserted": insert_result.runs_inserted,
            "runs_already_present": insert_result.runs_already_present,
            "node_attempts_inserted": insert_result.node_attempts_inserted,
            "node_attempts_already_present": (
                insert_result.node_attempts_already_present
            ),
            "experiments_touched": insert_result.experiments_touched,
        }
    )


def run_v0_encdec_backfill_chunked(
    engine: Engine,
    *,
    dry_run: bool,
    chunk_size: int,
    limit: int | None,
    target_experiment_name: str | None,
    reshape_workers: int,
    progress: OperationProgress | None = None,
) -> V0EncdecBackfillResult:
    validate_backfill_request(
        chunk_size=chunk_size,
        limit=limit,
        reshape_workers=reshape_workers,
    )
    with engine.connect() as connection:
        non_terminal = count_non_terminal_v0_rows(connection)
        total_candidates = count_v0_encdec_terminal_rows(
            connection, limit=limit
        )

    accumulated = V0EncdecBackfillResult(
        dry_run=dry_run,
        target_experiment_name=target_experiment_name,
        chunk_size=chunk_size,
        reshape_workers=reshape_workers,
    )
    if progress is not None:
        progress.update(
            phase="chunked",
            chunk_size=chunk_size,
            reshape_workers=reshape_workers,
            dry_run=dry_run,
            limit=limit,
            total_candidates=total_candidates,
        )
    offset = 0
    chunk_index = 0
    while limit is None or offset < limit:
        page_limit = (
            chunk_size if limit is None else min(chunk_size, limit - offset)
        )
        with engine.connect() as connection:
            rows = fetch_v0_encdec_terminal_rows_page(
                connection,
                limit=page_limit,
                offset=offset,
            )
        if not rows:
            break

        chunk_result = _process_backfill_chunk(
            engine,
            rows,
            dry_run=dry_run,
            target_experiment_name=target_experiment_name,
            reshape_workers=reshape_workers,
        )
        accumulated = merge_backfill_results(accumulated, chunk_result)
        chunk_index += 1
        _report_backfill_chunk_progress(
            progress,
            accumulated=accumulated,
            chunk_index=chunk_index,
            offset=offset,
            chunk_row_count=len(rows),
            total_candidates=total_candidates,
        )
        offset += len(rows)
        if len(rows) < page_limit:
            break

    return accumulated.model_copy(
        update={
            "non_terminal_v0_rows": non_terminal,
            "selected_v0_rows": accumulated.processed_v0_rows,
        }
    )


def run_v0_encdec_backfill(
    engine: Engine,
    *,
    dry_run: bool,
    limit: int | None = None,
    target_experiment_name: str | None = None,
    chunk_size: int | None = None,
    reshape_workers: int = 1,
    progress: OperationProgress | None = None,
) -> V0EncdecBackfillResult:
    validate_backfill_request(
        chunk_size=chunk_size,
        limit=limit,
        reshape_workers=reshape_workers,
    )
    if progress is not None:
        with engine.connect() as connection:
            total_candidates = count_v0_encdec_terminal_rows(
            connection, limit=limit
        )
        progress.event(
            "started",
            {
                "dry_run": dry_run,
                "limit": limit,
                "chunk_size": chunk_size,
                "reshape_workers": reshape_workers,
                "total_candidates": total_candidates,
            },
        )
    else:
        total_candidates = None
    if chunk_size is not None:
        outcome = run_v0_encdec_backfill_chunked(
            engine,
            dry_run=dry_run,
            chunk_size=chunk_size,
            limit=limit,
            target_experiment_name=target_experiment_name,
            reshape_workers=reshape_workers,
            progress=progress,
        )
        if progress is not None:
            progress.complete(
                _backfill_progress_metrics(
                    outcome,
                    phase="done",
                    total_candidates=total_candidates,
                )
            )
        return outcome

    if dry_run:
        with engine.connect() as connection:
            non_terminal = count_non_terminal_v0_rows(connection)
            if total_candidates is None:
                total_candidates = count_v0_encdec_terminal_rows(
                    connection,
                    limit=limit,
                )
            rows = fetch_v0_encdec_terminal_rows(connection, limit=limit)
            if progress is not None:
                progress.update(
                    phase="processing",
                    total_candidates=total_candidates,
                    processed=0,
                )
            outcome = backfill_v0_encdec_rows(
                connection,
                rows,
                dry_run=True,
                target_experiment_name=target_experiment_name,
                reshape_workers=reshape_workers,
            )
            outcome = outcome.model_copy(
                update={
                    "non_terminal_v0_rows": non_terminal,
                    "processed_v0_rows": outcome.selected_v0_rows,
                    "reshape_workers": reshape_workers,
                }
            )
            if progress is not None:
                progress.complete(
                    _backfill_progress_metrics(
                        outcome,
                        phase="done",
                        total_candidates=total_candidates,
                    )
                )
            return outcome

    with engine.begin() as connection:
        non_terminal = count_non_terminal_v0_rows(connection)
        if total_candidates is None:
            total_candidates = count_v0_encdec_terminal_rows(
            connection, limit=limit
        )
        rows = fetch_v0_encdec_terminal_rows(connection, limit=limit)
        if progress is not None:
            progress.update(
                phase="processing",
                total_candidates=total_candidates,
                processed=0,
            )
        outcome = backfill_v0_encdec_rows(
            connection,
            rows,
            dry_run=False,
            target_experiment_name=target_experiment_name,
            reshape_workers=reshape_workers,
        )
        outcome = outcome.model_copy(
            update={
                "non_terminal_v0_rows": non_terminal,
                "processed_v0_rows": outcome.selected_v0_rows,
                "reshape_workers": reshape_workers,
            }
        )
        if progress is not None:
            progress.complete(
                _backfill_progress_metrics(
                    outcome,
                    phase="done",
                    total_candidates=total_candidates,
                )
            )
        return outcome
