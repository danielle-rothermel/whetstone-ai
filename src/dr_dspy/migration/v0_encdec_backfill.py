"""Live enc-dec v0 table backfill into v1 append-only records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

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
    selected_v0_rows: StrictInt = 0
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


def terminal_rows_select_sql(*, limit: int | None) -> str:
    statuses = ", ".join(
        f"'{status}'" for status in sorted(V0_TERMINAL_GENERATION_STATUSES)
    )
    query = (
        f"SELECT * FROM {V0_ENC_DEC_TABLE} "
        f"WHERE generation_status IN ({statuses}) "
        "ORDER BY generation_status ASC, prediction_id ASC"
    )
    if limit is not None:
        query += " LIMIT :limit"
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


def fetch_v0_encdec_terminal_rows(
    connection: Connection,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    result = connection.execute(
        text(terminal_rows_select_sql(limit=limit)),
        {"limit": limit} if limit is not None else {},
    )
    return [dict(row) for row in result.mappings()]


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


def backfill_v0_encdec_rows(
    connection: Connection,
    rows: Sequence[Mapping[str, Any]],
    *,
    dry_run: bool,
    target_experiment_name: str | None = None,
) -> V0EncdecBackfillResult:
    result = V0EncdecBackfillResult(
        dry_run=dry_run,
        target_experiment_name=target_experiment_name,
        selected_v0_rows=len(rows),
    )
    experiments_seen: set[str] = set()
    for row in rows:
        mapped_row = with_target_experiment(row, target_experiment_name)
        try:
            reshaped = reshape_v0_encdec_row(mapped_row)
        except Exception as error:
            result = result.model_copy(
                update={
                    "reshape_failures": result.reshape_failures + 1,
                    "first_reshape_error": (
                        result.first_reshape_error or str(error)
                    ),
                }
            )
            continue

        result = result.model_copy(
            update={"reshaped_specs": result.reshaped_specs + 1}
        )
        if dry_run:
            continue

        insert_counts = insert_reshaped_encdec(
            connection,
            reshaped,
            experiments_seen=experiments_seen,
        )
        result = result.model_copy(
            update={
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


def run_v0_encdec_backfill(
    engine: Engine,
    *,
    dry_run: bool,
    limit: int | None = None,
    target_experiment_name: str | None = None,
) -> V0EncdecBackfillResult:
    if dry_run:
        with engine.connect() as connection:
            non_terminal = count_non_terminal_v0_rows(connection)
            rows = fetch_v0_encdec_terminal_rows(connection, limit=limit)
            outcome = backfill_v0_encdec_rows(
                connection,
                rows,
                dry_run=True,
                target_experiment_name=target_experiment_name,
            )
            return outcome.model_copy(
                update={"non_terminal_v0_rows": non_terminal}
            )

    with engine.begin() as connection:
        non_terminal = count_non_terminal_v0_rows(connection)
        rows = fetch_v0_encdec_terminal_rows(connection, limit=limit)
        outcome = backfill_v0_encdec_rows(
            connection,
            rows,
            dry_run=False,
            target_experiment_name=target_experiment_name,
        )
        return outcome.model_copy(
            update={"non_terminal_v0_rows": non_terminal}
        )
