from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
)

from dr_dspy.dbos_runtime import (
    DBOS_ACTIVE_WORKFLOW_STATUSES,
    DBOS_FAILED_WORKFLOW_STATUSES,
    DbosWorkflowStatus,
    QueueSelection,
    connect_db,
)


class WorkerMonitorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    database_url: StrictStr
    dbos_system_database_url: StrictStr
    experiment_name: StrictStr
    prediction_table: StrictStr
    queue_selection: QueueSelection
    queue_names: tuple[StrictStr, ...]
    interval_seconds: StrictFloat
    summary_interval_seconds: StrictFloat


class WorkerQueueSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dbos_status_counts: dict[StrictStr, StrictInt] = Field(
        default_factory=dict
    )
    generation_status_counts: dict[StrictStr, StrictInt] = Field(
        default_factory=dict
    )
    scoring_status_counts: dict[StrictStr, StrictInt] = Field(
        default_factory=dict
    )

    @property
    def active_total(self) -> int:
        return sum(
            self.dbos_status_counts.get(status, 0)
            for status in DBOS_ACTIVE_WORKFLOW_STATUSES
        )

    @property
    def success_total(self) -> int:
        return self.dbos_status_counts.get(DbosWorkflowStatus.SUCCESS.value, 0)

    @property
    def failure_total(self) -> int:
        return sum(
            self.dbos_status_counts.get(status, 0)
            for status in DBOS_FAILED_WORKFLOW_STATUSES
        )


def fetch_dbos_status_counts(
    dbos_system_database_url: str, queue_names: Sequence[str]
) -> dict[str, int]:
    with connect_db(dbos_system_database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT status, COUNT(*)
                FROM dbos.workflow_status
                WHERE queue_name = ANY(%s)
                GROUP BY status
                """,
                (list(queue_names),),
            )
            rows = cur.fetchall()
    return {row[0]: row[1] for row in rows}


def validate_prediction_table_name(prediction_table: str) -> None:
    if not prediction_table.replace("_", "").isalnum():
        raise ValueError(f"unsupported prediction table: {prediction_table}")


def fetch_prediction_phase_counts(
    database_url: str,
    *,
    prediction_table: str,
    status_column: str,
    experiment_name: str,
) -> dict[str, int]:
    validate_prediction_table_name(prediction_table)
    if status_column not in {"generation_status", "scoring_status"}:
        raise ValueError(f"unsupported status column: {status_column}")
    with connect_db(database_url) as conn:
        with conn.cursor() as cur:
            query = f"""
                SELECT {status_column}, COUNT(*)
                FROM {prediction_table}
                WHERE experiment_name = %s
                GROUP BY {status_column}
                """
            cur.execute(
                cast(Any, query),
                (experiment_name,),
            )
            rows = cur.fetchall()
    return {row[0]: row[1] for row in rows}


def fetch_worker_queue_snapshot(
    config: WorkerMonitorConfig,
) -> WorkerQueueSnapshot:
    generation_counts: dict[str, int] = {}
    scoring_counts: dict[str, int] = {}
    if config.queue_selection in (
        QueueSelection.GENERATION,
        QueueSelection.BOTH,
    ):
        generation_counts = fetch_prediction_phase_counts(
            config.database_url,
            prediction_table=config.prediction_table,
            status_column="generation_status",
            experiment_name=config.experiment_name,
        )
    if config.queue_selection in (QueueSelection.SCORING, QueueSelection.BOTH):
        scoring_counts = fetch_prediction_phase_counts(
            config.database_url,
            prediction_table=config.prediction_table,
            status_column="scoring_status",
            experiment_name=config.experiment_name,
        )
    return WorkerQueueSnapshot(
        dbos_status_counts=fetch_dbos_status_counts(
            config.dbos_system_database_url,
            config.queue_names,
        ),
        generation_status_counts=generation_counts,
        scoring_status_counts=scoring_counts,
    )


def count_for_status(counts: Mapping[str, int], status: str) -> int:
    return int(counts.get(status, 0))


def count_for_phase_status(counts: Mapping[str, int], status: str) -> int:
    if not counts:
        return -1
    return int(counts.get(status, 0))


def worker_monitor_counts(
    snapshot: WorkerQueueSnapshot,
    *,
    completed_since_start: int,
    failures_since_start: int,
) -> dict[str, int]:
    return {
        "active": snapshot.active_total,
        "enqueued": count_for_status(
            snapshot.dbos_status_counts,
            DbosWorkflowStatus.ENQUEUED.value,
        ),
        "pending": count_for_status(
            snapshot.dbos_status_counts,
            DbosWorkflowStatus.PENDING.value,
        ),
        "delayed": count_for_status(
            snapshot.dbos_status_counts,
            DbosWorkflowStatus.DELAYED.value,
        ),
        "completed": completed_since_start,
        "errors": failures_since_start,
        "gen_pending": count_for_phase_status(
            snapshot.generation_status_counts, "pending"
        ),
        "gen_started": count_for_phase_status(
            snapshot.generation_status_counts, "started"
        ),
        "gen_done": count_for_phase_status(
            snapshot.generation_status_counts, "generated"
        ),
        "gen_errors": count_for_phase_status(
            snapshot.generation_status_counts, "generation_error"
        ),
        "score_pending": count_for_phase_status(
            snapshot.scoring_status_counts, "pending"
        ),
        "score_queued": count_for_phase_status(
            snapshot.scoring_status_counts, "queued"
        ),
        "score_started": count_for_phase_status(
            snapshot.scoring_status_counts, "started"
        ),
        "score_done": count_for_phase_status(
            snapshot.scoring_status_counts, "scored"
        ),
        "score_errors": count_for_phase_status(
            snapshot.scoring_status_counts, "score_error"
        ),
    }


def format_worker_count(value: int, *, width: int) -> str:
    if value < 0:
        return f"{'-':>{width}}"
    return f"{value:>{width}}"


def worker_monitor_line(
    snapshot: WorkerQueueSnapshot,
    *,
    was_active: bool | None,
    initial_success_total: int,
    initial_failure_total: int,
    force_summary: bool,
) -> str | None:
    is_active = snapshot.active_total > 0
    changed_state = was_active is None or is_active != was_active
    if not changed_state and not force_summary:
        return None

    completed_since_start = max(
        snapshot.success_total - initial_success_total,
        0,
    )
    failures_since_start = max(
        snapshot.failure_total - initial_failure_total,
        0,
    )
    counts = worker_monitor_counts(
        snapshot,
        completed_since_start=completed_since_start,
        failures_since_start=failures_since_start,
    )
    state = "Queue Active" if is_active else "Queue Empty"
    return (
        f"{state:<12} | "
        f"active={format_worker_count(counts['active'], width=4)} | "
        f"enqueued={format_worker_count(counts['enqueued'], width=4)} | "
        f"pending={format_worker_count(counts['pending'], width=4)} | "
        f"delayed={format_worker_count(counts['delayed'], width=4)} | "
        f"completed={format_worker_count(counts['completed'], width=4)} | "
        f"errors={format_worker_count(counts['errors'], width=4)} | "
        "gen "
        f"pend={format_worker_count(counts['gen_pending'], width=4)} "
        f"start={format_worker_count(counts['gen_started'], width=4)} "
        f"done={format_worker_count(counts['gen_done'], width=4)} "
        f"err={format_worker_count(counts['gen_errors'], width=4)} | "
        "score "
        f"pend={format_worker_count(counts['score_pending'], width=4)} "
        f"queue={format_worker_count(counts['score_queued'], width=4)} "
        f"start={format_worker_count(counts['score_started'], width=4)} "
        f"done={format_worker_count(counts['score_done'], width=4)} "
        f"err={format_worker_count(counts['score_errors'], width=4)}"
    )


def worker_monitor_style(snapshot: WorkerQueueSnapshot) -> str:
    if snapshot.active_total > 0:
        return "green"
    return "yellow"


def run_worker_monitor(
    config: WorkerMonitorConfig,
    stop_event: threading.Event,
    *,
    operator_log: Callable[..., None],
    emit_worker_detail_log: Callable[[str, Mapping[str, object]], None],
) -> None:
    was_active: bool | None = None
    initial_success_total: int | None = None
    initial_failure_total: int | None = None
    last_summary_at = 0.0
    last_error: str | None = None
    while not stop_event.is_set():
        try:
            snapshot = fetch_worker_queue_snapshot(config)
            if initial_success_total is None:
                initial_success_total = snapshot.success_total
                initial_failure_total = snapshot.failure_total
            force_summary = (
                time.monotonic() - last_summary_at
                >= config.summary_interval_seconds
            )
            line = worker_monitor_line(
                snapshot,
                was_active=was_active,
                initial_success_total=initial_success_total,
                initial_failure_total=initial_failure_total or 0,
                force_summary=force_summary,
            )
            if line is not None:
                operator_log(line, style=worker_monitor_style(snapshot))
                last_summary_at = time.monotonic()
            was_active = snapshot.active_total > 0
            last_error = None
        except Exception as e:
            error = repr(e)
            emit_worker_detail_log("worker_monitor_error", {"error": error})
            if error != last_error:
                operator_log(
                    f"worker monitor error: {error}; retrying",
                    style="red",
                )
                last_error = error
        stop_event.wait(config.interval_seconds)


def start_worker_monitor(
    config: WorkerMonitorConfig,
    stop_event: threading.Event,
    *,
    operator_log: Callable[..., None],
    emit_worker_detail_log: Callable[[str, Mapping[str, object]], None],
) -> threading.Thread:
    thread = threading.Thread(
        target=run_worker_monitor,
        kwargs={
            "config": config,
            "stop_event": stop_event,
            "operator_log": operator_log,
            "emit_worker_detail_log": emit_worker_detail_log,
        },
        name=f"worker-monitor-{config.queue_selection.value}",
        daemon=True,
    )
    thread.start()
    return thread
