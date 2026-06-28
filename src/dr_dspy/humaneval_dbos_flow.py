from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rich.console import Console

from dr_dspy import dbos_runtime, eval_reporting, worker_resources
from dr_dspy.dbos_runtime import EvalDbosConfig, QueueSelection
from dr_dspy.eval_logging import operator_log as _console_operator_log
from dr_dspy.experiment_backend import ExperimentBackend
from dr_dspy.lm_utils import stable_json
from dr_dspy.worker_monitor import WorkerMonitorConfig

_CONSOLE = Console()


def operator_log(line: str, *, style: str | None = None) -> None:
    _console_operator_log(_CONSOLE, line, style=style)


def parse_float_csv(raw: str, *, value_name: str = "value") -> list[float]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"at least one {value_name} is required")
    return [float(value) for value in values]


def stable_prediction_id_from_dimensions(
    *,
    experiment_name: str,
    task_id: str,
    dimensions: Mapping[str, Any],
    repetition_seed: int,
    digest_length: int | None = None,
) -> str:
    raw = stable_json(
        {
            "experiment_name": experiment_name,
            "task_id": task_id,
            **dict(dimensions),
            "repetition_seed": repetition_seed,
        }
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if digest_length is None:
        return digest
    return digest[:digest_length]


def run_repair_command(
    backend: ExperimentBackend,
    *,
    config: EvalDbosConfig,
    experiment_name: str,
    generation_limit: int,
    scoring_limit: int,
    score_timeout: float,
) -> None:
    backend.create_schema(config.database_url)
    plan = backend.build_repair_plan(
        config.database_url,
        dbos_system_database_url=config.dbos_system_database_url,
        experiment_name=experiment_name,
        generation_limit=generation_limit,
        scoring_limit=scoring_limit,
    )
    counts = {
        "gen_stranded": len(plan.stranded_generations),
        "gen_errors": len(plan.generation_retry_prediction_ids),
        "gen_recoverable_errors": (
            plan.generation_retry_summary.recoverable_count
        ),
        "gen_excluded_errors": plan.generation_retry_summary.excluded_count,
        "score_pending": len(plan.pending_scoring_prediction_ids),
        "score_stranded": len(plan.stranded_scoring),
        "score_errors": len(plan.scoring_retry_prediction_ids),
        "score_recoverable_errors": (
            plan.scoring_retry_summary.recoverable_count
        ),
        "score_excluded_errors": plan.scoring_retry_summary.excluded_count,
    }
    operator_log(
        eval_reporting.repair_plan_line(
            experiment_name=experiment_name, apply=False, **counts
        ),
        style=eval_reporting.repair_plan_style(apply=False, **counts),
    )
    if (
        counts["gen_stranded"]
        or counts["gen_errors"]
        or counts["score_pending"]
        or counts["score_stranded"]
        or counts["score_errors"]
    ):
        operator_log(
            "dry run only; rerun with --apply to run durable repair batches",
            style="yellow",
        )


def run_worker_command(
    backend: ExperimentBackend,
    *,
    config: EvalDbosConfig,
    experiment_name: str,
    queue: QueueSelection,
    open_file_limit: str,
    log_file: Path | None,
    monitor: bool,
    monitor_interval: float,
    monitor_summary_interval: float,
    db_pool_max_size: str,
) -> None:
    backend.create_schema(config.database_url)
    pool_config = backend.configure_pooled_worker_runtime(
        config,
        experiment_name=experiment_name,
        queue=queue,
        raw_db_pool_max_size=db_pool_max_size,
    )
    resource_budget = worker_resources.build_worker_resource_budget(
        queue=queue,
        generation_concurrency=config.generation_concurrency,
        scoring_concurrency=config.scoring_concurrency,
        db_pool_max_size=pool_config.max_size,
    )
    requested_open_file_limit = (
        worker_resources.resolve_open_file_limit_request(
            open_file_limit,
            budget=resource_budget,
        )
    )
    file_limit = dbos_runtime.raise_open_file_limit(requested_open_file_limit)
    operator_log(
        worker_resources.resource_budget_line(resource_budget),
        style="cyan",
    )
    operator_log(
        dbos_runtime.open_file_limit_line(file_limit),
        style=dbos_runtime.open_file_limit_style(file_limit),
    )
    operator_log(
        f"{'DB Pool':<14} | max_size={pool_config.max_size:>5} | "
        f"urls={len(dbos_runtime.DB_POOLS):>2}",
        style="cyan",
    )
    if queue in (
        dbos_runtime.QueueSelection.GENERATION,
        dbos_runtime.QueueSelection.BOTH,
    ):
        http_client = worker_resources.configure_openrouter_client(
            max_connections=resource_budget.http_max_connections,
        )
        del http_client
        http_config = worker_resources.openrouter_client_config()
        if http_config is not None:
            operator_log(
                worker_resources.http_client_line(http_config),
                style="cyan",
            )
    backend.configure_runtime(config, experiment_name, queue=queue)
    resolved_log_file = backend.resolve_worker_log_path(
        experiment_name=experiment_name,
        queue=queue,
        log_file=log_file,
    )
    backend.configure_worker_file_logging(resolved_log_file)
    selected_queue_names = backend.queue_names_for_selection(
        queue, experiment_name=experiment_name
    )
    operator_log(
        f"worker listening on {queue.value} queue(s): "
        f"{', '.join(selected_queue_names)}",
        style="cyan",
    )
    operator_log(f"detailed worker log: {resolved_log_file}", style="cyan")

    stop_event = threading.Event()
    halt_event = threading.Event()
    monitor_thread: threading.Thread | None = None
    if monitor:
        monitor_config = WorkerMonitorConfig(
            database_url=config.database_url,
            dbos_system_database_url=config.dbos_system_database_url,
            experiment_name=experiment_name,
            prediction_table=backend.prediction_table,
            queue_selection=queue,
            queue_names=selected_queue_names,
            interval_seconds=monitor_interval,
            summary_interval_seconds=monitor_summary_interval,
        )
        monitor_thread = backend.start_worker_monitor(
            monitor_config, stop_event, halt_event
        )
        operator_log(
            "worker monitor enabled: "
            f"interval={monitor_interval}s, "
            f"summary_interval={monitor_summary_interval}s",
            style="cyan",
        )
    try:
        while not halt_event.is_set():
            time.sleep(1.0)
        operator_log("worker self-halt requested by monitor", style="red")
        raise SystemExit(2)
    except KeyboardInterrupt:
        operator_log("worker stopping", style="cyan")
    finally:
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=1.0)
        worker_resources.close_openrouter_client()
        dbos_runtime.destroy_dbos_runtime()
        dbos_runtime.close_db_connection_pools()
