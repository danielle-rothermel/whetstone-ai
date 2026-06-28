from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from dr_dspy import analysis, dbos_runtime, eval_reporting, worker_resources
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


def run_submit_jobs(
    backend: ExperimentBackend,
    *,
    config: EvalDbosConfig,
    experiment_name: str,
    seed: int,
    sample_count: int,
    metadata: Mapping[str, Any],
    jobs: Sequence[Any],
    score_timeout: float,
) -> None:
    backend.create_schema(config.database_url)
    backend.upsert_experiment(
        config.database_url,
        experiment_name=experiment_name,
        seed=seed,
        sample_count=sample_count,
        metadata=metadata,
    )
    inserted = backend.insert_prediction_jobs(config.database_url, jobs)
    backend.configure_runtime(config, experiment_name)
    backend.enqueue_generation_jobs(
        config.database_url, jobs, score_timeout=score_timeout
    )
    operator_log(
        f"inserted {inserted} new prediction rows",
        style="green" if inserted else "yellow",
    )
    operator_log(
        f"enqueued {len(jobs)} generation workflows",
        style="green" if jobs else "yellow",
    )


def run_status_command(
    backend: ExperimentBackend,
    *,
    database_url: str,
    experiment_name: str | None,
) -> None:
    rows = backend.fetch_status_counts(
        database_url, experiment_name=experiment_name
    )
    if not rows:
        operator_log("no prediction rows found", style="yellow")
        return
    _CONSOLE.print(
        backend.status_counts_table(rows, experiment_name=experiment_name)
    )


def run_enqueue_scores_command(
    backend: ExperimentBackend,
    *,
    config: EvalDbosConfig,
    experiment_name: str,
    limit: int,
    timeout: float,
) -> None:
    backend.create_schema(config.database_url)
    prediction_ids = backend.fetch_scoreable_prediction_ids(
        config.database_url, experiment_name=experiment_name, limit=limit
    )
    backend.configure_runtime(config, experiment_name)
    backend.enqueue_score_jobs(
        config.database_url,
        prediction_ids,
        experiment_name=experiment_name,
        timeout=timeout,
    )
    backend.mark_scoring_queued(config.database_url, prediction_ids)
    operator_log(
        eval_reporting.enqueue_scores_line(
            experiment_name=experiment_name,
            selected_count=len(prediction_ids),
            limit=limit,
            timeout=timeout,
        ),
        style=eval_reporting.enqueue_scores_style(len(prediction_ids)),
    )


def run_repair_command(
    backend: ExperimentBackend,
    *,
    config: EvalDbosConfig,
    experiment_name: str,
    generation_limit: int,
    scoring_limit: int,
    score_timeout: float,
    apply: bool,
) -> None:
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
        "score_pending": len(plan.pending_scoring_prediction_ids),
        "score_stranded": len(plan.stranded_scoring),
        "score_errors": len(plan.scoring_retry_prediction_ids),
    }
    operator_log(
        eval_reporting.repair_plan_line(
            experiment_name=experiment_name, apply=apply, **counts
        ),
        style=eval_reporting.repair_plan_style(apply=apply, **counts),
    )
    if not apply:
        if any(counts.values()):
            operator_log(
                "dry run only; rerun with --apply to reconcile statuses and "
                "enqueue fresh retry workflows",
                style="yellow",
            )
        return
    result = backend.apply_repair(
        config,
        experiment_name=experiment_name,
        generation_limit=generation_limit,
        scoring_limit=scoring_limit,
        score_timeout=score_timeout,
    )
    operator_log(
        eval_reporting.repair_apply_line(
            experiment_name=experiment_name,
            stranded_generations_marked=result.stranded_generations_marked,
            generation_retries_enqueued=result.generation_retries_enqueued,
            stranded_scoring_marked=result.stranded_scoring_marked,
            pending_scoring_enqueued=result.pending_scoring_enqueued,
            scoring_retries_enqueued=result.scoring_retries_enqueued,
            repair_token=result.repair_token,
        ),
        style="green",
    )


def run_analyze_command(
    backend: ExperimentBackend,
    *,
    database_url: str,
    experiment_name: str,
    csv_path: Path | None,
    markdown: bool,
) -> None:
    records = backend.fetch_analysis_records(
        database_url, experiment_name=experiment_name
    )
    summaries = backend.summarize_analysis_records(records)
    if markdown:
        typer.echo(
            backend.analysis_markdown(
                experiment_name=experiment_name, summaries=summaries
            ),
            nl=False,
        )
    else:
        _CONSOLE.print(
            backend.analysis_table(
                experiment_name=experiment_name, summaries=summaries
            )
        )
    if csv_path is not None:
        backend.write_analysis_csv(summaries, csv_path)
        operator_log(f"wrote {csv_path}", style="green")


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


def summarize_analysis_records[RecordT, SummaryT](
    records: Sequence[RecordT],
    *,
    group_key: Callable[[RecordT], object],
    dimension_values: Callable[[RecordT], Mapping[str, Any]],
    task_id: Callable[[RecordT], str],
    score: Callable[[RecordT], float],
    provider_cost: Callable[[RecordT], float | None],
    raw_compile_ok: Callable[[RecordT], bool | None],
    extracted_compile_ok: Callable[[RecordT], bool | None],
    raw_compression_ratio: Callable[[RecordT], float | None] = (
        lambda _record: None
    ),
    best_compression_ratio: Callable[[RecordT], float | None] = (
        lambda _record: None
    ),
    best_compression_percent_reduction: Callable[[RecordT], float | None] = (
        lambda _record: None
    ),
    summary_factory: Callable[..., SummaryT],
) -> list[SummaryT]:
    grouped: dict[object, list[RecordT]] = {}
    for record in records:
        grouped.setdefault(group_key(record), []).append(record)

    summaries: list[SummaryT] = []
    # repr keeps the sort total-ordered even when a group key tuple mixes
    # None and float (e.g. a swept budget_ratio of `none,0.5`).
    for _key, group in sorted(grouped.items(), key=lambda item: repr(item[0])):
        scores = [score(record) for record in group]
        costs: list[float] = []
        for record in group:
            cost = provider_cost(record)
            if cost is not None:
                costs.append(cost)
        by_task: dict[str, list[float]] = {}
        for record in group:
            by_task.setdefault(task_id(record), []).append(score(record))
        repetition_variances = [
            variance
            for variance in (
                analysis.variance_or_none(task_scores)
                for task_scores in by_task.values()
            )
            if variance is not None
        ]
        raw_compile_pass_count = sum(
            1 for record in group if raw_compile_ok(record) is True
        )
        extracted_compile_pass_count = sum(
            1 for record in group if extracted_compile_ok(record) is True
        )
        total_price = sum(costs) if costs else None
        raw_compression_ratios = [
            ratio
            for ratio in (raw_compression_ratio(record) for record in group)
            if ratio is not None
        ]
        compression_ratios = [
            ratio
            for ratio in (best_compression_ratio(record) for record in group)
            if ratio is not None
        ]
        compression_reductions = [
            reduction
            for reduction in (
                best_compression_percent_reduction(record) for record in group
            )
            if reduction is not None
        ]
        summaries.append(
            summary_factory(
                dimensions=dict(dimension_values(group[0])),
                sample_count=len(by_task),
                scored_count=len(group),
                total_price=total_price,
                avg_price_per_sample=(
                    total_price / len(group)
                    if total_price is not None
                    else None
                ),
                price_variance=analysis.variance_or_none(costs),
                avg_performance=analysis.average_or_none(scores) or 0.0,
                performance_variance=analysis.variance_or_none(scores),
                avg_repetition_variance=analysis.average_or_none(
                    repetition_variances
                ),
                raw_compile_pass_count=raw_compile_pass_count,
                extracted_compile_pass_count=extracted_compile_pass_count,
                extraction_lift=(
                    extracted_compile_pass_count - raw_compile_pass_count
                ),
                avg_raw_compression_ratio=analysis.average_or_none(
                    raw_compression_ratios
                ),
                avg_best_compression_ratio=analysis.average_or_none(
                    compression_ratios
                ),
                avg_best_compression_percent_reduction=(
                    analysis.average_or_none(compression_reductions)
                ),
            )
        )
    return summaries
