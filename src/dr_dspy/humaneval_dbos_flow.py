from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import typer
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr
from rich.console import Console

from dr_dspy import analysis
from dr_dspy.dbos_runtime import EvalDbosConfig, QueueSelection
from dr_dspy.lm_utils import stable_json
from dr_dspy.worker_monitor import WorkerMonitorConfig


class AnalysisSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: StrictStr
    temperature: float
    sample_count: StrictInt
    scored_count: StrictInt
    total_price: float | None
    avg_price_per_sample: float | None
    price_variance: float | None
    avg_performance: float
    performance_variance: float | None
    avg_repetition_variance: float | None
    raw_compile_pass_count: StrictInt
    extracted_compile_pass_count: StrictInt
    extraction_lift: StrictInt
    avg_best_compression_ratio: float | None = None
    avg_best_compression_percent_reduction: float | None = None


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


def run_generation_workflow[GenerationResultT](
    *,
    database_url: str,
    prediction_id: str,
    experiment_name: str,
    score_timeout: float,
    generate_prediction: Callable[[str, str], GenerationResultT],
    record_generation_success: Callable[[str, GenerationResultT], None],
    record_generation_error: Callable[[str, str, str], None],
    enqueue_score: Callable[[str, str, str, float], None],
    mark_scoring_queued: Callable[[str, str], None],
) -> str:
    try:
        result = generate_prediction(database_url, prediction_id)
        record_generation_success(database_url, result)
    except Exception as e:
        record_generation_error(database_url, prediction_id, repr(e))
        return "generation_error"
    enqueue_score(database_url, prediction_id, experiment_name, score_timeout)
    mark_scoring_queued(database_url, prediction_id)
    return "generated"


def run_score_workflow[ScoreResultT](
    *,
    database_url: str,
    prediction_id: str,
    timeout: float,
    score_prediction: Callable[[str, str, float], ScoreResultT],
    record_score_success: Callable[[str, ScoreResultT], None],
    record_score_error: Callable[[str, str, str], None],
) -> str:
    try:
        result = score_prediction(database_url, prediction_id, timeout)
        record_score_success(database_url, result)
        return "scored"
    except Exception as e:
        record_score_error(database_url, prediction_id, repr(e))
        return "score_error"


def run_generate_prediction_step[JobT, GenerationResultT](
    *,
    database_url: str,
    prediction_id: str,
    mark_generation_started: Callable[[str, str], None],
    fetch_prediction_job: Callable[[str, str], JobT],
    prediction_context_from_job: Callable[[JobT], Any],
    emit_prediction_log_event: Callable[..., None],
    generate_code_for_job: Callable[[JobT], GenerationResultT],
) -> GenerationResultT:
    mark_generation_started(database_url, prediction_id)
    job = fetch_prediction_job(database_url, prediction_id)
    emit_prediction_log_event(
        "generation_started",
        prediction_context_from_job(job),
    )
    return generate_code_for_job(job)


def run_record_generation_success_step[GenerationResultT](
    *,
    database_url: str,
    result: GenerationResultT,
    prediction_id: str,
    fetch_prediction_log_context: Callable[[str, str], Any],
    emit_prediction_log_event: Callable[..., None],
    success_extra: Callable[[GenerationResultT], Mapping[str, Any]],
    record_generation_success: Callable[[str, GenerationResultT], None],
) -> None:
    context = fetch_prediction_log_context(database_url, prediction_id)
    emit_prediction_log_event(
        "generation_succeeded",
        context,
        extra=success_extra(result),
    )
    record_generation_success(database_url, result)


def run_record_generation_error_step(
    *,
    database_url: str,
    prediction_id: str,
    error: str,
    fetch_prediction_log_context: Callable[[str, str], Any],
    emit_prediction_log_event: Callable[..., None],
    record_generation_error: Callable[[str, str, str], None],
) -> None:
    context = fetch_prediction_log_context(database_url, prediction_id)
    emit_prediction_log_event(
        "generation_failed",
        context,
        extra={"error": error},
    )
    record_generation_error(database_url, prediction_id, error)


def run_mark_scoring_queued_step(
    *,
    database_url: str,
    prediction_id: str,
    fetch_prediction_log_context: Callable[[str, str], Any],
    emit_prediction_log_event: Callable[..., None],
    mark_scoring_queued: Callable[[str, Sequence[str]], int],
) -> None:
    context = fetch_prediction_log_context(database_url, prediction_id)
    emit_prediction_log_event("scoring_enqueued", context)
    mark_scoring_queued(database_url, [prediction_id])


def run_score_prediction_step[ScoreResultT](
    *,
    database_url: str,
    prediction_id: str,
    timeout: float,
    mark_scoring_started: Callable[[str, str], None],
    fetch_prediction_log_context: Callable[[str, str], Any],
    emit_prediction_log_event: Callable[..., None],
    score_generated_prediction: Callable[[str, str, float], ScoreResultT],
) -> ScoreResultT:
    mark_scoring_started(database_url, prediction_id)
    context = fetch_prediction_log_context(database_url, prediction_id)
    emit_prediction_log_event(
        "scoring_started",
        context,
        extra={"timeout": timeout},
    )
    return score_generated_prediction(database_url, prediction_id, timeout)


def run_record_score_success_step[ScoreResultT](
    *,
    database_url: str,
    result: ScoreResultT,
    prediction_id: str,
    score: float,
    error: str | None,
    fetch_prediction_log_context: Callable[[str, str], Any],
    emit_prediction_log_event: Callable[..., None],
    record_score_success: Callable[[str, ScoreResultT], None],
) -> None:
    context = fetch_prediction_log_context(database_url, prediction_id)
    emit_prediction_log_event(
        "scoring_succeeded",
        context,
        extra={"score": score, "scoring_error": error},
    )
    record_score_success(database_url, result)


def run_record_score_error_step(
    *,
    database_url: str,
    prediction_id: str,
    error: str,
    fetch_prediction_log_context: Callable[[str, str], Any],
    emit_prediction_log_event: Callable[..., None],
    record_score_error: Callable[[str, str, str], None],
) -> None:
    context = fetch_prediction_log_context(database_url, prediction_id)
    emit_prediction_log_event(
        "scoring_failed",
        context,
        extra={"error": error},
    )
    record_score_error(database_url, prediction_id, error)


def run_submit_jobs[JobT](
    *,
    config: EvalDbosConfig,
    experiment_name: str,
    seed: int,
    sample_count: int,
    metadata: Mapping[str, Any],
    jobs: Sequence[JobT],
    score_timeout: float,
    create_schema: Callable[[str], None],
    upsert_experiment: Callable[..., None],
    insert_prediction_jobs: Callable[[str, Sequence[JobT]], int],
    configure_runtime: Callable[[EvalDbosConfig, str], None],
    enqueue_generation_jobs: Callable[[str, Sequence[JobT], float], None],
    operator_log: Callable[..., None],
) -> None:
    create_schema(config.database_url)
    upsert_experiment(
        config.database_url,
        experiment_name=experiment_name,
        seed=seed,
        sample_count=sample_count,
        metadata=metadata,
    )
    inserted = insert_prediction_jobs(config.database_url, jobs)
    configure_runtime(config, experiment_name)
    enqueue_generation_jobs(config.database_url, jobs, score_timeout)
    operator_log(
        f"inserted {inserted} new prediction rows",
        style="green" if inserted else "yellow",
    )
    operator_log(
        f"enqueued {len(jobs)} generation workflows",
        style="green" if jobs else "yellow",
    )


def run_status_command(
    *,
    database_url: str,
    experiment_name: str | None,
    fetch_status_counts: Callable[..., Sequence[Mapping[str, Any]]],
    status_counts_table: Callable[..., object],
    console: Console,
    operator_log: Callable[..., None],
) -> None:
    rows = fetch_status_counts(database_url, experiment_name=experiment_name)
    if not rows:
        operator_log("no prediction rows found", style="yellow")
        return
    console.print(status_counts_table(rows, experiment_name=experiment_name))


def run_enqueue_scores_command(
    *,
    config: EvalDbosConfig,
    experiment_name: str,
    limit: int,
    timeout: float,
    create_schema: Callable[[str], None],
    fetch_scoreable_prediction_ids: Callable[..., list[str]],
    configure_runtime: Callable[[EvalDbosConfig, str], None],
    enqueue_score_jobs: Callable[[str, Sequence[str], str, float], None],
    mark_scoring_queued: Callable[[str, Sequence[str]], int],
    enqueue_scores_line: Callable[..., str],
    enqueue_scores_style: Callable[[int], str],
    operator_log: Callable[..., None],
) -> None:
    create_schema(config.database_url)
    prediction_ids = fetch_scoreable_prediction_ids(
        config.database_url, experiment_name=experiment_name, limit=limit
    )
    configure_runtime(config, experiment_name)
    enqueue_score_jobs(
        config.database_url,
        prediction_ids,
        experiment_name,
        timeout,
    )
    mark_scoring_queued(config.database_url, prediction_ids)
    operator_log(
        enqueue_scores_line(
            experiment_name=experiment_name,
            selected_count=len(prediction_ids),
            limit=limit,
            timeout=timeout,
        ),
        style=enqueue_scores_style(len(prediction_ids)),
    )


def run_repair_command(
    *,
    config: EvalDbosConfig,
    experiment_name: str,
    generation_limit: int,
    scoring_limit: int,
    score_timeout: float,
    apply: bool,
    build_repair_plan: Callable[..., Any],
    apply_repair: Callable[..., Any],
    repair_plan_line: Callable[..., str],
    repair_plan_style: Callable[..., str],
    repair_apply_line: Callable[..., str],
    plan_has_work: Callable[[Any], bool],
    operator_log: Callable[..., None],
) -> None:
    plan = build_repair_plan(
        config.database_url,
        dbos_system_database_url=config.dbos_system_database_url,
        experiment_name=experiment_name,
        generation_limit=generation_limit,
        scoring_limit=scoring_limit,
    )
    operator_log(
        repair_plan_line(
            experiment_name=experiment_name,
            plan=plan,
            apply=apply,
        ),
        style=repair_plan_style(plan, apply=apply),
    )
    if not apply:
        if plan_has_work(plan):
            operator_log(
                "dry run only; rerun with --apply to reconcile statuses and "
                "enqueue fresh retry workflows",
                style="yellow",
            )
        return
    result = apply_repair(
        config,
        experiment_name=experiment_name,
        generation_limit=generation_limit,
        scoring_limit=scoring_limit,
        score_timeout=score_timeout,
    )
    operator_log(
        repair_apply_line(experiment_name=experiment_name, result=result),
        style="green",
    )


def run_analyze_command[RecordT, SummaryT](
    *,
    database_url: str,
    experiment_name: str,
    csv_path: Path | None,
    markdown: bool,
    fetch_analysis_records: Callable[..., Sequence[RecordT]],
    summarize_analysis_records: Callable[
        [Sequence[RecordT]], Sequence[SummaryT]
    ],
    analysis_markdown: Callable[..., str],
    analysis_table: Callable[..., object],
    write_analysis_csv: Callable[[Sequence[SummaryT], Path], None],
    console: Console,
    operator_log: Callable[..., None],
) -> None:
    records = fetch_analysis_records(
        database_url, experiment_name=experiment_name
    )
    summaries = summarize_analysis_records(records)
    if markdown:
        typer.echo(
            analysis_markdown(
                experiment_name=experiment_name, summaries=summaries
            ),
            nl=False,
        )
    else:
        console.print(
            analysis_table(
                experiment_name=experiment_name, summaries=summaries
            )
        )
    if csv_path is not None:
        write_analysis_csv(summaries, csv_path)
        operator_log(f"wrote {csv_path}", style="green")


def run_worker_command(
    *,
    config: EvalDbosConfig,
    experiment_name: str,
    queue: QueueSelection,
    open_file_limit: int,
    log_file: Path | None,
    monitor: bool,
    monitor_interval: float,
    monitor_summary_interval: float,
    db_pool_max_size: str,
    prediction_table: str,
    db_pools: Mapping[str, object],
    raise_open_file_limit: Callable[[int], Any],
    open_file_limit_line: Callable[[Any], str],
    open_file_limit_style: Callable[[Any], str],
    create_schema: Callable[[str], None],
    configure_pooled_worker_runtime: Callable[..., Any],
    resolve_worker_log_path: Callable[..., Path],
    configure_worker_file_logging: Callable[[Path], object],
    queue_names_for_selection: Callable[..., tuple[str, ...]],
    start_worker_monitor: Callable[
        [WorkerMonitorConfig, threading.Event], threading.Thread
    ],
    close_db_connection_pools: Callable[[], None],
    operator_log: Callable[..., None],
) -> None:
    file_limit = raise_open_file_limit(open_file_limit)
    operator_log(
        open_file_limit_line(file_limit),
        style=open_file_limit_style(file_limit),
    )
    create_schema(config.database_url)
    pool_config = configure_pooled_worker_runtime(
        config,
        experiment_name=experiment_name,
        queue=queue,
        raw_db_pool_max_size=db_pool_max_size,
    )
    operator_log(
        f"{'DB Pool':<14} | max_size={pool_config.max_size:>5} | "
        f"urls={len(db_pools):>2}",
        style="cyan",
    )
    resolved_log_file = resolve_worker_log_path(
        experiment_name=experiment_name,
        queue=queue,
        log_file=log_file,
    )
    configure_worker_file_logging(resolved_log_file)
    selected_queue_names = queue_names_for_selection(
        queue, experiment_name=experiment_name
    )
    operator_log(
        f"worker listening on {queue.value} queue(s): "
        f"{', '.join(selected_queue_names)}",
        style="cyan",
    )
    operator_log(f"detailed worker log: {resolved_log_file}", style="cyan")

    stop_event = threading.Event()
    monitor_thread: threading.Thread | None = None
    if monitor:
        monitor_config = WorkerMonitorConfig(
            database_url=config.database_url,
            dbos_system_database_url=config.dbos_system_database_url,
            experiment_name=experiment_name,
            prediction_table=prediction_table,
            queue_selection=queue,
            queue_names=selected_queue_names,
            interval_seconds=monitor_interval,
            summary_interval_seconds=monitor_summary_interval,
        )
        monitor_thread = start_worker_monitor(monitor_config, stop_event)
        operator_log(
            "worker monitor enabled: "
            f"interval={monitor_interval}s, "
            f"summary_interval={monitor_summary_interval}s",
            style="cyan",
        )
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        operator_log("worker stopping", style="cyan")
    finally:
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=1.0)
        close_db_connection_pools()


def summarize_analysis_records[RecordT, SummaryT](
    records: Sequence[RecordT],
    *,
    group_key: Callable[[RecordT], object],
    model_label: Callable[[RecordT], str],
    temperature: Callable[[RecordT], float],
    task_id: Callable[[RecordT], str],
    score: Callable[[RecordT], float],
    provider_cost: Callable[[RecordT], float | None],
    raw_compile_ok: Callable[[RecordT], bool | None],
    extracted_compile_ok: Callable[[RecordT], bool | None],
    best_compression_ratio: Callable[[RecordT], float | None] = (
        lambda _record: None
    ),
    best_compression_percent_reduction: Callable[
        [RecordT], float | None
    ] = (lambda _record: None),
    summary_factory: Callable[..., SummaryT],
) -> list[SummaryT]:
    grouped: dict[object, list[RecordT]] = {}
    for record in records:
        grouped.setdefault(group_key(record), []).append(record)

    summaries: list[SummaryT] = []
    for _key, group in sorted(grouped.items(), key=lambda item: item[0]):
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
        compression_ratios = [
            ratio
            for ratio in (best_compression_ratio(record) for record in group)
            if ratio is not None
        ]
        compression_reductions = [
            reduction
            for reduction in (
                best_compression_percent_reduction(record)
                for record in group
            )
            if reduction is not None
        ]
        summaries.append(
            summary_factory(
                model=model_label(group[0]),
                temperature=temperature(group[0]),
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
                avg_best_compression_ratio=analysis.average_or_none(
                    compression_ratios
                ),
                avg_best_compression_percent_reduction=(
                    analysis.average_or_none(compression_reductions)
                ),
            )
        )
    return summaries
