from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, StrictStr
from rich import box
from rich.console import Group
from rich.table import Table

from dr_dspy import analysis, dbos_runtime

ANALYSIS_TOTAL_LABEL = "Total"
TABLE_ROW_STYLES = ("", "on grey7")
TABLE_TOTAL_ROW_STYLE = "bold black on green3"


class StatusDimension(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: StrictStr
    title: StrictStr
    justify: Literal["default", "left", "center", "right", "full"] = "left"


class AnalysisSummaryLike(Protocol):
    model: str
    temperature: float
    sample_count: int
    scored_count: int
    total_price: float | None
    avg_price_per_sample: float | None
    price_variance: float | None
    avg_performance: float
    performance_variance: float | None
    avg_repetition_variance: float | None
    raw_compile_pass_count: int
    extracted_compile_pass_count: int
    extraction_lift: int
    avg_best_compression_ratio: float | None
    avg_best_compression_percent_reduction: float | None

    def model_dump(self, *, mode: str) -> dict[str, Any]: ...


def validate_sql_identifier(identifier: str) -> None:
    if not identifier.replace("_", "").isalnum():
        raise ValueError(f"unsupported SQL identifier: {identifier}")


def fetch_status_counts(
    database_url: str,
    *,
    prediction_table: str,
    dimension_columns: Sequence[str],
    experiment_name: str | None,
) -> list[dict[str, Any]]:
    validate_sql_identifier(prediction_table)
    for column in dimension_columns:
        validate_sql_identifier(column)
    where_clause = ""
    params: tuple[str, ...] = ()
    if experiment_name is not None:
        where_clause = "WHERE experiment_name = %s"
        params = (experiment_name,)

    selected_dimensions = ", ".join(dimension_columns)
    group_dimensions = ", ".join(["experiment_name", *dimension_columns])
    order_dimensions = ", ".join(["experiment_name", *dimension_columns])
    query = f"""
        SELECT
            experiment_name,
            {selected_dimensions},
            generation_status,
            scoring_status,
            COUNT(*) AS count
        FROM {prediction_table}
        {where_clause}
        GROUP BY
            {group_dimensions},
            generation_status,
            scoring_status
        ORDER BY
            {order_dimensions},
            generation_status,
            scoring_status
    """
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(cast(Any, query), params)
            rows = cur.fetchall()
    records: list[dict[str, Any]] = []
    for row in rows:
        record = {"experiment_name": row[0]}
        for index, column in enumerate(dimension_columns, start=1):
            record[column] = row[index]
        offset = len(dimension_columns) + 1
        record["generation_status"] = row[offset]
        record["scoring_status"] = row[offset + 1]
        record["count"] = row[offset + 2]
        records.append(record)
    return records


def status_counts_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str,
    dimensions: Sequence[StatusDimension],
    experiment_name: str | None,
) -> Table:
    table_title = title
    if experiment_name is not None:
        table_title = f"{title}: {experiment_name}"
    table = Table(
        title=table_title,
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        row_styles=TABLE_ROW_STYLES,
    )
    if experiment_name is None:
        table.add_column("Experiment", overflow="fold")
    for dimension in dimensions:
        table.add_column(
            dimension.title,
            justify=dimension.justify,
            overflow="fold",
        )
    table.add_column("Generation", justify="center")
    table.add_column("Scoring", justify="center")
    table.add_column("Count", justify="right")
    for row in rows:
        values = []
        if experiment_name is None:
            values.append(str(row["experiment_name"]))
        for dimension in dimensions:
            value = row[dimension.key]
            if isinstance(value, float):
                values.append(analysis.format_float(value))
            else:
                values.append(str(value))
        values.extend(
            [
                str(row["generation_status"]),
                str(row["scoring_status"]),
                str(row["count"]),
            ]
        )
        table.add_row(*values)
    return table


def analysis_markdown(
    *, experiment_name: str, summaries: Sequence[AnalysisSummaryLike]
) -> str:
    lines = [
        f"# Eval Analysis: {experiment_name}",
        "",
        "| Model | Temp | Samples | Scored | Total Price | "
        "Avg Price/1k Samples | Avg Perf | Raw Compile | "
        "Extracted Compile | Extraction Lift | Avg Compression Ratio | "
        "Avg Compression Reduction | Price Var | Perf Var | Rep Var |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    total_price_values = [summary.total_price for summary in summaries]
    total_price_sum = analysis.sum_present_float(total_price_values)
    total_prices = analysis.format_cost_column(
        [*total_price_values, total_price_sum]
        if summaries
        else total_price_values
    )
    row_total_prices = total_prices[: len(summaries)]
    prices_per_thousand_samples = analysis.format_cost_column(
        [
            analysis.price_per_thousand_samples(
                summary.avg_price_per_sample
            )
            for summary in summaries
        ]
    )
    for summary, total_price, price_per_thousand in zip(
        summaries,
        row_total_prices,
        prices_per_thousand_samples,
        strict=True,
    ):
        lines.append(
            "| {model} | {temperature} | {sample_count} | {scored_count} | "
            "{total_price} | {avg_price_per_sample} | {avg_performance} | "
            "{raw_compile_pass_count} | {extracted_compile_pass_count} | "
            "{extraction_lift} | {avg_best_compression_ratio} | "
            "{avg_best_compression_percent_reduction} | {price_variance} | "
            "{performance_variance} | {avg_repetition_variance} |".format(
                model=summary.model,
                temperature=analysis.format_float(summary.temperature),
                sample_count=summary.sample_count,
                scored_count=summary.scored_count,
                total_price=total_price,
                avg_price_per_sample=price_per_thousand,
                avg_performance=analysis.format_float(
                    summary.avg_performance
                ),
                raw_compile_pass_count=summary.raw_compile_pass_count,
                extracted_compile_pass_count=(
                    summary.extracted_compile_pass_count
                ),
                extraction_lift=summary.extraction_lift,
                avg_best_compression_ratio=analysis.format_float(
                    summary.avg_best_compression_ratio
                ),
                avg_best_compression_percent_reduction=(
                    analysis.format_float(
                        summary.avg_best_compression_percent_reduction
                    )
                ),
                price_variance=analysis.format_float(
                    summary.price_variance
                ),
                performance_variance=analysis.format_float(
                    summary.performance_variance
                ),
                avg_repetition_variance=analysis.format_float(
                    summary.avg_repetition_variance
                ),
            )
        )
    if summaries:
        lines.append(
            (
                "| {model} |  | {sample_count} | {scored_count} | "
                "{total_price} |  |  | {raw_compile_pass_count} | "
                "{extracted_compile_pass_count} | {extraction_lift} | "
                " |  |  |  |  |"
            ).format(
                model=ANALYSIS_TOTAL_LABEL,
                sample_count=sum(
                    summary.sample_count for summary in summaries
                ),
                scored_count=sum(
                    summary.scored_count for summary in summaries
                ),
                total_price=total_prices[-1],
                raw_compile_pass_count=sum(
                    summary.raw_compile_pass_count for summary in summaries
                ),
                extracted_compile_pass_count=sum(
                    summary.extracted_compile_pass_count
                    for summary in summaries
                ),
                extraction_lift=sum(
                    summary.extraction_lift for summary in summaries
                ),
            )
        )
    return "\n".join(lines) + "\n"


def analysis_table(
    *, experiment_name: str, summaries: Sequence[AnalysisSummaryLike]
) -> Group:
    performance_table = Table(
        title=f"Eval Analysis: {experiment_name}",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        row_styles=TABLE_ROW_STYLES,
    )
    performance_table.add_column("Model", min_width=28, overflow="fold")
    performance_table.add_column("Temp", justify="right")
    performance_table.add_column("Samples", justify="right")
    performance_table.add_column("Scored", justify="right")
    performance_table.add_column("Avg Perf", justify="right")
    performance_table.add_column("Raw Compile", justify="right")
    performance_table.add_column("Extracted Compile", justify="right")
    performance_table.add_column("Lift", justify="right")
    performance_table.add_column("Comp Ratio", justify="right")
    performance_table.add_column("Comp Reduction", justify="right")

    cost_table = Table(
        title="Cost",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        row_styles=TABLE_ROW_STYLES,
    )
    cost_table.add_column("Model", min_width=28, overflow="fold")
    cost_table.add_column("Temp", justify="right")
    cost_table.add_column("Total $", justify="right")
    cost_table.add_column("Avg $/1k Samples", justify="right")

    variance_table = Table(
        title="Variance",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        row_styles=TABLE_ROW_STYLES,
    )
    variance_table.add_column("Model", min_width=28, overflow="fold")
    variance_table.add_column("Temp", justify="right")
    variance_table.add_column("Price Var", justify="right")
    variance_table.add_column("Perf Var", justify="right")
    variance_table.add_column("Rep Var", justify="right")

    total_price_values = [summary.total_price for summary in summaries]
    total_price_sum = analysis.sum_present_float(total_price_values)
    temperatures = analysis.format_float_column(
        [summary.temperature for summary in summaries]
    )
    avg_performances = analysis.format_float_column(
        [summary.avg_performance for summary in summaries]
    )
    total_prices = analysis.format_cost_column(
        [*total_price_values, total_price_sum]
        if summaries
        else total_price_values
    )
    row_total_prices = total_prices[: len(summaries)]
    prices_per_thousand_samples = analysis.format_cost_column(
        [
            analysis.price_per_thousand_samples(
                summary.avg_price_per_sample
            )
            for summary in summaries
        ]
    )
    price_variances = analysis.format_float_column(
        [summary.price_variance for summary in summaries]
    )
    performance_variances = analysis.format_float_column(
        [summary.performance_variance for summary in summaries]
    )
    repetition_variances = analysis.format_float_column(
        [summary.avg_repetition_variance for summary in summaries]
    )
    compression_ratios = analysis.format_float_column(
        [summary.avg_best_compression_ratio for summary in summaries]
    )
    compression_reductions = analysis.format_float_column(
        [
            summary.avg_best_compression_percent_reduction
            for summary in summaries
        ]
    )

    for (
        summary,
        temperature,
        avg_performance,
        total_price,
        price_per_thousand,
        price_variance,
        performance_variance,
        repetition_variance,
        compression_ratio,
        compression_reduction,
    ) in zip(
        summaries,
        temperatures,
        avg_performances,
        row_total_prices,
        prices_per_thousand_samples,
        price_variances,
        performance_variances,
        repetition_variances,
        compression_ratios,
        compression_reductions,
        strict=True,
    ):
        performance_table.add_row(
            summary.model,
            temperature,
            str(summary.sample_count),
            str(summary.scored_count),
            avg_performance,
            str(summary.raw_compile_pass_count),
            str(summary.extracted_compile_pass_count),
            str(summary.extraction_lift),
            compression_ratio,
            compression_reduction,
        )
        cost_table.add_row(
            summary.model,
            temperature,
            total_price,
            price_per_thousand,
        )
        variance_table.add_row(
            summary.model,
            temperature,
            price_variance,
            performance_variance,
            repetition_variance,
        )
    if summaries:
        performance_table.add_row(
            ANALYSIS_TOTAL_LABEL,
            "",
            str(sum(summary.sample_count for summary in summaries)),
            str(sum(summary.scored_count for summary in summaries)),
            "",
            str(sum(summary.raw_compile_pass_count for summary in summaries)),
            str(
                sum(
                    summary.extracted_compile_pass_count
                    for summary in summaries
                )
            ),
            str(sum(summary.extraction_lift for summary in summaries)),
            "",
            "",
            style=TABLE_TOTAL_ROW_STYLE,
        )
        cost_table.add_row(
            ANALYSIS_TOTAL_LABEL,
            "",
            total_prices[-1],
            "",
            style=TABLE_TOTAL_ROW_STYLE,
        )
    return Group(performance_table, cost_table, variance_table)


def write_analysis_csv(
    summaries: Sequence[AnalysisSummaryLike],
    *,
    csv_path: Path,
    fieldnames: Sequence[str],
) -> None:
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.model_dump(mode="json"))


def enqueue_scores_line(
    *,
    experiment_name: str,
    selected_count: int,
    limit: int,
    timeout: float,
) -> str:
    return (
        f"{'Enqueue Scores':<14} | "
        f"selected={selected_count:>5} | "
        f"limit={limit:>5} | "
        f"timeout={timeout:>6.1f}s | "
        f"experiment={experiment_name}"
    )


def enqueue_scores_style(selected_count: int) -> str:
    if selected_count == 0:
        return "yellow"
    return "green"


def repair_plan_line(
    *,
    experiment_name: str,
    gen_stranded: int,
    gen_errors: int,
    score_pending: int,
    score_stranded: int,
    score_errors: int,
    apply: bool,
) -> str:
    mode = "apply" if apply else "dry-run"
    return (
        f"{'Repair Plan':<14} | "
        f"gen_stranded={gen_stranded:>5} | "
        f"gen_errors={gen_errors:>5} | "
        f"score_pending={score_pending:>5} | "
        f"score_stranded={score_stranded:>5} | "
        f"score_errors={score_errors:>5} | "
        f"mode={mode} | "
        f"experiment={experiment_name}"
    )


def repair_apply_line(
    *,
    experiment_name: str,
    stranded_generations_marked: int,
    generation_retries_enqueued: int,
    stranded_scoring_marked: int,
    pending_scoring_enqueued: int,
    scoring_retries_enqueued: int,
    repair_token: str,
) -> str:
    return (
        f"{'Repair Apply':<14} | "
        f"gen_marked={stranded_generations_marked:>5} | "
        f"gen_retry={generation_retries_enqueued:>5} | "
        f"score_marked={stranded_scoring_marked:>5} | "
        f"score_pending={pending_scoring_enqueued:>5} | "
        f"score_retry={scoring_retries_enqueued:>5} | "
        f"token={repair_token} | "
        f"experiment={experiment_name}"
    )


def repair_plan_style(
    *,
    apply: bool,
    gen_stranded: int,
    gen_errors: int,
    score_pending: int,
    score_stranded: int,
    score_errors: int,
) -> str:
    if apply:
        return "green"
    if (
        gen_stranded
        or gen_errors
        or score_pending
        or score_stranded
        or score_errors
    ):
        return "cyan"
    return "yellow"
