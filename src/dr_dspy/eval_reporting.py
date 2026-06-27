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
    dimensions: dict[str, Any]
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


def _dimension_cell(value: Any) -> str:
    if isinstance(value, float):
        return analysis.format_float(value)
    return str(value)


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
    *,
    experiment_name: str,
    summaries: Sequence[AnalysisSummaryLike],
    dimensions: Sequence[StatusDimension],
) -> str:
    metric_headers = [
        "Samples",
        "Scored",
        "Total Price",
        "Avg Price/1k Samples",
        "Avg Perf",
        "Raw Compile",
        "Extracted Compile",
        "Extraction Lift",
        "Avg Compression Ratio",
        "Avg Compression Reduction",
        "Price Var",
        "Perf Var",
        "Rep Var",
    ]
    dim_titles = [dimension.title for dimension in dimensions]
    header = "| " + " | ".join([*dim_titles, *metric_headers]) + " |"
    align = (
        "|"
        + "|".join(["---"] * len(dimensions) + ["---:"] * len(metric_headers))
        + "|"
    )
    lines = [f"# Eval Analysis: {experiment_name}", "", header, align]
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
            analysis.price_per_thousand_samples(summary.avg_price_per_sample)
            for summary in summaries
        ]
    )
    for summary, total_price, price_per_thousand in zip(
        summaries, row_total_prices, prices_per_thousand_samples, strict=True
    ):
        dim_cells = [
            _dimension_cell(summary.dimensions.get(dimension.key))
            for dimension in dimensions
        ]
        metric_cells = [
            str(summary.sample_count),
            str(summary.scored_count),
            total_price,
            price_per_thousand,
            analysis.format_float(summary.avg_performance),
            str(summary.raw_compile_pass_count),
            str(summary.extracted_compile_pass_count),
            str(summary.extraction_lift),
            analysis.format_float(summary.avg_best_compression_ratio),
            analysis.format_float(
                summary.avg_best_compression_percent_reduction
            ),
            analysis.format_float(summary.price_variance),
            analysis.format_float(summary.performance_variance),
            analysis.format_float(summary.avg_repetition_variance),
        ]
        lines.append("| " + " | ".join([*dim_cells, *metric_cells]) + " |")
    if summaries:
        total_dim_cells = _total_dimension_cells(dimensions)
        total_metric_cells = [
            str(sum(summary.sample_count for summary in summaries)),
            str(sum(summary.scored_count for summary in summaries)),
            total_prices[-1],
            "",
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
            "",
            "",
            "",
        ]
        lines.append(
            "| " + " | ".join([*total_dim_cells, *total_metric_cells]) + " |"
        )
    return "\n".join(lines) + "\n"


def _total_dimension_cells(
    dimensions: Sequence[StatusDimension],
) -> list[str]:
    if not dimensions:
        return []
    return [ANALYSIS_TOTAL_LABEL, *([""] * (len(dimensions) - 1))]


def _add_dimension_columns(
    table: Table, dimensions: Sequence[StatusDimension]
) -> None:
    for index, dimension in enumerate(dimensions):
        if index == 0:
            table.add_column(dimension.title, min_width=28, overflow="fold")
        else:
            table.add_column(dimension.title, justify=dimension.justify)


def analysis_table(
    *,
    experiment_name: str,
    summaries: Sequence[AnalysisSummaryLike],
    dimensions: Sequence[StatusDimension],
) -> Group:
    def new_table(title: str) -> Table:
        return Table(
            title=title,
            box=box.SIMPLE_HEAVY,
            show_lines=False,
            row_styles=TABLE_ROW_STYLES,
        )

    performance_table = new_table(f"Eval Analysis: {experiment_name}")
    _add_dimension_columns(performance_table, dimensions)
    for column in (
        "Samples",
        "Scored",
        "Avg Perf",
        "Raw Compile",
        "Extracted Compile",
        "Lift",
        "Comp Ratio",
        "Comp Reduction",
    ):
        performance_table.add_column(column, justify="right")

    cost_table = new_table("Cost")
    _add_dimension_columns(cost_table, dimensions)
    for column in ("Total $", "Avg $/1k Samples"):
        cost_table.add_column(column, justify="right")

    variance_table = new_table("Variance")
    _add_dimension_columns(variance_table, dimensions)
    for column in ("Price Var", "Perf Var", "Rep Var"):
        variance_table.add_column(column, justify="right")

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
            analysis.price_per_thousand_samples(summary.avg_price_per_sample)
            for summary in summaries
        ]
    )

    for summary, total_price, price_per_thousand in zip(
        summaries, row_total_prices, prices_per_thousand_samples, strict=True
    ):
        dim_cells = [
            _dimension_cell(summary.dimensions.get(dimension.key))
            for dimension in dimensions
        ]
        performance_table.add_row(
            *dim_cells,
            str(summary.sample_count),
            str(summary.scored_count),
            analysis.format_float(summary.avg_performance),
            str(summary.raw_compile_pass_count),
            str(summary.extracted_compile_pass_count),
            str(summary.extraction_lift),
            analysis.format_float(summary.avg_best_compression_ratio),
            analysis.format_float(
                summary.avg_best_compression_percent_reduction
            ),
        )
        cost_table.add_row(
            *dim_cells,
            total_price,
            price_per_thousand,
        )
        variance_table.add_row(
            *dim_cells,
            analysis.format_float(summary.price_variance),
            analysis.format_float(summary.performance_variance),
            analysis.format_float(summary.avg_repetition_variance),
        )
    if summaries:
        total_dim_cells = _total_dimension_cells(dimensions)
        performance_table.add_row(
            *total_dim_cells,
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
            *total_dim_cells,
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
            data = summary.model_dump(mode="json")
            dimensions = data.pop("dimensions", {})
            writer.writerow({**dimensions, **data})


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
