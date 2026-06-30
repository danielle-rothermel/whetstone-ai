#!/usr/bin/env python3
"""Q3: estimate repeat count stability for enc-dec pass-rate estimates."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from rich.table import Table

from dr_dspy.analysis.cli_options import (
    DatabaseUrlOption,
    EnvFileOption,
    ExperimentNameOption,
    LimitOption,
)
from dr_dspy.analysis.db import create_analysis_engine
from dr_dspy.analysis.figures import FigureRun
from dr_dspy.analysis.frames import (
    load_encdec_analysis_frame,
    pass_mask,
    score_success_mask,
)
from dr_dspy.analysis.plotting import apply_light_plot_style, save_figure
from dr_dspy.analysis.report import AnalysisReporter
from dr_dspy.platform.cli_env import load_env_file, run_typer_app

app = typer.Typer(add_completion=False)
SCRIPT_NAME = "q3_repeat_stability"

DEFAULT_BOOTSTRAP_SAMPLES = 500
DEFAULT_MAX_SAMPLE_SIZE = 10
GROUP_KEY = ("task_id", "model", "compression_target")


def bootstrap_interval_width(
    passes: np.ndarray,
    *,
    sample_size: int,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> float:
    if sample_size <= 0 or len(passes) == 0:
        return float("nan")
    if sample_size > len(passes):
        return float("nan")
    rng = np.random.default_rng(seed)
    rates = np.empty(bootstrap_samples, dtype=float)
    for index in range(bootstrap_samples):
        sample = rng.choice(passes, size=sample_size, replace=True)
        rates[index] = sample.mean()
    low, high = np.quantile(rates, [0.05, 0.95])
    return float(high - low)


def estimate_repeat_stability(
    frame: pd.DataFrame,
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    max_sample_size: int = DEFAULT_MAX_SAMPLE_SIZE,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    coverage_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []
    aggregate: dict[int, list[float]] = {}

    scoreable = frame.loc[score_success_mask(frame)].copy()
    scoreable["passed"] = pass_mask(scoreable).astype(int)

    for group_key, group in scoreable.groupby(list(GROUP_KEY), dropna=False):
        task_id, model, compression_target = group_key
        passes = group["passed"].to_numpy(dtype=float)
        observed_variance = float(np.var(passes)) if len(passes) > 1 else 0.0
        coverage_rows.append(
            {
                "task_id": task_id,
                "model": model,
                "compression_target": compression_target,
                "repeat_count": len(group),
                "pass_rate": float(passes.mean()) if len(passes) else pd.NA,
                "observed_variance": observed_variance,
            }
        )
        max_size = min(len(passes), max_sample_size)
        for sample_size in range(1, max_size + 1):
            width = bootstrap_interval_width(
                passes,
                sample_size=sample_size,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
            interval_rows.append(
                {
                    "task_id": task_id,
                    "model": model,
                    "compression_target": compression_target,
                    "sample_size": sample_size,
                    "interval_width_90": width,
                    "repeat_count": len(group),
                }
            )
            if len(passes) >= sample_size and not np.isnan(width):
                aggregate.setdefault(sample_size, []).append(width)

    coverage = pd.DataFrame(coverage_rows)
    intervals = pd.DataFrame(interval_rows)
    aggregate_rows = [
        {
            "task_id": "",
            "model": "",
            "compression_target": pd.NA,
            "sample_size": size,
            "interval_width_90": pd.NA,
            "repeat_count": pd.NA,
            "median_interval_width_90": float(np.median(values)),
            "group_count": len(values),
        }
        for size, values in sorted(aggregate.items())
    ]
    intervals_with_summary = pd.concat(
        [intervals, pd.DataFrame(aggregate_rows)],
        ignore_index=True,
        sort=False,
    )
    return coverage, intervals_with_summary


def build_stability_summary(
    coverage: pd.DataFrame,
    intervals: pd.DataFrame,
) -> pd.DataFrame:
    aggregate = intervals[
        intervals["task_id"].eq("") & intervals["model"].eq("")
    ].copy()
    if aggregate.empty:
        aggregate = (
            intervals.dropna(subset=["interval_width_90"])
            .groupby("sample_size", as_index=False)
            .agg(median_interval_width_90=("interval_width_90", "median"))
        )
        aggregate["group_count"] = 0
    detail = intervals[
        ~(intervals["task_id"].eq("") & intervals["model"].eq(""))
    ].copy()
    if detail.empty:
        return coverage
    return detail.merge(
        coverage,
        on=["task_id", "model", "compression_target", "repeat_count"],
        how="left",
    )


def render_repeat_stability_markdown(
    coverage: pd.DataFrame,
    intervals: pd.DataFrame,
) -> str:
    lines = ["# Q3 repeat stability", ""]
    if coverage.empty:
        lines.append("No score-success enc-dec rows loaded.")
        return "\n".join(lines)
    sparse_groups = int((coverage["repeat_count"] < 2).sum())
    if sparse_groups == len(coverage):
        lines.append(
            "Data is too sparse for meaningful bootstrap intervals; only "
            "coverage counts are trustworthy."
        )
    elif sparse_groups > 0:
        lines.append(
            f"{sparse_groups}/{len(coverage)} groups have "
            "fewer than 2 repeats."
        )
    aggregate = intervals[
        intervals["task_id"].eq("") & intervals["model"].eq("")
    ].sort_values("sample_size")
    if not aggregate.empty:
        lines.extend(["", "## Median 90% interval width by sample size", ""])
        for row in aggregate.itertuples():
            width = (
                f"{row.median_interval_width_90:.3f}"
                if pd.notna(row.median_interval_width_90)
                else "n/a"
            )
            lines.append(
                f"- sample size {int(row.sample_size)}: width {width} "
                f"({int(row.group_count)} groups)."
            )
    return "\n".join(lines)


def plot_stability_by_sample_size(
    intervals: pd.DataFrame,
    output_path: Path,
) -> Path:
    apply_light_plot_style()
    aggregate = intervals[
        intervals["task_id"].eq("") & intervals["model"].eq("")
    ].sort_values("sample_size")
    fig, ax = plt.subplots(figsize=(8, 4))
    if aggregate.empty:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
    else:
        ax.plot(
            aggregate["sample_size"],
            aggregate["median_interval_width_90"],
            marker="o",
        )
        ax.set_xlabel("Repeats per (task, model, compression)")
        ax.set_ylabel("Median 90% pass-rate interval width")
        ax.set_title("Repeat stability by sample size")
        ax.grid(alpha=0.3)
    return save_figure(fig, output_path)


@app.command()
def main(
    experiment_name: ExperimentNameOption,
    database_url: DatabaseUrlOption = None,
    env_file: EnvFileOption = None,
    limit: LimitOption = None,
) -> None:
    if env_file is not None:
        load_env_file(env_file)
    else:
        load_env_file()
    if not experiment_name:
        raise typer.BadParameter("At least one --experiment-name is required")
    reporter = AnalysisReporter(
        title="Q3: Repeat stability",
        script_name=SCRIPT_NAME,
    )
    output_run = FigureRun.start(SCRIPT_NAME)
    output_run.ensure_directories()
    reporter.print_header()
    engine = create_analysis_engine(database_url, env_file=None)
    try:
        frame = load_encdec_analysis_frame(
            engine,
            experiment_name,
            limit=limit,
        )
    finally:
        engine.dispose()

    reporter.print_run_context(
        experiment_names=experiment_name,
        row_count=len(frame),
        limit=limit,
    )

    coverage, intervals = estimate_repeat_stability(frame)
    summary = build_stability_summary(coverage, intervals)
    csv_path = output_run.artifact_path("repeat_stability", suffix="csv")
    md_path = output_run.artifact_path("repeat_stability", suffix="md")
    plot_path = output_run.path("stability_by_sample_size")
    summary.to_csv(csv_path, index=False)
    md_path.write_text(
        render_repeat_stability_markdown(coverage, intervals),
        encoding="utf-8",
    )
    figure_paths: list[Path] = []
    if not intervals.empty:
        figure_paths.append(
            plot_stability_by_sample_size(intervals, plot_path)
        )

    if coverage.empty:
        reporter.print_note(
            "No score-success rows loaded; bootstrap intervals unavailable."
        )
    elif (coverage["repeat_count"] < 2).all():
        reporter.print_note(
            "Every group has fewer than 2 repeats; coverage table only."
        )

    reporter.section("Repeat coverage (top 10 groups)")
    table = Table(title="Group coverage", header_style="bold cyan")
    columns = (
        "task_id",
        "model",
        "compression_target",
        "repeat_count",
        "pass_rate",
    )
    for column in columns:
        table.add_column(column)
    for _, row in coverage.head(10).iterrows():
        table.add_row(
            str(row["task_id"]),
            str(row["model"]),
            str(row["compression_target"]),
            str(row["repeat_count"]),
            f"{row['pass_rate']:.1%}" if pd.notna(row["pass_rate"]) else "n/a",
        )
    reporter.print_table(table)
    reporter.finish(
        output_run=output_run,
        csv_paths=[csv_path],
        md_paths=[md_path],
        figure_paths=figure_paths,
    )


if __name__ == "__main__":
    run_typer_app(app)
