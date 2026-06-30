#!/usr/bin/env python3
"""Q2: choose compression target range from enc-dec pass rates and coverage."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
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
from dr_dspy.analysis.plotting import (
    annotate_bars,
    apply_light_plot_style,
    save_figure,
)
from dr_dspy.analysis.report import AnalysisReporter
from dr_dspy.platform.cli_env import load_env_file, run_typer_app
from dr_dspy.records import GenerationRunStatus, ScoreAttemptStatus

app = typer.Typer(add_completion=False)
SCRIPT_NAME = "q2_compression_range"


def build_compression_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(
        ["compression_target", "model"],
        dropna=False,
    ):
        compression_target, model = cast(tuple[object, object], key)
        score_success = score_success_mask(group)
        scoreable = group.loc[score_success]
        passed = pass_mask(scoreable)
        pass_rate = float(passed.mean()) if score_success.any() else pd.NA
        gen_errors = (
            group["generation_status"] == GenerationRunStatus.ERROR.value
        ).sum()
        score_errors = (
            group["score_status"] == ScoreAttemptStatus.ERROR.value
        ).sum()
        rows.append(
            {
                "compression_target": compression_target,
                "model": model,
                "total_runs": len(group),
                "scoreable_count": int(score_success.sum()),
                "pass_count": int(passed.sum()),
                "pass_rate": pass_rate,
                "generation_error_count": int(gen_errors),
                "score_error_count": int(score_errors),
                "avg_realized_compression_ratio": scoreable[
                    "realized_compression_ratio"
                ].mean(),
            }
        )
    summary = pd.DataFrame(rows)
    overall = (
        summary.groupby("compression_target", dropna=False)
        .agg(
            total_runs=("total_runs", "sum"),
            scoreable_count=("scoreable_count", "sum"),
            pass_count=("pass_count", "sum"),
            generation_error_count=("generation_error_count", "sum"),
            score_error_count=("score_error_count", "sum"),
            avg_realized_compression_ratio=(
                "avg_realized_compression_ratio",
                "mean",
            ),
            model_count=("model", "nunique"),
        )
        .reset_index()
    )
    scoreable_counts = overall["scoreable_count"].replace(0, pd.NA)
    overall["pass_rate"] = overall["pass_count"] / scoreable_counts
    overall["rollup"] = "overall"
    detail = summary.copy()
    detail["rollup"] = "model"
    return pd.concat([detail, overall], ignore_index=True, sort=False)


def render_compression_markdown(summary: pd.DataFrame) -> str:
    overall = summary[summary["rollup"] == "overall"].sort_values(
        "compression_target"
    )
    lines = ["# Q2 compression range", ""]
    if overall.empty:
        lines.append("No enc-dec rows loaded.")
        return "\n".join(lines)
    trusted = overall[overall["scoreable_count"] >= 5]
    if trusted.empty:
        lines.append(
            "Coverage is sparse across all compression targets; "
            "treat pass rates as directional only."
        )
    else:
        best = trusted.sort_values("pass_rate", ascending=False).iloc[0]
        worst = trusted.sort_values("pass_rate", ascending=True).iloc[0]
        lines.extend(
            [
                f"Best covered target: `{best['compression_target']}` "
                f"({best['pass_rate']:.1%} pass, "
                f"{int(best['scoreable_count'])} scoreable).",
                f"Weakest covered target: `{worst['compression_target']}` "
                f"({worst['pass_rate']:.1%} pass, "
                f"{int(worst['scoreable_count'])} scoreable).",
            ]
        )
    lines.extend(["", "## Coverage by compression target", ""])
    for row in overall.itertuples():
        if pd.notna(row.pass_rate):
            pass_rate = f"{row.pass_rate:.1%}"
        else:
            pass_rate = "n/a"
        lines.append(
            f"- `{row.compression_target}`: pass rate {pass_rate}, "
            f"{row.scoreable_count} scoreable, {row.model_count} models."
        )
    return "\n".join(lines)


def plot_pass_rate_vs_compression(
    summary: pd.DataFrame,
    output_path: Path,
) -> Path:
    apply_light_plot_style()
    overall = summary[summary["rollup"] == "overall"].sort_values(
        "compression_target"
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    pass_rates = overall["pass_rate"].fillna(0.0)
    labels = [
        f"{value}" for value in overall["compression_target"].tolist()
    ]
    bars = ax.bar(
        labels,
        pass_rates,
    )
    annotate_bars(ax, bars)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Compression target")
    ax.set_ylabel("Pass rate")
    ax.set_title("Pass rate vs compression target")
    ax.grid(axis="y", alpha=0.3)
    return save_figure(fig, output_path)


def plot_coverage_vs_compression(
    summary: pd.DataFrame,
    output_path: Path,
) -> Path:
    apply_light_plot_style()
    overall = summary[summary["rollup"] == "overall"].sort_values(
        "compression_target"
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [
        f"{value}" for value in overall["compression_target"].tolist()
    ]
    ax.bar(
        labels,
        overall["scoreable_count"],
        color="#4472C4",
    )
    ax.set_xlabel("Compression target")
    ax.set_ylabel("Scoreable run count")
    ax.set_title("Coverage vs compression target")
    ax.grid(axis="y", alpha=0.3)
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
        title="Q2: Compression range",
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

    summary = build_compression_summary(frame)
    csv_path = output_run.artifact_path("compression_range", suffix="csv")
    md_path = output_run.artifact_path("compression_range", suffix="md")
    pass_plot_path = output_run.path("pass_rate_vs_compression")
    coverage_plot_path = output_run.path("coverage_vs_compression")
    summary.to_csv(csv_path, index=False)
    md_path.write_text(render_compression_markdown(summary), encoding="utf-8")
    figure_paths: list[Path] = []
    if not summary.empty:
        figure_paths.append(
            plot_pass_rate_vs_compression(summary, pass_plot_path)
        )
        figure_paths.append(
            plot_coverage_vs_compression(summary, coverage_plot_path)
        )

    reporter.section("Compression target summary")
    table = Table(title="Pass rate by compression", header_style="bold cyan")
    for column in (
        "compression_target",
        "scoreable_count",
        "pass_rate",
        "model_count",
    ):
        table.add_column(column)
    overall = summary[summary["rollup"] == "overall"].sort_values(
        "compression_target"
    )
    for row in overall.itertuples():
        table.add_row(
            str(row.compression_target),
            str(row.scoreable_count),
            f"{row.pass_rate:.1%}" if pd.notna(row.pass_rate) else "n/a",
            str(row.model_count),
        )
    reporter.print_table(table)
    if overall.empty:
        reporter.print_note("No enc-dec rows matched the experiment filter.")
    reporter.finish(
        output_run=output_run,
        csv_paths=[csv_path],
        md_paths=[md_path],
        figure_paths=figure_paths,
    )


if __name__ == "__main__":
    run_typer_app(app)
