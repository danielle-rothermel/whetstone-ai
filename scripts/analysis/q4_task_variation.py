#!/usr/bin/env python3
"""Q4: rank HumanEval tasks by optimization signal quality."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import typer
from rich.table import Table

from dr_dspy.analysis.cli_options import (
    DatabaseUrlOption,
    EnvFileOption,
    ExperimentNameOption,
    LimitOption,
    RequireScoreOption,
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
SCRIPT_NAME = "q4_task_variation"

MIN_SCOREABLE = 3


def binary_entropy(pass_rate: float) -> float:
    if pass_rate <= 0.0 or pass_rate >= 1.0:
        return 0.0
    fail_rate = 1 - pass_rate
    return -(pass_rate * math.log(pass_rate) + fail_rate * math.log(fail_rate))


def classify_task_signal(
    *,
    pass_rate: float,
    scoreable_count: int,
    min_scoreable: int = MIN_SCOREABLE,
) -> dict[str, bool]:
    sparse = scoreable_count < min_scoreable
    always_pass = (
        not sparse and pass_rate == 1.0 and scoreable_count >= min_scoreable
    )
    always_fail = (
        not sparse and pass_rate == 0.0 and scoreable_count >= min_scoreable
    )
    useful_signal = (
        not sparse
        and not always_pass
        and not always_fail
        and binary_entropy(pass_rate) > 0.0
    )
    return {
        "sparse": sparse,
        "always_pass": always_pass,
        "always_fail": always_fail,
        "useful_signal": useful_signal,
    }


def build_task_variation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    scoreable = frame.loc[score_success_mask(frame)].copy()
    scoreable["passed"] = pass_mask(scoreable).astype(int)
    rows: list[dict[str, object]] = []
    for task_id, group in scoreable.groupby("task_id"):
        if len(group):
            pass_rate_value = float(group["passed"].mean())
            pass_rate = pass_rate_value
        else:
            pass_rate_value = 0.0
            pass_rate = pd.NA
        flags = classify_task_signal(
            pass_rate=pass_rate_value,
            scoreable_count=len(group),
        )
        rows.append(
            {
                "task_id": task_id,
                "pass_rate": pass_rate,
                "scoreable_count": len(group),
                "distinct_models": group["model"].nunique(),
                "distinct_compression_targets": group[
                    "compression_target"
                ].nunique(),
                "pass_variance": float(group["passed"].var(ddof=0))
                if len(group) > 1
                else 0.0,
                "entropy": binary_entropy(pass_rate_value),
                **flags,
            }
        )
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["useful_signal", "entropy", "scoreable_count"],
        ascending=[False, False, False],
    )


def render_task_variation_markdown(summary: pd.DataFrame) -> str:
    lines = ["# Q4 task variation", ""]
    if summary.empty:
        lines.append("No score-success enc-dec rows loaded.")
        return "\n".join(lines)
    useful = summary[summary["useful_signal"]]
    sparse = summary[summary["sparse"]]
    always_pass = summary[summary["always_pass"]]
    always_fail = summary[summary["always_fail"]]
    lines.extend(
        [
            f"- Useful signal tasks: {len(useful)}",
            f"- Sparse tasks: {len(sparse)}",
            f"- Always pass: {len(always_pass)}",
            f"- Always fail: {len(always_fail)}",
            "",
            "## Top useful-signal tasks",
            "",
        ]
    )
    for row in useful.head(10).itertuples():
        lines.append(
            f"- `{row.task_id}`: pass rate {row.pass_rate:.1%}, "
            f"entropy {row.entropy:.3f}, {row.scoreable_count} scoreable."
        )
    return "\n".join(lines)


def plot_task_signal_rank(summary: pd.DataFrame, output_path: Path) -> Path:
    apply_light_plot_style()
    ranked = summary.sort_values("entropy", ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(ranked))))
    ax.barh(ranked["task_id"].astype(str), ranked["entropy"], color="#70AD47")
    ax.set_xlabel("Binary pass-rate entropy")
    ax.set_title("Task signal rank (top 20 by entropy)")
    ax.grid(axis="x", alpha=0.3)
    return save_figure(fig, output_path)


def plot_task_pass_rate_distribution(
    summary: pd.DataFrame,
    output_path: Path,
) -> Path:
    apply_light_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = min(10, len(summary))
    ax.hist(
        summary["pass_rate"].dropna(),
        bins=bins,
        color="#ED7D31",
    )
    ax.set_xlabel("Task pass rate")
    ax.set_ylabel("Task count")
    ax.set_title("Task pass-rate distribution")
    ax.grid(axis="y", alpha=0.3)
    return save_figure(fig, output_path)


@app.command()
def main(
    experiment_name: ExperimentNameOption,
    database_url: DatabaseUrlOption = None,
    env_file: EnvFileOption = None,
    limit: LimitOption = None,
    require_score: RequireScoreOption = True,
) -> None:
    if env_file is not None:
        load_env_file(env_file)
    else:
        load_env_file()
    if not experiment_name:
        raise typer.BadParameter("At least one --experiment-name is required")
    reporter = AnalysisReporter(
        title="Q4: Task variation",
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
            require_score=require_score,
            limit=limit,
        )
    finally:
        engine.dispose()

    reporter.print_run_context(
        experiment_names=experiment_name,
        row_count=len(frame),
        limit=limit,
        require_score=require_score,
    )

    summary = build_task_variation_summary(frame)
    csv_path = output_run.artifact_path("task_variation", suffix="csv")
    md_path = output_run.artifact_path("task_variation", suffix="md")
    rank_plot_path = output_run.path("task_signal_rank")
    dist_plot_path = output_run.path("task_pass_rate_distribution")
    summary.to_csv(csv_path, index=False)
    md_path.write_text(
        render_task_variation_markdown(summary),
        encoding="utf-8",
    )
    figure_paths: list[Path] = []
    if not summary.empty:
        figure_paths.append(plot_task_signal_rank(summary, rank_plot_path))
        figure_paths.append(
            plot_task_pass_rate_distribution(summary, dist_plot_path)
        )

    reporter.section("Task signal flags (top 10)")
    table = Table(title="Task variation", header_style="bold cyan")
    for column in (
        "task_id",
        "pass_rate",
        "scoreable_count",
        "useful_signal",
        "always_pass",
        "always_fail",
    ):
        table.add_column(column)
    for _, row in summary.head(10).iterrows():
        table.add_row(
            str(row["task_id"]),
            f"{row['pass_rate']:.1%}" if pd.notna(row["pass_rate"]) else "n/a",
            str(row["scoreable_count"]),
            str(row["useful_signal"]),
            str(row["always_pass"]),
            str(row["always_fail"]),
        )
    reporter.print_table(table)
    if summary.empty:
        reporter.print_note(
            "No score-success rows matched the experiment filter."
        )
    elif not summary["useful_signal"].any():
        reporter.print_note(
            "No tasks flagged as useful signal at current coverage thresholds."
        )
    reporter.finish(
        output_run=output_run,
        csv_paths=[csv_path],
        md_paths=[md_path],
        figure_paths=figure_paths,
    )


if __name__ == "__main__":
    run_typer_app(app)
