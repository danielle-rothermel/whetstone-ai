#!/usr/bin/env python3
"""Q1: rank enc-dec model candidates by reliability, pass rate, and cost."""

from __future__ import annotations

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
    generation_success_mask,
    load_encdec_analysis_frame,
    pass_mask,
    score_success_mask,
)
from dr_dspy.analysis.plotting import apply_light_plot_style, save_figure
from dr_dspy.analysis.report import AnalysisReporter
from dr_dspy.platform.cli_env import load_env_file, run_typer_app
from dr_dspy.records import GenerationRunStatus, ScoreAttemptStatus

app = typer.Typer(add_completion=False)
SCRIPT_NAME = "q1_model_candidates"


def build_model_candidate_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    def summarize(group: pd.DataFrame, *, rollup: str) -> dict[str, object]:
        gen_success = generation_success_mask(group)
        score_success = score_success_mask(group)
        scoreable = group.loc[score_success]
        passed = pass_mask(scoreable)
        pass_rate = float(passed.mean()) if score_success.any() else pd.NA
        gen_errors = (
            group["generation_status"] == GenerationRunStatus.ERROR.value
        ).sum()
        gen_blocked = (
            group["generation_status"] == GenerationRunStatus.BLOCKED.value
        ).sum()
        score_errors = (
            group["score_status"] == ScoreAttemptStatus.ERROR.value
        ).sum()
        return {
            "rollup": rollup,
            "model": group["model"].iloc[0],
            "provider_kind": (
                group["provider_kind"].iloc[0] if rollup == "detail" else ""
            ),
            "compression_target": (
                group["compression_target"].iloc[0]
                if rollup == "detail"
                else pd.NA
            ),
            "total_runs": len(group),
            "generation_success_rate": gen_success.mean(),
            "score_success_rate": score_success.mean(),
            "scoreable_count": int(score_success.sum()),
            "pass_count": int(passed.sum()),
            "pass_rate": pass_rate,
            "avg_provider_cost": group["total_provider_cost"].mean(),
            "generation_error_count": int(gen_errors),
            "generation_blocked_count": int(gen_blocked),
            "score_error_count": int(score_errors),
        }

    detail_rows = [
        summarize(group, rollup="detail")
        for _, group in frame.groupby(
            ["model", "provider_kind", "compression_target"],
            dropna=False,
        )
    ]
    model_rows = [
        summarize(group, rollup="model")
        for _, group in frame.groupby(["model"], dropna=False)
    ]
    return pd.DataFrame([*detail_rows, *model_rows])


def render_model_candidates_markdown(summary: pd.DataFrame) -> str:
    model_summary = summary[summary["rollup"] == "model"].sort_values(
        ["pass_rate", "scoreable_count"],
        ascending=[False, False],
        na_position="last",
    )
    lines = ["# Q1 model candidates", ""]
    if model_summary.empty:
        lines.append("No enc-dec rows loaded.")
        return "\n".join(lines)
    lines.extend(
        [
            "Ranked by pass rate on score-success rows, then scoreable count.",
            "",
        ]
    )
    for index, row in enumerate(model_summary.itertuples(), start=1):
        pass_rate = (
            f"{row.pass_rate:.1%}"
            if pd.notna(row.pass_rate)
            else "n/a"
        )
        caveats: list[str] = []
        if row.scoreable_count < 10:
            caveats.append("low scoreable count")
        if row.generation_success_rate < 0.95:
            caveats.append("generation failures present")
        if row.score_success_rate < 0.95:
            caveats.append("score failures present")
        if pd.notna(row.avg_provider_cost) and row.avg_provider_cost > 0.01:
            caveats.append("relatively high avg provider cost")
        caveat_text = f" Caveats: {', '.join(caveats)}." if caveats else ""
        lines.append(
            f"{index}. `{row.model}` — pass rate {pass_rate} "
            f"({row.pass_count}/{row.scoreable_count} scoreable)."
            f"{caveat_text}"
        )
    return "\n".join(lines)


def plot_pass_rate_by_model(summary: pd.DataFrame, output_path: Path) -> Path:
    apply_light_plot_style()
    model_summary = summary[summary["rollup"] == "model"].sort_values(
        "pass_rate",
        ascending=True,
        na_position="first",
    )
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(model_summary))))
    pass_rates = model_summary["pass_rate"].fillna(0.0)
    bars = ax.barh(model_summary["model"], pass_rates)
    for bar, rate in zip(bars, pass_rates, strict=True):
        if rate <= 0:
            continue
        ax.annotate(
            f"{rate:.0%}",
            xy=(rate, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pass rate (score-success rows)")
    ax.set_title(
        "Pass rate by model "
        f"(n={int(model_summary['scoreable_count'].sum())} scoreable runs)"
    )
    ax.grid(axis="x", alpha=0.3)
    return save_figure(fig, output_path)


def plot_generation_score_health(
    frame: pd.DataFrame,
    output_path: Path,
) -> Path:
    apply_light_plot_style()
    rows: list[dict[str, object]] = []
    for model, group in frame.groupby("model"):
        gen_success = int(generation_success_mask(group).sum())
        gen_other = int((~generation_success_mask(group)).sum())
        score_success = int(score_success_mask(group).sum())
        score_error = int(
            (group["score_status"] == ScoreAttemptStatus.ERROR.value).sum()
        )
        score_missing = int(group["score_status"].isna().sum())
        rows.append(
            {
                "model": model,
                "generation_success": gen_success,
                "generation_other": gen_other,
                "score_success": score_success,
                "score_error": score_error,
                "score_missing": score_missing,
            }
        )
    health = pd.DataFrame(rows).sort_values("model")
    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(health))))
    x = range(len(health))
    width = 0.35
    ax.bar(
        [value - width / 2 for value in x],
        health["generation_success"],
        width=width,
        label="generation success",
    )
    ax.bar(
        [value - width / 2 for value in x],
        health["generation_other"],
        width=width,
        bottom=health["generation_success"],
        label="generation other",
    )
    ax.bar(
        [value + width / 2 for value in x],
        health["score_success"],
        width=width,
        label="score success",
    )
    ax.bar(
        [value + width / 2 for value in x],
        health["score_error"],
        width=width,
        bottom=health["score_success"],
        label="score error",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(health["model"], rotation=20, ha="right")
    ax.set_ylabel("Run count")
    ax.set_title("Generation and score health by model")
    ax.legend(fontsize=8)
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
        title="Q1: Model candidates",
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

    summary = build_model_candidate_summary(frame)
    csv_path = output_run.artifact_path("model_candidates", suffix="csv")
    md_path = output_run.artifact_path("model_candidates", suffix="md")
    plot_path = output_run.path("pass_rate_by_model")
    health_plot_path = output_run.path("generation_score_health")
    summary.to_csv(csv_path, index=False)
    md_path.write_text(
        render_model_candidates_markdown(summary),
        encoding="utf-8",
    )
    figure_paths: list[Path] = []
    if not summary.empty:
        figure_paths.append(plot_pass_rate_by_model(summary, plot_path))
        figure_paths.append(
            plot_generation_score_health(frame, health_plot_path)
        )

    reporter.section("Model rankings")
    table = Table(title="Pass rate by model", header_style="bold cyan")
    columns = (
        "model",
        "scoreable_count",
        "pass_rate",
        "generation_success_rate",
    )
    for column in columns:
        table.add_column(column)
    model_summary = summary[summary["rollup"] == "model"]
    for row in model_summary.itertuples():
        table.add_row(
            str(row.model),
            str(row.scoreable_count),
            f"{row.pass_rate:.1%}" if pd.notna(row.pass_rate) else "n/a",
            f"{row.generation_success_rate:.1%}",
        )
    reporter.print_table(table)
    if model_summary.empty:
        reporter.print_note("No enc-dec rows matched the experiment filter.")
    reporter.finish(
        output_run=output_run,
        csv_paths=[csv_path],
        md_paths=[md_path],
        figure_paths=figure_paths,
    )


if __name__ == "__main__":
    run_typer_app(app)
