#!/usr/bin/env python3
"""Spot-check one enc-dec run end-to-end as a horizontal HTML report."""

from __future__ import annotations

import typer
from rich.panel import Panel

from dr_dspy.analysis.cli_options import (
    DatabaseUrlOption,
    EnvFileOption,
    LimitOption,
)
from dr_dspy.analysis.db import create_analysis_engine
from dr_dspy.analysis.figures import SampleInspectionRun
from dr_dspy.analysis.inspect import (
    SCRIPT_NAME,
    SampleIndexError,
    build_debug_metadata,
    is_passing_run,
    list_encdec_run_index,
    load_run_bundle,
    reconstruct_prompts,
    resolve_sample_index,
    summarize_test_results,
)
from dr_dspy.analysis.sample_html import write_sample_report
from dr_dspy.platform.cli_env import load_env_file, run_typer_app

app = typer.Typer(add_completion=False)


@app.command()
def main(
    experiment_name: str = typer.Option(
        ...,
        "--experiment-name",
        help="Experiment name to inspect.",
    ),
    sample_index: int = typer.Option(
        ...,
        "--sample-index",
        help="0-based index into enc-dec runs ordered by fair_order_key.",
    ),
    require_score: bool = typer.Option(
        True,
        "--require-score/--no-require-score",
        help=(
            "Only include runs with a humaneval@v1 score attempt "
            "in the index."
        ),
    ),
    database_url: DatabaseUrlOption = None,
    env_file: EnvFileOption = None,
    limit: LimitOption = None,
) -> None:
    if env_file is not None:
        load_env_file(env_file)
    else:
        load_env_file()
    engine = create_analysis_engine(database_url, env_file=None)
    index_rows = list_encdec_run_index(
        engine,
        experiment_name,
        require_score=require_score,
        limit=limit,
    )
    try:
        index_row = resolve_sample_index(index_rows, sample_index)
    except SampleIndexError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    bundle = load_run_bundle(
        engine,
        index_row,
        sample_index=sample_index,
        sample_count=len(index_rows),
    )
    reconstructed_prompts, reconstruction_errors = reconstruct_prompts(bundle)

    output_run = SampleInspectionRun.start(experiment_name, sample_index)
    output_run.ensure_directory()
    html_path = output_run.html_path()
    json_path = output_run.json_path()

    metadata = build_debug_metadata(
        bundle,
        reconstructed_prompts=reconstructed_prompts,
        reconstruction_errors=reconstruction_errors,
        output_paths={
            "html": str(html_path.resolve()),
            "json": str(json_path.resolve()),
        },
    )
    write_sample_report(
        bundle=bundle,
        metadata=metadata,
        reconstructed_prompts=reconstructed_prompts,
        reconstruction_errors=reconstruction_errors,
        html_path=html_path,
        json_path=json_path,
    )

    test_summary = summarize_test_results(bundle.score_attempt)
    outcome = "PASS" if is_passing_run(bundle) else "FAIL / incomplete"
    score_status = (
        bundle.score_attempt.status.value
        if bundle.score_attempt is not None
        else "missing"
    )
    body = (
        f"[bold]Task[/]              {bundle.spec.task_id}\n"
        f"[bold]Sample index[/]    {sample_index} "
        f"(of {len(index_rows)} runs)\n"
        f"[bold]Outcome[/]          {outcome}\n"
        f"[bold]Generation[/]      {bundle.generation_run.status.value}\n"
        f"[bold]Score status[/]    {score_status}\n"
        f"[bold]Tests failed[/]    {test_summary['failed']} / "
        f"{test_summary['total']}\n"
        f"[bold]Prediction ID[/]   {bundle.spec.prediction_id}\n"
        f"[bold]Generation run[/]  {bundle.generation_run.generation_run_id}"
    )
    if reconstruction_errors:
        body += "\n\n[yellow]Prompt reconstruction warnings:[/]\n"
        body += "\n".join(f"  • {error}" for error in reconstruction_errors)

    from rich.console import Console

    console = Console(width=120)
    console.print()
    console.print(
        Panel(
            body,
            title="Sample run inspector",
            subtitle=SCRIPT_NAME,
            border_style="blue",
            padding=(1, 2),
        )
    )
    console.print()
    console.print(
        Panel(
            str(html_path.resolve()),
            title="HTML report — open in browser",
            border_style="green",
            padding=(0, 2),
        )
    )
    console.print(
        Panel(
            str(json_path.resolve()),
            title="JSON debug bundle — use with jq",
            border_style="yellow",
            padding=(0, 2),
        )
    )
    console.print()
    console.print(
        f"[dim]Example:[/] jq '.spec.prediction_id' {json_path.resolve()}"
    )
    console.print()


if __name__ == "__main__":
    run_typer_app(app)
