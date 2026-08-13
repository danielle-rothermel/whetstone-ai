from __future__ import annotations

import typer

from whetstone.sandbox.copro_step import run_copro_seed_preview
from whetstone.sandbox.gepa_step import run_gepa_step_preview
from whetstone.sandbox.graph_step import run_toy_graph_preview
from whetstone.sandbox.miprov2_step import run_miprov2_plan_preview
from whetstone.sandbox.render import (
    render_copro_preview,
    render_gepa_preview,
    render_graph_preview,
    render_miprov2_preview,
)

app = typer.Typer(
    help="Generic optimizer step previews on the whetstone toy graph.",
    no_args_is_help=True,
)


@app.command("copro")
def copro_command(
    breadth: int = typer.Option(3, min=2, help="COPRO breadth (>1)."),
    depth: int = typer.Option(1, min=1, help="Configured COPRO depth."),
    task_prompt: str = typer.Option(
        "Say hello to the user.",
        help="Baseline instruction body for the toy task.",
    ),
) -> None:
    from rich.console import Console

    transcript = run_copro_seed_preview(
        breadth=breadth,
        depth=depth,
        task_prompt=task_prompt,
    )
    render_copro_preview(Console(), transcript)


@app.command("miprov2")
def miprov2_command(
    round: int = typer.Option(0, min=0, help="Planner round index to preview."),
) -> None:
    from rich.console import Console

    preview = run_miprov2_plan_preview(round_index=round)
    render_miprov2_preview(Console(), preview)


@app.command("gepa")
def gepa_command(
    show_intent: bool = typer.Option(
        False,
        "--show-intent",
        help="Include the evaluation-intent boundary note.",
    ),
) -> None:
    from rich.console import Console

    preview = run_gepa_step_preview(show_intent=show_intent)
    render_gepa_preview(Console(), preview)


@app.command("graph")
def graph_command(
    run: bool = typer.Option(
        False,
        "--run",
        help="Execute the toy single-node graph with stub transport.",
    ),
    prompt: str = typer.Option("hello sandbox", help="External graph prompt input."),
) -> None:
    from rich.console import Console

    if not run:
        typer.echo("Pass --run to exercise the toy graph stack.")
        raise typer.Exit(code=0)
    preview = run_toy_graph_preview(prompt=prompt)
    render_graph_preview(Console(), preview)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
