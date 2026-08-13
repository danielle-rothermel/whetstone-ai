from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("run")
def run_command(
    control_ref: str = typer.Option(..., "--control-ref", help="Optimizer control ref"),
) -> None:
    """Submit an optimization run via dr-platform (stub)."""
    typer.echo(
        "whetstone-optim run is not yet wired to a live platform deployment; "
        f"received control_ref={control_ref!r}"
    )
    raise typer.Exit(code=2)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
