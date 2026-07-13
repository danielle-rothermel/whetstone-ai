"""Fail-closed, dry-run operator interface for the approved HumanEval sweep.

This command deliberately has no provider dispatch path.  It validates the
immutable campaign inputs and writes no queue state; a separately approved
operator may use the existing submission API after this preflight passes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from whetstone.platform.spec_builder import load_model_config_fragment

APP = typer.Typer(no_args_is_help=True)
EXPECTED_CELLS = 5_904
CANARY_CELLS = 12
GENERATION_CEILING_USD = 4.62


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_campaign(
    campaign_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = _load_json(campaign_dir / "campaign-metadata.json")
    manifest_path = campaign_dir / "manifest.jsonl"
    cells = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        len(cells) != EXPECTED_CELLS
        or metadata["expected_cell_count"] != EXPECTED_CELLS
    ):
        raise typer.BadParameter("campaign must contain exactly 5,904 cells")
    cell_ids = [cell["cell_id"] for cell in cells]
    if len(cell_ids) != len(set(cell_ids)):
        raise typer.BadParameter("campaign cell IDs must be unique")
    if {cell["budget_key"] for cell in cells} != {
        "direct",
        "1",
        "0.75",
        "0.5",
    }:
        raise typer.BadParameter(
            "campaign budgets do not match approved matrix"
        )
    for model in metadata["models"]:
        fragment = campaign_dir / "models" / f"{model['slug']}-openrouter.json"
        if model["provider_kind"] == "openai":
            fragment = campaign_dir / "models" / "gpt54-nano-openai.json"
        validated = load_model_config_fragment(fragment)
        if any(
            provider.model != model["model"]
            for provider in validated.providers
        ):
            raise typer.BadParameter("model fragment does not match campaign")
    return metadata, cells


def _emit(
    command: str, campaign_dir: Path, cells: list[dict[str, Any]]
) -> None:
    typer.echo(
        json.dumps(
            {
                "command": command,
                "dry_run": True,
                "dispatch": False,
                "cell_count": len(cells),
                "generation_ceiling_usd": GENERATION_CEILING_USD,
                "manifest_sha256": _sha256(campaign_dir / "manifest.jsonl"),
            },
            sort_keys=True,
        )
    )


@APP.command()
def plan(
    campaign_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False)
    ],
) -> None:
    """Validate the immutable matrix and print its no-spend plan."""
    _metadata, cells = validate_campaign(campaign_dir)
    _emit("plan", campaign_dir, cells)


@APP.command("submit-canary")
def submit_canary(
    campaign_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False)
    ],
) -> None:
    """Select exactly one task's 12 cells; always dry-run and no-dispatch."""
    _metadata, cells = validate_campaign(campaign_dir)
    selected = [
        cell
        for cell in cells
        if cell["task_id"] == "HumanEval/0" and cell["repetition_seed"] == 0
    ]
    if len(selected) != CANARY_CELLS:
        raise typer.BadParameter("canary must contain exactly 12 cells")
    _emit("submit-canary", campaign_dir, selected)


@APP.command("submit-remaining")
def submit_remaining(
    campaign_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False)
    ],
) -> None:
    """Print the 5,892 non-canary cells without queue or provider mutation."""
    _metadata, cells = validate_campaign(campaign_dir)
    selected = [
        cell
        for cell in cells
        if not (
            cell["task_id"] == "HumanEval/0" and cell["repetition_seed"] == 0
        )
    ]
    _emit("submit-remaining", campaign_dir, selected)


@APP.command("submit-retry")
def submit_retry(
    campaign_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False)
    ],
) -> None:
    """Retries require persisted typed attempt state; refuse without it."""
    validate_campaign(campaign_dir)
    raise typer.BadParameter(
        "retry selection requires persisted platform attempts"
    )


@APP.command()
def status(
    campaign_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False)
    ],
) -> None:
    _metadata, cells = validate_campaign(campaign_dir)
    _emit("status", campaign_dir, cells)


def main() -> None:
    APP()


if __name__ == "__main__":
    main()
