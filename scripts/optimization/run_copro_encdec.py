#!/usr/bin/env python3
"""Run a minimal COPRO-style enc-dec optimizer loop."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Annotated

import typer
from dr_platform import destroy_dbos_runtime
from dr_providers import EndpointKind, ProviderKind
from rich.console import Console
from sqlalchemy import create_engine

from whetstone.optimization.copro import (
    CoproExecutionMode,
    CoproProposalMode,
    CoproRunConfig,
    append_testing_log_entry,
    run_copro_loop,
    write_copro_artifacts,
)
from whetstone.platform.cli_env import load_env_file, run_typer_app
from whetstone.platform.spec_builder import DEFAULT_CONFIGS_ROOT
from whetstone.platform.worker import configure_platform_dbos_runtime

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TESTING_LOG = REPO_ROOT / "docs" / "testing_logs.md"


def _format_command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


@app.command()
def main(
    model_config: Annotated[
        Path,
        typer.Option(
            "--model-config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    split: Annotated[
        Path,
        typer.Option(
            "--split",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    compression_target: Annotated[
        list[float],
        typer.Option("--compression-target"),
    ],
    breadth: Annotated[int, typer.Option("--breadth", min=1)] = 3,
    depth: Annotated[int, typer.Option("--depth", min=1)] = 2,
    repeats: Annotated[int, typer.Option("--repeats", min=1)] = 1,
    proposal_mode: Annotated[
        CoproProposalMode,
        typer.Option("--proposal-mode", case_sensitive=False),
    ] = CoproProposalMode.MANUAL,
    prompt_model: Annotated[
        str | None,
        typer.Option("--prompt-model"),
    ] = None,
    prompt_provider_kind: Annotated[
        ProviderKind,
        typer.Option("--prompt-provider-kind", case_sensitive=False),
    ] = ProviderKind.OPENAI,
    prompt_endpoint_kind: Annotated[
        EndpointKind,
        typer.Option("--prompt-endpoint-kind", case_sensitive=False),
    ] = EndpointKind.RESPONSES,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir", file_okay=False, dir_okay=True, writable=True
        ),
    ] = Path("artifacts/optimization/copro_smoke"),
    database_url: Annotated[
        str | None,
        typer.Option("--database-url"),
    ] = None,
    env_file: Annotated[Path | None, typer.Option("--env-file")] = None,
    configs_root: Annotated[
        Path,
        typer.Option(
            "--configs-root", file_okay=False, dir_okay=True, readable=True
        ),
    ] = DEFAULT_CONFIGS_ROOT,
    execution_mode: Annotated[
        CoproExecutionMode,
        typer.Option("--execution-mode", case_sensitive=False),
    ] = CoproExecutionMode.SYNC,
    rescore_max_in_flight: Annotated[
        int,
        typer.Option("--rescore-max-in-flight", min=1),
    ] = 100,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    append_testing_log: Annotated[
        bool,
        typer.Option(
            "--append-testing-log/--no-append-testing-log",
            help="Append a short run summary to docs/testing_logs.md.",
        ),
    ] = True,
    testing_log_path: Annotated[
        Path,
        typer.Option("--testing-log-path", file_okay=True, dir_okay=False),
    ] = DEFAULT_TESTING_LOG,
) -> None:
    load_env_file(env_file) if env_file is not None else load_env_file()
    if not compression_target:
        raise typer.BadParameter(
            "At least one --compression-target is required."
        )
    command = _format_command()
    run_config = CoproRunConfig(
        model_config_path=model_config,
        split_path=split,
        compression_targets=tuple(compression_target),
        breadth=breadth,
        depth=depth,
        repetition_seeds=tuple(range(repeats)),
        proposal_mode=proposal_mode,
        execution_mode=execution_mode,
        prompt_model=prompt_model,
        prompt_provider_kind=prompt_provider_kind,
        prompt_endpoint_kind=prompt_endpoint_kind,
        output_dir=output_dir,
        configs_root=configs_root,
        rescore_max_in_flight=rescore_max_in_flight,
        dry_run=dry_run,
    )
    commands = [
        f"# execution_mode={execution_mode.value}",
        command,
    ]
    if dry_run:
        engine = create_engine("sqlite:///:memory:")
        result = run_copro_loop(
            engine,
            database_url="sqlite:///:memory:",
            config=run_config,
        )
        result = result.model_copy(update={"command": command})
        artifact_paths = write_copro_artifacts(
            result,
            output_dir=output_dir,
            commands=commands,
        )
        console.print(
            {
                "dry_run": True,
                "candidates": len(result.candidates),
                "artifacts": {
                    key: str(path) for key, path in artifact_paths.items()
                },
            }
        )
        return

    launched_dbos = False
    try:
        dbos_config = configure_platform_dbos_runtime(
            database_url=database_url,
            dbos_system_database_url=None,
            consume_generation_queue=False,
            database_url_error_suffix="for COPRO enc-dec optimizer",
        )
        launched_dbos = True
        engine = create_engine(dbos_config.database_url)
        try:
            result = run_copro_loop(
                engine,
                database_url=dbos_config.database_url,
                config=run_config,
            )
            result = result.model_copy(update={"command": command})
            artifact_paths = write_copro_artifacts(
                result,
                output_dir=output_dir,
                commands=commands,
            )
            if execution_mode is CoproExecutionMode.QUEUE:
                commands.append(
                    "# queue mode requires: "
                    "uv run python -m whetstone.platform.worker worker"
                )
            console.print(
                {
                    "run_id": result.run_id,
                    "experiment_name": result.experiment_name,
                    "best_candidate": (
                        result.best_candidate.candidate_id
                        if result.best_candidate is not None
                        else None
                    ),
                    "best_pass_rate": (
                        result.best_attempt.pass_rate
                        if result.best_attempt is not None
                        else None
                    ),
                    "artifacts": {
                        key: str(path) for key, path in artifact_paths.items()
                    },
                }
            )
            if append_testing_log:
                verdict = (
                    "PASS" if result.best_attempt is not None else "INCOMPLETE"
                )
                if result.caveats:
                    verdict = (
                        f"{verdict}; caveats: {'; '.join(result.caveats)}"
                    )
                append_testing_log_entry(
                    testing_log_path=testing_log_path,
                    result=result,
                    artifact_paths=artifact_paths,
                    verdict=verdict,
                )
        finally:
            engine.dispose()
    finally:
        if launched_dbos:
            destroy_dbos_runtime()


if __name__ == "__main__":
    run_typer_app(app)
