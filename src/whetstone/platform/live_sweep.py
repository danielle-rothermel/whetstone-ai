"""Fail-closed operator commands for the immutable HumanEval live sweep.

The ledger is deliberately local to an operator run: it contains identities and
money facts, never prompts, provider headers, or credentials.  ``--execute``
is the only path that can call Platform submission; provider work is performed
later by the existing worker.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from sqlalchemy import create_engine

from whetstone.platform.runtime import resolve_application_database_url
from whetstone.platform.spec_builder import (
    iter_experiment_specs_from_file,
    load_model_config_fragment,
)
from whetstone.platform.submission import submit_prediction_specs

APP = typer.Typer(no_args_is_help=True)
EXPECTED_CELLS = 5_904
CANARY_CELLS = 12
GENERATION_CEILING_USD = 4.62
MAX_RETRIES_PER_CELL = 2


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SweepLedger:
    """Run-scoped, WAL-backed journal with atomic ceiling checks."""

    def __init__(self, path: Path, *, manifest_hash: str) -> None:
        if not path.is_absolute():
            raise ValueError(
                "ledger path must be absolute and outside the repository"
            )
        self.path = path
        self.manifest_hash = manifest_hash
        self.connection = sqlite3.connect(
            path, isolation_level=None, timeout=30
        )
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sweep_cells (
              manifest_hash TEXT NOT NULL, cell_id TEXT NOT NULL,
              estimated_cost REAL NOT NULL, reserved_cost REAL,
              actual_cost REAL, operation_key TEXT, prediction_id TEXT,
              platform_item_id TEXT, platform_attempt INTEGER,
              status TEXT NOT NULL, retry_count INTEGER NOT NULL DEFAULT 0,
              retry_of_attempt INTEGER, error_classification TEXT,
              provider_tokens_json TEXT, created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL, PRIMARY KEY (manifest_hash, cell_id)
            );
            CREATE TABLE IF NOT EXISTS sweep_events (
              id INTEGER PRIMARY KEY, manifest_hash TEXT NOT NULL,
              cell_id TEXT NOT NULL, event TEXT NOT NULL,
              detail_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
            """
        )

    def close(self) -> None:
        self.connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

    def reserve(
        self, cells: list[dict[str, Any]], estimates: dict[str, float]
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        with self._transaction() as connection:
            totals = connection.execute(
                "SELECT COALESCE(SUM(actual_cost), 0), "
                "COALESCE(SUM(reserved_cost), 0) "
                "FROM sweep_cells WHERE manifest_hash=?",
                (self.manifest_hash,),
            ).fetchone()
            actual, reserved = float(totals[0]), float(totals[1])
            for cell in cells:
                cell_id = str(cell["cell_id"])
                estimate = estimates.get(cell_id)
                if estimate is None or estimate < 0:
                    raise ValueError(
                        f"unknown or invalid cost estimate for {cell_id}"
                    )
                row = connection.execute(
                    "SELECT status FROM sweep_cells "
                    "WHERE manifest_hash=? AND cell_id=?",
                    (self.manifest_hash, cell_id),
                ).fetchone()
                if row is not None:
                    continue
                if actual + reserved + estimate > GENERATION_CEILING_USD:
                    raise ValueError(
                        "authorized generation ceiling would be exceeded"
                    )
                timestamp = _now()
                connection.execute(
                    "INSERT INTO sweep_cells("
                    "manifest_hash,cell_id,estimated_cost,reserved_cost,"
                    "status,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (
                        self.manifest_hash,
                        cell_id,
                        estimate,
                        estimate,
                        "reserved",
                        timestamp,
                        timestamp,
                    ),
                )
                connection.execute(
                    "INSERT INTO sweep_events("
                    "manifest_hash,cell_id,event,detail_json,created_at) "
                    "VALUES(?,?,?,?,?)",
                    (self.manifest_hash, cell_id, "reserved", "{}", timestamp),
                )
                reserved += estimate
                selected.append(cell)
        return selected

    def submitted(
        self,
        cells: list[dict[str, Any]],
        *,
        operation_key: str,
        prediction_ids: dict[str, str],
    ) -> None:
        with self._transaction() as connection:
            for cell in cells:
                cell_id = str(cell["cell_id"])
                connection.execute(
                    "UPDATE sweep_cells SET status='submitted',"
                    "operation_key=?,prediction_id=?,updated_at=? "
                    "WHERE manifest_hash=? AND cell_id=?",
                    (
                        operation_key,
                        prediction_ids.get(cell_id),
                        _now(),
                        self.manifest_hash,
                        cell_id,
                    ),
                )

    def selected_remaining(
        self, all_cells: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT cell_id FROM sweep_cells WHERE manifest_hash=? "
            "AND status IN ('reserved','submitted','in_flight','succeeded')",
            (self.manifest_hash,),
        ).fetchall()
        excluded = {str(row[0]) for row in rows}
        return [
            cell for cell in all_cells if str(cell["cell_id"]) not in excluded
        ]

    def summary(self) -> dict[str, Any]:
        rows = self.connection.execute(
            "SELECT status, COUNT(*), COALESCE(SUM(reserved_cost),0), "
            "COALESCE(SUM(actual_cost),0) "
            "FROM sweep_cells WHERE manifest_hash=? GROUP BY status",
            (self.manifest_hash,),
        ).fetchall()
        return {
            str(status): {
                "count": count,
                "reserved_usd": reserved,
                "actual_usd": actual,
            }
            for status, count, reserved, actual in rows
        }


def validate_campaign(
    campaign_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    metadata = _load_json(campaign_dir / "campaign-metadata.json")
    manifest_path = campaign_dir / "manifest.jsonl"
    manifest_hash = _sha256(manifest_path)
    index = _load_json(campaign_dir / "manifest-index.json")
    if index.get("manifest_sha256") != manifest_hash:
        raise typer.BadParameter("manifest hash does not match locked index")
    cells = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        len(cells) != EXPECTED_CELLS
        or metadata.get("expected_cell_count") != EXPECTED_CELLS
    ):
        raise typer.BadParameter("campaign must contain exactly 5,904 cells")
    if len({cell["cell_id"] for cell in cells}) != len(cells):
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
    return metadata, cells, manifest_hash


def _estimates(path: Path, cells: list[dict[str, Any]]) -> dict[str, float]:
    """Read a locked per-cell estimate artifact; aggregates are unsafe."""
    payload = _load_json(path)
    if payload.get("manifest_sha256") is None or not isinstance(
        payload.get("cells"), dict
    ):
        raise typer.BadParameter(
            "estimate artifact must contain manifest_sha256 and cells map"
        )
    result = {
        str(key): float(value) for key, value in payload["cells"].items()
    }
    if set(result) != {str(cell["cell_id"]) for cell in cells}:
        raise typer.BadParameter(
            "estimate artifact must price every immutable cell exactly once"
        )
    return result


def _operation_key(metadata: dict[str, Any], suffix: str) -> str:
    return f"{metadata['campaign']}-generation-{suffix}"


def _specs_for_cells(
    campaign_dir: Path, cells: list[dict[str, Any]]
) -> dict[str, Any]:
    specs = iter_experiment_specs_from_file(
        campaign_dir / "requested-full-matrix-specification.json",
        configs_root=campaign_dir,
    )
    expected = {
        (
            str(c["task_id"]),
            int(c["repetition_seed"]),
            str(c["model"]),
            c["compression_target"],
        ): str(c["cell_id"])
        for c in cells
    }
    selected: dict[str, Any] = {}
    for spec in specs:
        target = spec.dimensions.values.get("compression_target")
        key = (
            spec.task_id,
            spec.repetition_seed,
            spec.provider_axis.model,
            target,
        )
        cell_id = expected.get(key)
        if cell_id is not None:
            selected[cell_id] = spec
    if len(selected) != len(cells):
        raise ValueError(
            "manifest/spec mapping is incomplete; refusing submission"
        )
    return selected


def _emit(
    command: str,
    *,
    cells: list[dict[str, Any]],
    manifest_hash: str,
    execute: bool,
    ledger: SweepLedger | None = None,
) -> None:
    typer.echo(
        json.dumps(
            {
                "command": command,
                "dry_run": not execute,
                "dispatch": execute,
                "cell_count": len(cells),
                "generation_ceiling_usd": GENERATION_CEILING_USD,
                "manifest_sha256": manifest_hash,
                "ledger": ledger.summary() if ledger else {},
            },
            sort_keys=True,
        )
    )


def _submit(
    campaign_dir: Path,
    metadata: dict[str, Any],
    cells: list[dict[str, Any]],
    ledger: SweepLedger,
    *,
    suffix: str,
) -> None:
    specs = _specs_for_cells(campaign_dir, cells)
    operation_key = _operation_key(metadata, suffix)
    engine = create_engine(resolve_application_database_url())
    submit_prediction_specs(
        engine,
        operation_key=operation_key,
        experiment_name=metadata["campaign"],
        specs=specs.values(),
        metadata={
            "manifest_sha256": ledger.manifest_hash,
            "operator": "whetstone-live-sweep",
        },
    )
    ledger.submitted(
        cells,
        operation_key=operation_key,
        prediction_ids={
            cell_id: spec.prediction_id for cell_id, spec in specs.items()
        },
    )


@APP.command()
def plan(
    campaign_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False)
    ],
) -> None:
    """Validate immutable matrix without touching Platform or providers."""
    _metadata, cells, manifest_hash = validate_campaign(campaign_dir)
    _emit("plan", cells=cells, manifest_hash=manifest_hash, execute=False)


@APP.command("submit-canary")
def submit_canary(
    campaign_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False)
    ],
    execute: Annotated[bool, typer.Option("--execute")] = False,
    ledger_path: Annotated[Path | None, typer.Option("--ledger")] = None,
    estimates_path: Annotated[Path | None, typer.Option("--estimates")] = None,
) -> None:
    """Reserve and submit exactly the stable 12-cell canary when confirmed."""
    metadata, cells, manifest_hash = validate_campaign(campaign_dir)
    selected = [
        cell
        for cell in cells
        if cell["task_id"] == "HumanEval/0" and cell["repetition_seed"] == 0
    ]
    if len(selected) != CANARY_CELLS:
        raise typer.BadParameter("canary must contain exactly 12 cells")
    if not execute:
        _emit(
            "submit-canary",
            cells=selected,
            manifest_hash=manifest_hash,
            execute=False,
        )
        return
    if ledger_path is None or estimates_path is None:
        raise typer.BadParameter(
            "--execute requires absolute --ledger and --estimates paths"
        )
    estimates = _estimates(estimates_path, cells)
    if _load_json(estimates_path).get("manifest_sha256") != manifest_hash:
        raise typer.BadParameter(
            "estimate artifact is for a different manifest"
        )
    ledger = SweepLedger(ledger_path, manifest_hash=manifest_hash)
    try:
        reserved = ledger.reserve(selected, estimates)
        if reserved:
            _submit(campaign_dir, metadata, reserved, ledger, suffix="canary")
        _emit(
            "submit-canary",
            cells=reserved,
            manifest_hash=manifest_hash,
            execute=True,
            ledger=ledger,
        )
    finally:
        ledger.close()


@APP.command("submit-remaining")
def submit_remaining(
    campaign_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False)
    ],
    execute: Annotated[bool, typer.Option("--execute")] = False,
    ledger_path: Annotated[Path | None, typer.Option("--ledger")] = None,
    estimates_path: Annotated[Path | None, typer.Option("--estimates")] = None,
    page_size: Annotated[int, typer.Option(min=1, max=500)] = 100,
) -> None:
    """Submit cells not already reserved, submitted, or successful."""
    metadata, cells, manifest_hash = validate_campaign(campaign_dir)
    if not execute:
        _emit(
            "submit-remaining",
            cells=cells[CANARY_CELLS:],
            manifest_hash=manifest_hash,
            execute=False,
        )
        return
    if ledger_path is None or estimates_path is None:
        raise typer.BadParameter(
            "--execute requires absolute --ledger and --estimates paths"
        )
    estimates = _estimates(estimates_path, cells)
    ledger = SweepLedger(ledger_path, manifest_hash=manifest_hash)
    try:
        remaining = ledger.selected_remaining(cells)
        for start in range(0, len(remaining), page_size):
            reserved = ledger.reserve(
                remaining[start : start + page_size], estimates
            )
            if reserved:
                _submit(
                    campaign_dir,
                    metadata,
                    reserved,
                    ledger,
                    suffix=f"remaining-{start // page_size:04d}",
                )
        _emit(
            "submit-remaining",
            cells=remaining,
            manifest_hash=manifest_hash,
            execute=True,
            ledger=ledger,
        )
    finally:
        ledger.close()


@APP.command("submit-retry")
def submit_retry(
    campaign_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False)
    ],
    execute: Annotated[bool, typer.Option("--execute")] = False,
) -> None:
    """Fail closed until typed attempt-to-cell reconciliation is available."""
    validate_campaign(campaign_dir)
    if execute:
        raise typer.BadParameter(
            "retry needs typed Platform/Whetstone reconciliation; "
            "no cells selected"
        )
    typer.echo(
        json.dumps(
            {
                "command": "submit-retry",
                "dry_run": True,
                "dispatch": False,
                "cell_count": 0,
            },
            sort_keys=True,
        )
    )


@APP.command()
def status(
    campaign_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False)
    ],
    ledger_path: Annotated[Path | None, typer.Option("--ledger")] = None,
) -> None:
    _metadata, cells, manifest_hash = validate_campaign(campaign_dir)
    if ledger_path is None:
        _emit(
            "status", cells=cells, manifest_hash=manifest_hash, execute=False
        )
        return
    ledger = SweepLedger(ledger_path, manifest_hash=manifest_hash)
    try:
        _emit(
            "status",
            cells=cells,
            manifest_hash=manifest_hash,
            execute=False,
            ledger=ledger,
        )
    finally:
        ledger.close()


def main() -> None:
    APP()


if __name__ == "__main__":
    main()
