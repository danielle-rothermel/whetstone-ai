"""Fail-closed operator tooling for live-sweep estimates and run stores."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from dbos import DBOS
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from whetstone.platform.platform_db import ensure_whetstone_application_schema
from whetstone.platform.runtime import shutdown_dbos_runtime

EXPECTED_CELLS = 5904
MAX_GENERATION_USD = Decimal("4.62")
SCHEMA_VERSION = 1
_RUN_ID = re.compile(r"^[a-z][a-z0-9_]{2,23}$")
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
NonEmpty = Annotated[StrictStr, Field(min_length=1)]

APP = typer.Typer(no_args_is_help=True)
ESTIMATES = typer.Typer(no_args_is_help=True)
STORES = typer.Typer(no_args_is_help=True)
APP.add_typer(ESTIMATES, name="estimates")
APP.add_typer(STORES, name="stores")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{field} must be an exact decimal") from error
    if not result.is_finite() or result < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


class ModelPrice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_usd_per_million: StrictStr
    output_usd_per_million: StrictStr
    assumed_input_tokens: StrictInt = Field(ge=0)
    assumed_output_tokens: StrictInt = Field(ge=0)

    def estimate(self) -> Decimal:
        input_price = _decimal(
            self.input_usd_per_million,
            field="input_usd_per_million",
        )
        output_price = _decimal(
            self.output_usd_per_million,
            field="output_usd_per_million",
        )
        return (
            input_price * self.assumed_input_tokens
            + output_price * self.assumed_output_tokens
        ) / Decimal(1_000_000)


class PriceBook(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    effective_at: NonEmpty
    currency: Literal["USD"]
    assumptions_version: NonEmpty
    source: NonEmpty
    models: dict[NonEmpty, ModelPrice]


def _campaign_cells(campaign_dir: Path) -> list[dict[str, Any]]:
    path = campaign_dir / "manifest.jsonl"
    cells = [json.loads(line) for line in path.read_text().splitlines()]
    identities = {str(cell.get("cell_id")) for cell in cells}
    if len(cells) != EXPECTED_CELLS or len(identities) != EXPECTED_CELLS:
        raise ValueError("campaign must contain exactly 5,904 unique cells")
    return cells


def generate_estimates(
    campaign_dir: Path, price_book_path: Path
) -> dict[str, object]:
    """Build a deterministic estimate artifact without provider access."""
    cells = _campaign_cells(campaign_dir)
    price_book_payload = json.loads(price_book_path.read_text())
    price_book = PriceBook.model_validate(price_book_payload)
    unknown = sorted(
        {str(cell["model"]) for cell in cells} - set(price_book.models)
    )
    if unknown:
        raise ValueError(f"price book does not cover models: {unknown}")
    estimates = {
        str(cell["cell_id"]): format(
            price_book.models[str(cell["model"])].estimate(), "f"
        )
        for cell in cells
    }
    total = sum(
        (
            _decimal(value, field="cell estimate")
            for value in estimates.values()
        ),
        Decimal(),
    )
    if total > MAX_GENERATION_USD:
        raise ValueError(
            f"estimated total {total} exceeds ceiling {MAX_GENERATION_USD}"
        )
    manifest_path = campaign_dir / "manifest.jsonl"
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": _file_sha256(manifest_path),
        "cells": estimates,
        "summary": {
            "cell_count": EXPECTED_CELLS,
            "total_usd": format(total, "f"),
            "ceiling_usd": format(MAX_GENERATION_USD, "f"),
        },
        "provenance": {
            "price_book_sha256": _sha256(price_book_payload),
            "price_book_schema_version": price_book.schema_version,
            "effective_at": price_book.effective_at,
            "assumptions_version": price_book.assumptions_version,
            "source": price_book.source,
            "implicit_price_fetch": False,
        },
    }
    payload["artifact_sha256"] = _sha256(payload)
    return payload


def validate_estimates(campaign_dir: Path, artifact_path: Path) -> None:
    """Validate full coverage, provenance hash, and spend ceiling."""
    cells = _campaign_cells(campaign_dir)
    payload = json.loads(artifact_path.read_text())
    artifact_hash = payload.pop("artifact_sha256", None)
    if artifact_hash != _sha256(payload):
        raise ValueError("estimate artifact checksum does not match")
    if payload.get("manifest_sha256") != _file_sha256(
        campaign_dir / "manifest.jsonl"
    ):
        raise ValueError("estimate artifact is for a different manifest")
    estimates = payload.get("cells")
    if not isinstance(estimates, dict) or set(estimates) != {
        str(cell["cell_id"]) for cell in cells
    }:
        raise ValueError(
            "estimate artifact must price every cell exactly once"
        )
    total = sum(
        (
            _decimal(value, field=f"estimate for {key}")
            for key, value in estimates.items()
        ),
        Decimal(),
    )
    if total > MAX_GENERATION_USD:
        raise ValueError("estimate artifact exceeds the $4.62 ceiling")
    summary = payload.get("summary")
    if not isinstance(summary, dict) or summary.get("total_usd") != format(
        total, "f"
    ):
        raise ValueError("estimate summary does not match cell total")


@ESTIMATES.command("generate")
def estimates_generate(
    campaign_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False)
    ],
    price_book: Annotated[Path, typer.Option("--price-book", exists=True)],
    output: Annotated[Path, typer.Option("--output")],
    execute: Annotated[bool, typer.Option("--execute")] = False,
) -> None:
    """Generate a locked artifact; dry-run validates and prints only facts."""
    payload = generate_estimates(campaign_dir, price_book)
    if execute:
        if output.exists():
            raise typer.BadParameter("refusing to replace estimate artifact")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    typer.echo(
        json.dumps(
            {
                "dry_run": not execute,
                "output": str(output),
                "artifact_sha256": payload["artifact_sha256"],
                "summary": payload["summary"],
            },
            sort_keys=True,
        )
    )


@ESTIMATES.command("validate")
def estimates_validate(
    campaign_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False)
    ],
    artifact: Annotated[Path, typer.Option("--artifact", exists=True)],
) -> None:
    validate_estimates(campaign_dir, artifact)
    typer.echo("verified")


class StoreBoundary(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True
    )

    environment: StrictStr
    schema_name: StrictStr = Field(alias="schema")


class StoreDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    run_id: StrictStr
    created_at: StrictStr
    source: StoreBoundary
    dbos_path: StrictStr
    motherduck: StoreBoundary
    neon: StoreBoundary
    journal_path: StrictStr


def _store_descriptor(run_id: str, descriptor_path: Path) -> StoreDescriptor:
    if _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run ID must match [a-z][a-z0-9_]{2,23}")
    return StoreDescriptor(
        schema_version=1,
        run_id=run_id,
        created_at=datetime.now(UTC).isoformat(),
        source=StoreBoundary(
            environment="DATABASE_URL", schema=f"whetstone_run_{run_id}"
        ),
        dbos_path=f"{run_id}-dbos.sqlite3",
        motherduck=StoreBoundary(
            environment="MOTHERDUCK_DATABASE_URL",
            schema=f"whetstone_analysis_{run_id}",
        ),
        neon=StoreBoundary(
            environment="NEON_DATABASE_URL",
            schema=f"whetstone_detail_{run_id}",
        ),
        journal_path=descriptor_path.name + ".journal.json",
    )


def _bound_url(value: str, schema: str) -> str:
    url = make_url(value)
    return str(
        url.update_query_dict({"options": f"-c search_path={schema},public"})
    )


def _require_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _schema_exists(url: str, schema: str) -> bool:
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            return bool(
                connection.execute(
                    text(
                        "SELECT EXISTS (SELECT 1 FROM "
                        "information_schema.schemata "
                        "WHERE schema_name=:schema)"
                    ),
                    {"schema": schema},
                ).scalar_one()
            )
    finally:
        engine.dispose()


def _create_schema(url: str, schema: str) -> None:
    if _IDENTIFIER.fullmatch(schema) is None:
        raise ValueError("invalid generated schema")
    engine = create_engine(url)
    try:
        with engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    finally:
        engine.dispose()


def _drop_schema(url: str, schema: str) -> None:
    if _IDENTIFIER.fullmatch(schema) is None:
        raise ValueError("invalid generated schema")
    engine = create_engine(url)
    try:
        with engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
    finally:
        engine.dispose()


def _initialize_dbos_store(path: Path, run_id: str) -> None:
    """Run DBOS's supported initialization against the isolated store."""
    app_suffix = hashlib.sha256(run_id.encode()).hexdigest()[:12]
    try:
        DBOS(
            config={
                "name": f"whetstone-cutover-{app_suffix}",
                "system_database_url": "sqlite:///" + str(path.resolve()),
            }
        )
        DBOS.launch()
    finally:
        shutdown_dbos_runtime()


def prepare_stores(descriptor_path: Path, run_id: str) -> StoreDescriptor:
    """Create fresh schemas and migrate the source; journal before mutation."""
    descriptor = _store_descriptor(run_id, descriptor_path)
    if (
        descriptor_path.exists()
        or (descriptor_path.parent / descriptor.journal_path).exists()
    ):
        raise ValueError("descriptor or journal already exists")
    boundaries = (descriptor.source, descriptor.motherduck, descriptor.neon)
    for boundary in boundaries:
        url = _require_environment(boundary.environment)
        if _schema_exists(url, boundary.schema_name):
            raise ValueError(
                f"refusing schema collision: {boundary.schema_name}"
            )
    dbos_path = descriptor_path.parent / descriptor.dbos_path
    if dbos_path.exists():
        raise ValueError("refusing DBOS path collision")
    journal_path = descriptor_path.parent / descriptor.journal_path
    journal = {
        "schema_version": 1,
        "run_id": run_id,
        "descriptor_sha256": _sha256(descriptor.model_dump(mode="json")),
        "created": [],
        "cleanup_complete": False,
    }
    journal_path.write_text(
        json.dumps(journal, indent=2, sort_keys=True) + "\n"
    )
    try:
        for boundary in boundaries:
            _create_schema(
                _require_environment(boundary.environment),
                boundary.schema_name,
            )
            journal["created"].append(boundary.schema_name)
            journal_path.write_text(
                json.dumps(journal, indent=2, sort_keys=True) + "\n"
            )
        ensure_whetstone_application_schema(
            _bound_url(
                _require_environment(descriptor.source.environment),
                descriptor.source.schema_name,
            )
        )
        _initialize_dbos_store(dbos_path, run_id)
        descriptor_path.write_text(descriptor.model_dump_json(indent=2) + "\n")
        return descriptor
    except Exception:
        for boundary in reversed(boundaries):
            if boundary.schema_name in journal["created"]:
                try:
                    _drop_schema(
                        _require_environment(boundary.environment),
                        boundary.schema_name,
                    )
                except Exception:
                    pass
        dbos_path.unlink(missing_ok=True)
        raise


def validate_store_state(descriptor_path: Path) -> StoreDescriptor:
    descriptor = StoreDescriptor.model_validate_json(
        descriptor_path.read_text()
    )
    _validate_descriptor_journal(descriptor_path, descriptor)
    for boundary in (
        descriptor.source,
        descriptor.motherduck,
        descriptor.neon,
    ):
        if not _schema_exists(
            _require_environment(boundary.environment), boundary.schema_name
        ):
            raise ValueError(
                f"missing run-owned schema: {boundary.schema_name}"
            )
    dbos_path = descriptor_path.parent / descriptor.dbos_path
    if not dbos_path.is_file() or dbos_path.stat().st_size == 0:
        raise ValueError("missing run-owned DBOS store")
    return descriptor


def _validate_descriptor_journal(
    descriptor_path: Path, descriptor: StoreDescriptor
) -> dict[str, object]:
    journal_path = descriptor_path.parent / descriptor.journal_path
    journal = json.loads(journal_path.read_text())
    if journal.get("descriptor_sha256") != _sha256(
        descriptor.model_dump(mode="json")
    ):
        raise ValueError("descriptor and journal disagree")
    if journal.get("run_id") != descriptor.run_id:
        raise ValueError("journal run identity does not match descriptor")
    return journal


@STORES.command("prepare")
def stores_prepare(
    run_id: Annotated[str, typer.Option("--run-id")],
    descriptor: Annotated[Path, typer.Option("--descriptor")],
    execute: Annotated[bool, typer.Option("--execute")] = False,
    confirm: Annotated[str | None, typer.Option("--confirm")] = None,
) -> None:
    """Plan or create fresh run-owned boundaries."""
    planned = _store_descriptor(run_id, descriptor)
    if not execute:
        typer.echo(planned.model_dump_json())
        return
    if confirm != run_id:
        raise typer.BadParameter(
            "--execute requires --confirm equal to run ID"
        )
    descriptor.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(prepare_stores(descriptor, run_id).model_dump_json())


@STORES.command("verify")
def stores_verify(
    descriptor: Annotated[Path, typer.Option("--descriptor", exists=True)],
) -> None:
    validate_store_state(descriptor)
    typer.echo("verified")


@STORES.command("run")
def stores_run(
    descriptor: Annotated[Path, typer.Option("--descriptor", exists=True)],
    command: Annotated[list[str], typer.Argument()],
) -> None:
    """Run a command with secret-safe, schema-bound child environment."""
    facts = validate_store_state(descriptor)
    if not command:
        raise typer.BadParameter("command is required")
    environment = os.environ.copy()
    environment["DATABASE_URL"] = _bound_url(
        _require_environment(facts.source.environment),
        facts.source.schema_name,
    )
    environment["DBOS_SYSTEM_DATABASE_URL"] = "sqlite:///" + str(
        (descriptor.parent / facts.dbos_path).resolve()
    )
    environment["MOTHERDUCK_DATABASE_URL"] = _bound_url(
        _require_environment(facts.motherduck.environment),
        facts.motherduck.schema_name,
    )
    environment["NEON_DATABASE_URL"] = _bound_url(
        _require_environment(facts.neon.environment), facts.neon.schema_name
    )
    raise typer.Exit(
        subprocess.run(command, env=environment, check=False).returncode
    )


@STORES.command("cleanup")
def stores_cleanup(
    descriptor: Annotated[Path, typer.Option("--descriptor", exists=True)],
    execute: Annotated[bool, typer.Option("--execute")] = False,
    confirm: Annotated[str | None, typer.Option("--confirm")] = None,
) -> None:
    """Plan or remove only descriptor-owned boundaries."""
    facts = StoreDescriptor.model_validate_json(descriptor.read_text())
    if not execute:
        typer.echo(json.dumps({"dry_run": True, "run_id": facts.run_id}))
        return
    if confirm != facts.run_id:
        raise typer.BadParameter(
            "--execute requires --confirm equal to run ID"
        )
    _validate_descriptor_journal(descriptor, facts)
    for boundary in (facts.neon, facts.motherduck, facts.source):
        url = _require_environment(boundary.environment)
        if _schema_exists(url, boundary.schema_name):
            _drop_schema(url, boundary.schema_name)
    (descriptor.parent / facts.dbos_path).unlink(missing_ok=True)
    journal_path = descriptor.parent / facts.journal_path
    journal = json.loads(journal_path.read_text())
    journal["cleanup_complete"] = True
    journal["cleaned_at"] = datetime.now(UTC).isoformat()
    journal_path.write_text(
        json.dumps(journal, indent=2, sort_keys=True) + "\n"
    )
    typer.echo("cleaned")


@STORES.command("verify-cleanup")
def stores_verify_cleanup(
    descriptor: Annotated[Path, typer.Option("--descriptor", exists=True)],
) -> None:
    """Verify journaled resources are absent after cleanup."""
    facts = StoreDescriptor.model_validate_json(descriptor.read_text())
    journal = _validate_descriptor_journal(descriptor, facts)
    if journal.get("cleanup_complete") is not True:
        raise ValueError("cleanup journal is not complete")
    for boundary in (facts.source, facts.motherduck, facts.neon):
        if _schema_exists(
            _require_environment(boundary.environment), boundary.schema_name
        ):
            raise ValueError(
                f"run-owned schema remains: {boundary.schema_name}"
            )
    if (descriptor.parent / facts.dbos_path).exists():
        raise ValueError("run-owned DBOS store remains")
    typer.echo("verified")


def main() -> None:
    APP()


if __name__ == "__main__":
    main()
