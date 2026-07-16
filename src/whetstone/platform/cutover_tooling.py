"""Fail-closed operator tooling for isolated run stores."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import typer
from dbos import DBOS
from pydantic import BaseModel, ConfigDict, Field, StrictStr
from sqlalchemy import text
from sqlalchemy.engine import Engine

from whetstone.platform.connections import (
    DatabaseBoundary,
    bind_schema,
    create_whetstone_engine,
    render_connection_url,
)
from whetstone.platform.platform_db import (
    assert_run_schema_owns_objects,
    ensure_whetstone_application_schema,
)
from whetstone.platform.runtime import shutdown_dbos_runtime

_RUN_ID = re.compile(r"^[a-z][a-z0-9_]{2,23}$")
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OWNERSHIP_TABLE = "whetstone_cutover_ownership"
Sha256 = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
DatabaseEnvironment = Literal[
    "DATABASE_URL",
    "MOTHERDUCK_DATABASE_URL",
    "NEON_DATABASE_URL",
]

APP = typer.Typer(no_args_is_help=True)
STORES = typer.Typer(no_args_is_help=True)
APP.add_typer(STORES, name="stores")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


class StoreBoundary(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True
    )

    environment: DatabaseEnvironment
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


class StoreJournal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    run_id: StrictStr
    descriptor_sha256: Sha256
    descriptor: StoreDescriptor
    cleanup_complete: bool = False
    cleaned_at: StrictStr | None = None


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
    return render_connection_url(bind_schema(value, schema))


def _require_environment(name: DatabaseEnvironment) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _sqlalchemy_engine(
    url: str, *, environment: DatabaseEnvironment
) -> Engine:
    boundary = (
        DatabaseBoundary.MOTHERDUCK_POSTGRES
        if environment == "MOTHERDUCK_DATABASE_URL"
        else (
            DatabaseBoundary.NEON_POSTGRES
            if environment == "NEON_DATABASE_URL"
            else DatabaseBoundary.SOURCE_SCHEMA
        )
    )
    return create_whetstone_engine(url, boundary=boundary)


def _schema_exists(
    url: str, schema: str, *, environment: DatabaseEnvironment
) -> bool:
    engine = _sqlalchemy_engine(url, environment=environment)
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


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _descriptor_sha256(descriptor: StoreDescriptor) -> str:
    return _sha256(descriptor.model_dump(mode="json"))


def _new_store_journal(descriptor: StoreDescriptor) -> StoreJournal:
    return StoreJournal(
        schema_version=1,
        run_id=descriptor.run_id,
        descriptor_sha256=_descriptor_sha256(descriptor),
        descriptor=descriptor,
    )


def _create_schema(
    url: str,
    schema: str,
    *,
    environment: DatabaseEnvironment,
    run_id: str,
    descriptor_sha256: str,
) -> None:
    if _IDENTIFIER.fullmatch(schema) is None:
        raise ValueError("invalid generated schema")
    if _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("invalid marker run ID")
    if _SHA256.fullmatch(descriptor_sha256) is None:
        raise ValueError("invalid marker descriptor digest")
    engine = _sqlalchemy_engine(url, environment=environment)
    try:
        with engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            connection.execute(
                text(
                    f'CREATE TABLE "{schema}"."{_OWNERSHIP_TABLE}" ('
                    "marker_id SMALLINT PRIMARY KEY "
                    "CONSTRAINT ownership_singleton CHECK (marker_id = 1), "
                    "run_id TEXT NOT NULL "
                    "CONSTRAINT ownership_run_id_locked "
                    f"CHECK (run_id = '{run_id}'), "
                    "descriptor_sha256 TEXT NOT NULL "
                    "CONSTRAINT ownership_digest_locked "
                    "CHECK (descriptor_sha256 = "
                    f"'{descriptor_sha256}'))"
                )
            )
            connection.execute(
                text(
                    f'INSERT INTO "{schema}"."{_OWNERSHIP_TABLE}" '
                    "(marker_id, run_id, descriptor_sha256) "
                    "VALUES (1, :run_id, :digest)"
                ),
                {"run_id": run_id, "digest": descriptor_sha256},
            )
            if environment != "MOTHERDUCK_DATABASE_URL":
                connection.execute(
                    text(
                        f'CREATE FUNCTION "{schema}".'
                        '"reject_ownership_marker_mutation"() '
                        "RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
                        "RAISE EXCEPTION 'ownership marker is immutable'; "
                        "END $$"
                    )
                )
                connection.execute(
                    text(
                        f'CREATE TRIGGER "ownership_marker_is_immutable" '
                        f'BEFORE UPDATE OR DELETE ON "{schema}".'
                        f'"{_OWNERSHIP_TABLE}" FOR EACH ROW EXECUTE FUNCTION '
                        f'"{schema}"."reject_ownership_marker_mutation"()'
                    )
                )
    finally:
        engine.dispose()


def _schema_owner(
    url: str, schema: str, *, environment: DatabaseEnvironment
) -> tuple[str, str] | None:
    engine = _sqlalchemy_engine(url, environment=environment)
    try:
        with engine.connect() as connection:
            exists = connection.execute(
                text(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema=:schema AND table_name=:table)"
                ),
                {"schema": schema, "table": _OWNERSHIP_TABLE},
            ).scalar_one()
            if not exists:
                return None
            row = connection.execute(
                text(
                    f'SELECT run_id, descriptor_sha256 FROM "{schema}".'
                    f'"{_OWNERSHIP_TABLE}"'
                )
            ).one_or_none()
            if row is None:
                return None
            return str(row[0]), str(row[1])
    finally:
        engine.dispose()


def _require_schema_owner(
    url: str,
    schema: str,
    *,
    environment: DatabaseEnvironment,
    run_id: str,
    descriptor_sha256: str,
) -> None:
    if _schema_owner(url, schema, environment=environment) != (
        run_id,
        descriptor_sha256,
    ):
        raise ValueError(f"schema ownership marker disagrees: {schema}")


def _drop_schema(
    url: str, schema: str, *, environment: DatabaseEnvironment
) -> None:
    if _IDENTIFIER.fullmatch(schema) is None:
        raise ValueError("invalid generated schema")
    engine = _sqlalchemy_engine(url, environment=environment)
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


def _create_dbos_marker(
    path: Path, *, run_id: str, descriptor_sha256: str
) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            f"CREATE TABLE {_OWNERSHIP_TABLE} ("
            "run_id TEXT PRIMARY KEY, descriptor_sha256 TEXT NOT NULL)"
        )
        connection.execute(
            f"INSERT INTO {_OWNERSHIP_TABLE} "
            "(run_id, descriptor_sha256) VALUES (?, ?)",
            (run_id, descriptor_sha256),
        )
        for operation in ("UPDATE", "DELETE"):
            connection.execute(
                f"CREATE TRIGGER ownership_marker_no_{operation.lower()} "
                f"BEFORE {operation} ON {_OWNERSHIP_TABLE} BEGIN "
                "SELECT RAISE(ABORT, 'ownership marker is immutable'); END"
            )
        connection.commit()
    finally:
        connection.close()


def _require_dbos_owner(
    path: Path, *, run_id: str, descriptor_sha256: str
) -> None:
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            row = connection.execute(
                f"SELECT run_id, descriptor_sha256 FROM {_OWNERSHIP_TABLE}"
            ).fetchone()
        finally:
            connection.close()
    except sqlite3.Error as error:
        raise ValueError("DBOS ownership marker is unreadable") from error
    if row != (run_id, descriptor_sha256):
        raise ValueError("DBOS ownership marker disagrees")


def _load_store_recovery(
    descriptor_path: Path | None,
    journal_path: Path | None = None,
) -> tuple[StoreDescriptor, StoreJournal, Path]:
    """Load complete recovery facts even if prepare crashed pre-descriptor."""
    if descriptor_path is None and journal_path is None:
        raise ValueError("--descriptor or --journal is required")
    descriptor: StoreDescriptor | None = None
    if descriptor_path is not None and descriptor_path.exists():
        descriptor = StoreDescriptor.model_validate_json(
            descriptor_path.read_text()
        )
    if journal_path is None:
        if descriptor is not None:
            assert descriptor_path is not None
            journal_path = descriptor_path.parent / descriptor.journal_path
        else:
            assert descriptor_path is not None
            journal_path = descriptor_path.with_name(
                descriptor_path.name + ".journal.json"
            )
    journal = StoreJournal.model_validate_json(journal_path.read_text())
    recovered = journal.descriptor
    if journal.run_id != recovered.run_id:
        raise ValueError("journal run identity does not match descriptor")
    if journal.descriptor_sha256 != _descriptor_sha256(recovered):
        raise ValueError("embedded descriptor and journal disagree")
    if descriptor is not None and descriptor != recovered:
        raise ValueError("descriptor and journal disagree")
    return recovered, journal, journal_path


def _owned_resources_preflight(
    descriptor: StoreDescriptor,
    descriptor_sha256: str,
    base_dir: Path,
) -> None:
    for boundary in (
        descriptor.source,
        descriptor.motherduck,
        descriptor.neon,
    ):
        url = _require_environment(boundary.environment)
        if _schema_exists(
            url,
            boundary.schema_name,
            environment=boundary.environment,
        ):
            _require_schema_owner(
                url,
                boundary.schema_name,
                environment=boundary.environment,
                run_id=descriptor.run_id,
                descriptor_sha256=descriptor_sha256,
            )
    dbos_path = base_dir / descriptor.dbos_path
    if dbos_path.exists():
        _require_dbos_owner(
            dbos_path,
            run_id=descriptor.run_id,
            descriptor_sha256=descriptor_sha256,
        )


def _cleanup_owned_resources(
    descriptor: StoreDescriptor,
    descriptor_sha256: str,
    base_dir: Path,
) -> None:
    """Remove only resources whose persistent marker matches this run."""
    _owned_resources_preflight(descriptor, descriptor_sha256, base_dir)
    for boundary in (
        descriptor.neon,
        descriptor.motherduck,
        descriptor.source,
    ):
        url = _require_environment(boundary.environment)
        if _schema_exists(
            url,
            boundary.schema_name,
            environment=boundary.environment,
        ):
            _require_schema_owner(
                url,
                boundary.schema_name,
                environment=boundary.environment,
                run_id=descriptor.run_id,
                descriptor_sha256=descriptor_sha256,
            )
            _drop_schema(
                url,
                boundary.schema_name,
                environment=boundary.environment,
            )
    dbos_path = base_dir / descriptor.dbos_path
    if dbos_path.exists():
        _require_dbos_owner(
            dbos_path,
            run_id=descriptor.run_id,
            descriptor_sha256=descriptor_sha256,
        )
        dbos_path.unlink()


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
        if _schema_exists(
            url,
            boundary.schema_name,
            environment=boundary.environment,
        ):
            raise ValueError(
                f"refusing schema collision: {boundary.schema_name}"
            )
    dbos_path = descriptor_path.parent / descriptor.dbos_path
    if dbos_path.exists():
        raise ValueError("refusing DBOS path collision")
    journal_path = descriptor_path.parent / descriptor.journal_path
    journal = _new_store_journal(descriptor)
    _write_json_atomic(journal_path, journal.model_dump(mode="json"))
    try:
        for boundary in boundaries:
            _create_schema(
                _require_environment(boundary.environment),
                boundary.schema_name,
                environment=boundary.environment,
                run_id=run_id,
                descriptor_sha256=journal.descriptor_sha256,
            )
        ensure_whetstone_application_schema(
            _require_environment(descriptor.source.environment),
            expected_schema=descriptor.source.schema_name,
        )
        _create_dbos_marker(
            dbos_path,
            run_id=run_id,
            descriptor_sha256=journal.descriptor_sha256,
        )
        _initialize_dbos_store(dbos_path, run_id)
        _write_json_atomic(descriptor_path, descriptor.model_dump(mode="json"))
        return descriptor
    except Exception:
        try:
            _cleanup_owned_resources(
                descriptor, journal.descriptor_sha256, descriptor_path.parent
            )
        except Exception:
            pass
        raise


def validate_store_state(descriptor_path: Path) -> StoreDescriptor:
    descriptor, journal, _journal_path = _load_store_recovery(descriptor_path)
    if journal.cleanup_complete:
        raise ValueError("store journal records completed cleanup")
    for boundary in (
        descriptor.source,
        descriptor.motherduck,
        descriptor.neon,
    ):
        url = _require_environment(boundary.environment)
        if not _schema_exists(
            url,
            boundary.schema_name,
            environment=boundary.environment,
        ):
            raise ValueError(
                f"missing run-owned schema: {boundary.schema_name}"
            )
        _require_schema_owner(
            url,
            boundary.schema_name,
            environment=boundary.environment,
            run_id=descriptor.run_id,
            descriptor_sha256=journal.descriptor_sha256,
        )
    _verify_source_run_schema_objects(descriptor)
    dbos_path = descriptor_path.parent / descriptor.dbos_path
    if not dbos_path.is_file() or dbos_path.stat().st_size == 0:
        raise ValueError("missing run-owned DBOS store")
    _require_dbos_owner(
        dbos_path,
        run_id=descriptor.run_id,
        descriptor_sha256=journal.descriptor_sha256,
    )
    return descriptor


def _verify_source_run_schema_objects(descriptor: StoreDescriptor) -> None:
    """Prove migrated source objects resolve from the run schema."""
    bound = bind_schema(
        _require_environment(descriptor.source.environment),
        descriptor.source.schema_name,
    )
    engine = create_whetstone_engine(
        bound, boundary=DatabaseBoundary.SOURCE_SCHEMA
    )
    try:
        with engine.connect() as connection:
            assert_run_schema_owns_objects(
                connection, descriptor.source.schema_name
            )
    finally:
        engine.dispose()


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
    _recovered, journal, _journal_path = _load_store_recovery(descriptor)
    if not command:
        raise typer.BadParameter("command is required")
    environment = os.environ.copy()
    for name, boundary in (
        ("DATABASE_URL", facts.source),
        ("MOTHERDUCK_DATABASE_URL", facts.motherduck),
        ("NEON_DATABASE_URL", facts.neon),
    ):
        url = _require_environment(boundary.environment)
        _require_schema_owner(
            url,
            boundary.schema_name,
            environment=boundary.environment,
            run_id=facts.run_id,
            descriptor_sha256=journal.descriptor_sha256,
        )
        if name == "MOTHERDUCK_DATABASE_URL":
            # MotherDuck's Postgres-wire endpoint rejects the startup
            # search_path option outright, so the analysis boundary uses
            # the explicit-schema strategy: consumers receive the base URL
            # and must qualify or SET search_path to this schema.
            environment[name] = url
            environment["WHETSTONE_ANALYSIS_SCHEMA"] = boundary.schema_name
        else:
            environment[name] = _bound_url(url, boundary.schema_name)
    dbos_path = descriptor.parent / facts.dbos_path
    _require_dbos_owner(
        dbos_path,
        run_id=facts.run_id,
        descriptor_sha256=journal.descriptor_sha256,
    )
    environment["DBOS_SYSTEM_DATABASE_URL"] = "sqlite:///" + str(
        dbos_path.resolve()
    )
    raise typer.Exit(
        subprocess.run(command, env=environment, check=False).returncode
    )


@STORES.command("cleanup")
def stores_cleanup(
    descriptor_path: Annotated[
        Path | None, typer.Option("--descriptor")
    ] = None,
    journal_path: Annotated[Path | None, typer.Option("--journal")] = None,
    execute: Annotated[bool, typer.Option("--execute")] = False,
    confirm: Annotated[str | None, typer.Option("--confirm")] = None,
) -> None:
    """Plan or remove marker-owned boundaries, including crash recovery."""
    facts, journal, recovered_journal_path = _load_store_recovery(
        descriptor_path, journal_path
    )
    if not execute:
        typer.echo(
            json.dumps(
                {
                    "dry_run": True,
                    "run_id": facts.run_id,
                    "journal": str(recovered_journal_path),
                }
            )
        )
        return
    if confirm != facts.run_id:
        raise typer.BadParameter(
            "--execute requires --confirm equal to run ID"
        )
    base_dir = (
        descriptor_path.parent
        if descriptor_path is not None and descriptor_path.exists()
        else recovered_journal_path.parent
    )
    _cleanup_owned_resources(facts, journal.descriptor_sha256, base_dir)
    completed = journal.model_copy(
        update={
            "cleanup_complete": True,
            "cleaned_at": datetime.now(UTC).isoformat(),
        }
    )
    _write_json_atomic(
        recovered_journal_path, completed.model_dump(mode="json")
    )
    typer.echo("cleaned")


@STORES.command("verify-cleanup")
def stores_verify_cleanup(
    descriptor_path: Annotated[
        Path | None, typer.Option("--descriptor")
    ] = None,
    journal_path: Annotated[Path | None, typer.Option("--journal")] = None,
) -> None:
    """Verify journaled resources are absent after cleanup."""
    facts, journal, recovered_journal_path = _load_store_recovery(
        descriptor_path, journal_path
    )
    if journal.cleanup_complete is not True:
        raise ValueError("cleanup journal is not complete")
    for boundary in (facts.source, facts.motherduck, facts.neon):
        if _schema_exists(
            _require_environment(boundary.environment),
            boundary.schema_name,
            environment=boundary.environment,
        ):
            raise ValueError(
                f"run-owned schema remains: {boundary.schema_name}"
            )
    base_dir = (
        descriptor_path.parent
        if descriptor_path is not None and descriptor_path.exists()
        else recovered_journal_path.parent
    )
    if (base_dir / facts.dbos_path).exists():
        raise ValueError("run-owned DBOS store remains")
    typer.echo("verified")


def main() -> None:
    APP()


if __name__ == "__main__":
    main()
