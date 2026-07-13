from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy.engine import URL
from typer.testing import CliRunner

from whetstone.platform import connections, cutover_tooling
from whetstone.platform.cutover_tooling import (
    APP,
    DatabaseEnvironment,
    _bound_url,
    _create_dbos_marker,
    _create_schema,
    _initialize_dbos_store,
    _new_store_journal,
    _require_dbos_owner,
    _store_descriptor,
)


def test_cutover_cli_has_no_estimates_commands() -> None:
    runner = CliRunner()

    help_result = runner.invoke(APP, ["--help"])
    rejected = runner.invoke(APP, ["estimates", "generate"])

    assert help_result.exit_code == 0
    assert "estimates" not in help_result.output
    assert rejected.exit_code != 0


@pytest.mark.parametrize(
    "scheme",
    ["postgresql", "postgresql+psycopg"],
)
def test_bound_url_normalizes_driver_and_preserves_encoded_credentials_query(
    scheme: str, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "p%2Fss%40word"

    bound = _bound_url(
        f"{scheme}://operator:{secret}@db.example/test"
        "?sslmode=require&application_name=cutover",
        "run_schema",
    )

    assert bound.startswith("postgresql+psycopg://")
    assert bound.count("+psycopg") == 1
    assert f"operator:{secret}@" in bound
    assert "sslmode=require" in bound
    assert "application_name=cutover" in bound
    assert "options=-c+search_path%3Drun_schema%2Cpublic" in bound
    assert "***" not in bound
    assert secret not in caplog.text


@pytest.mark.parametrize(
    ("value", "environment", "expected", "expected_options"),
    [
        (
            "postgresql://operator:p%2Fss@db.example/test?sslmode=require",
            "DATABASE_URL",
            "postgresql+psycopg://operator:p%2Fss@db.example/test"
            "?sslmode=require",
            {},
        ),
        (
            "postgresql+psycopg://operator:p%2Fss@db.example/test"
            "?sslmode=require",
            "NEON_DATABASE_URL",
            "postgresql+psycopg://operator:p%2Fss@db.example/test"
            "?sslmode=require",
            {},
        ),
        (
            "postgresql://operator:p%2Fss@db.example/test?sslmode=require",
            "MOTHERDUCK_DATABASE_URL",
            "postgresql+psycopg://operator:p%2Fss@db.example/test"
            "?sslmode=require",
            {"use_native_hstore": False},
        ),
        (
            "duckdb:///md:warehouse?motherduck_token=token%2Fvalue",
            "MOTHERDUCK_DATABASE_URL",
            "duckdb:///md:warehouse?motherduck_token=token%2Fvalue",
            {},
        ),
    ],
)
def test_sqlalchemy_engine_normalizes_only_postgres_urls(
    value: str,
    environment: DatabaseEnvironment,
    expected: str,
    expected_options: dict[str, bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[URL, dict[str, object]]] = []
    marker = object()
    monkeypatch.setattr(
        connections,
        "create_engine",
        lambda url, **options: captured.append((url, options)) or marker,
    )

    assert (
        cutover_tooling._sqlalchemy_engine(
            value, environment=environment
        )
        is marker
    )
    rendered = [
        (url.render_as_string(hide_password=False), options)
        for url, options in captured
    ]
    assert rendered == [(expected, expected_options)]


def test_sqlalchemy_engine_selects_installed_psycopg_driver() -> None:
    engine = cutover_tooling._sqlalchemy_engine(
        "postgresql://operator:placeholder@db.example/test?sslmode=require",
        environment="DATABASE_URL",
    )
    try:
        assert engine.url.drivername == "postgresql+psycopg"
        assert engine.dialect.driver == "psycopg"
    finally:
        engine.dispose()


def test_bound_motherduck_engine_disables_only_native_hstore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[URL, dict[str, object]]] = []
    marker = object()
    monkeypatch.setattr(
        connections,
        "create_engine",
        lambda url, **options: captured.append((url, options)) or marker,
    )
    bound = _bound_url(
        "postgresql://operator:p%2Fss@db.example/test?sslmode=require",
        "analysis_schema",
    )

    assert (
        cutover_tooling._sqlalchemy_engine(
            bound, environment="MOTHERDUCK_DATABASE_URL"
        )
        is marker
    )
    assert [
        (url.render_as_string(hide_password=False), options)
        for url, options in captured
    ] == [(bound, {"use_native_hstore": False})]
    assert "operator:p%2Fss@" in bound
    assert "sslmode=require" in bound


def test_bound_url_preserves_non_postgres_motherduck_semantics() -> None:
    bound = _bound_url(
        "duckdb:///md:warehouse?motherduck_token=token%2Fvalue",
        "analysis_schema",
    )

    assert bound.startswith("duckdb:///md:warehouse?")
    assert "motherduck_token=token%2Fvalue" in bound
    assert "+psycopg" not in bound


@pytest.mark.parametrize(
    ("environment", "expect_trigger"),
    [
        ("DATABASE_URL", True),
        ("MOTHERDUCK_DATABASE_URL", False),
        ("NEON_DATABASE_URL", True),
    ],
)
def test_schema_marker_uses_backend_specific_immutability(
    environment: DatabaseEnvironment,
    expect_trigger: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = MagicMock()
    engine = MagicMock()
    engine.begin.return_value.__enter__.return_value = connection
    monkeypatch.setattr(
        cutover_tooling, "_sqlalchemy_engine", lambda *_args, **_kwargs: engine
    )
    digest = "a" * 64

    _create_schema(
        "postgresql://operator:encoded%2Fsecret@db.example/test",
        "owned_schema",
        environment=environment,
        run_id="acceptance_171",
        descriptor_sha256=digest,
    )

    statements = "\n".join(
        str(call.args[0]) for call in connection.execute.call_args_list
    )
    assert "CHECK (marker_id = 1)" in statements
    assert "CHECK (run_id = 'acceptance_171')" in statements
    assert f"CHECK (descriptor_sha256 = '{digest}')" in statements
    assert ("LANGUAGE plpgsql" in statements) is expect_trigger
    assert ("CREATE TRIGGER" in statements) is expect_trigger
    assert "encoded%2Fsecret" not in statements


def test_invalid_marker_identity_is_rejected_before_engine_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = MagicMock()
    monkeypatch.setattr(cutover_tooling, "_sqlalchemy_engine", engine)

    with pytest.raises(ValueError, match="invalid marker run ID"):
        _create_schema(
            "postgresql://db.example/test",
            "owned_schema",
            environment="MOTHERDUCK_DATABASE_URL",
            run_id="unsafe'run",
            descriptor_sha256="a" * 64,
        )

    engine.assert_not_called()


def test_store_prepare_defaults_to_zero_mutation(tmp_path: Path) -> None:
    descriptor = tmp_path / "stores.json"
    result = CliRunner().invoke(
        APP,
        [
            "stores",
            "prepare",
            "--run-id",
            "acceptance_171",
            "--descriptor",
            str(descriptor),
        ],
    )

    assert result.exit_code == 0
    assert not descriptor.exists()
    assert "whetstone_run_acceptance_171" in result.stdout
    assert "postgresql" not in result.stdout


def test_store_prepare_execute_requires_exact_confirmation(
    tmp_path: Path,
) -> None:
    result = CliRunner().invoke(
        APP,
        [
            "stores",
            "prepare",
            "--run-id",
            "acceptance_171",
            "--descriptor",
            str(tmp_path / "stores.json"),
            "--execute",
            "--confirm",
            "wrong",
        ],
    )

    assert result.exit_code != 0
    assert "equal to run ID" in result.output


def test_prepare_failure_after_partial_marker_uses_journal_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor_path = tmp_path / "stores.json"
    created: list[DatabaseEnvironment] = []
    cleaned: list[tuple[str, str, Path]] = []
    monkeypatch.setattr(
        cutover_tooling, "_require_environment", lambda name: f"url:{name}"
    )
    monkeypatch.setattr(
        cutover_tooling,
        "_schema_exists",
        lambda *_args, **_kwargs: False,
    )

    def fail_after_motherduck_marker(
        _url: str,
        _schema: str,
        *,
        environment: DatabaseEnvironment,
        **_kwargs: object,
    ) -> None:
        created.append(environment)
        if environment == "MOTHERDUCK_DATABASE_URL":
            raise RuntimeError("failure after durable marker creation")

    monkeypatch.setattr(
        cutover_tooling, "_create_schema", fail_after_motherduck_marker
    )
    monkeypatch.setattr(
        cutover_tooling,
        "_cleanup_owned_resources",
        lambda descriptor, digest, base_dir: cleaned.append(
            (descriptor.run_id, digest, base_dir)
        ),
    )

    with pytest.raises(RuntimeError, match="durable marker"):
        cutover_tooling.prepare_stores(descriptor_path, "acceptance_171")

    assert created == ["DATABASE_URL", "MOTHERDUCK_DATABASE_URL"]
    assert len(cleaned) == 1
    assert cleaned[0][0] == "acceptance_171"
    assert len(cleaned[0][1]) == 64
    assert cleaned[0][2] == tmp_path
    assert (tmp_path / "stores.json.journal.json").is_file()


def test_dbos_store_is_initialized_not_just_touched(tmp_path: Path) -> None:
    path = tmp_path / "dbos.sqlite3"

    _initialize_dbos_store(path, "acceptance_171")

    assert path.stat().st_size > 0


def test_journal_recovers_complete_descriptor_when_descriptor_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor_path = tmp_path / "stores.json"
    descriptor = _store_descriptor("acceptance_171", descriptor_path)
    journal = _new_store_journal(descriptor)
    journal_path = tmp_path / descriptor.journal_path
    journal_path.write_text(journal.model_dump_json())
    monkeypatch.setattr(
        cutover_tooling, "_require_environment", lambda _name: "database-url"
    )
    monkeypatch.setattr(
        cutover_tooling,
        "_schema_exists",
        lambda *_args, **_kwargs: False,
    )

    result = CliRunner().invoke(
        APP,
        [
            "stores",
            "cleanup",
            "--journal",
            str(journal_path),
            "--execute",
            "--confirm",
            descriptor.run_id,
        ],
    )

    assert result.exit_code == 0
    verify = CliRunner().invoke(
        APP,
        ["stores", "verify-cleanup", "--journal", str(journal_path)],
    )
    assert verify.exit_code == 0
    assert not descriptor_path.exists()


def test_dbos_ownership_marker_is_persistent_and_immutable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dbos.sqlite3"
    digest = "a" * 64
    _create_dbos_marker(
        path, run_id="acceptance_171", descriptor_sha256=digest
    )

    _require_dbos_owner(
        path, run_id="acceptance_171", descriptor_sha256=digest
    )
    with (
        sqlite3.connect(path) as connection,
        pytest.raises(sqlite3.IntegrityError, match="immutable"),
    ):
        connection.execute(
            "UPDATE whetstone_cutover_ownership SET run_id='replacement'"
        )


def test_dbos_replacement_without_marker_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dbos.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE replacement (value TEXT)")

    with pytest.raises(ValueError, match="marker is unreadable"):
        _require_dbos_owner(
            path,
            run_id="acceptance_171",
            descriptor_sha256="a" * 64,
        )


def test_cleanup_preflights_all_markers_before_any_drop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = _store_descriptor("acceptance_171", tmp_path / "stores.json")
    dropped: list[str] = []
    monkeypatch.setattr(
        cutover_tooling, "_require_environment", lambda _name: "database-url"
    )
    monkeypatch.setattr(
        cutover_tooling,
        "_schema_exists",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        cutover_tooling,
        "_schema_owner",
        lambda *_args, **_kwargs: ("replacement", "b" * 64),
    )
    monkeypatch.setattr(
        cutover_tooling,
        "_drop_schema",
        lambda _url, schema, **_kwargs: dropped.append(schema),
    )

    with pytest.raises(ValueError, match="ownership marker disagrees"):
        cutover_tooling._cleanup_owned_resources(
            descriptor, "a" * 64, tmp_path
        )

    assert dropped == []


def test_store_binding_rejects_replaced_schema_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor_path = tmp_path / "stores.json"
    descriptor = _store_descriptor("acceptance_171", descriptor_path)
    journal = _new_store_journal(descriptor)
    descriptor_path.write_text(descriptor.model_dump_json())
    (tmp_path / descriptor.journal_path).write_text(journal.model_dump_json())
    monkeypatch.setattr(
        cutover_tooling, "_require_environment", lambda _name: "database-url"
    )
    monkeypatch.setattr(
        cutover_tooling,
        "_schema_exists",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        cutover_tooling,
        "_schema_owner",
        lambda *_args, **_kwargs: ("replacement", "b" * 64),
    )

    with pytest.raises(ValueError, match="ownership marker disagrees"):
        cutover_tooling.validate_store_state(descriptor_path)


def test_stores_run_uses_explicit_schema_strategy_for_motherduck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MotherDuck rejects startup search_path; source/Neon stay bound."""
    descriptor_path = tmp_path / "stores.json"
    descriptor = _store_descriptor("acceptance_171", descriptor_path)
    journal = _new_store_journal(descriptor)
    descriptor_path.write_text(descriptor.model_dump_json())
    (tmp_path / descriptor.journal_path).write_text(
        journal.model_dump_json()
    )
    monkeypatch.setattr(
        cutover_tooling, "validate_store_state", lambda _path: descriptor
    )
    monkeypatch.setattr(
        cutover_tooling,
        "_require_environment",
        lambda name: f"postgresql://operator:pw@db.example/{name.lower()}",
    )
    monkeypatch.setattr(
        cutover_tooling,
        "_require_schema_owner",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        cutover_tooling,
        "_require_dbos_owner",
        lambda *_args, **_kwargs: None,
    )
    captured: dict[str, str] = {}

    def record_run(command: list[str], *, env: dict[str, str], check: bool):
        captured.update(env)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cutover_tooling.subprocess, "run", record_run)

    result = CliRunner().invoke(
        APP,
        [
            "stores",
            "run",
            "--descriptor",
            str(descriptor_path),
            "--",
            "echo",
            "ok",
        ],
    )

    assert result.exit_code == 0
    assert captured["MOTHERDUCK_DATABASE_URL"] == (
        "postgresql://operator:pw@db.example/motherduck_database_url"
    )
    assert "options" not in captured["MOTHERDUCK_DATABASE_URL"]
    assert captured["WHETSTONE_ANALYSIS_SCHEMA"] == (
        descriptor.motherduck.schema_name
    )
    assert (
        "search_path%3Dwhetstone_run_acceptance_171%2Cpublic"
        in captured["DATABASE_URL"]
    )
    assert (
        "search_path%3Dwhetstone_detail_acceptance_171%2Cpublic"
        in captured["NEON_DATABASE_URL"]
    )
