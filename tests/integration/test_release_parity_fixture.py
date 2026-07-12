from __future__ import annotations

# ruff: noqa: E501
import os
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, text

from whetstone.platform.release_parity_fixture import (
    _journal_path,
    cleanup,
    cleanup_local,
    prepare,
    prepare_local,
    verify_evidence,
)


@pytest.mark.integration
def test_release_parity_fixture_prepare_resolve_and_cleanup(
    tmp_path: Path,
) -> None:
    """Exercise the real MotherDuck/Neon boundary only with explicit secrets."""

    required = ("DATABASE_URL", "MOTHERDUCK_DATABASE_URL", "NEON_DATABASE_URL")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.skip("release-parity credentials are not configured")
    descriptor = tmp_path / "descriptor.json"
    proof = tmp_path / "cleanup-proof.json"
    try:
        prepared = prepare(descriptor)
        prepared.validate_contract()
    finally:
        cleanup(descriptor, proof)
    verify_evidence(descriptor, proof)


@pytest.mark.integration
def test_prepare_local_post_schema_failure_recovers_from_journal(
    tmp_path: Path,
    postgres_base_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real source schema failure remains recoverable without a descriptor."""
    descriptor_path = tmp_path / "descriptor.json"
    monkeypatch.setenv("DATABASE_URL", postgres_base_url)
    created_schema: list[str] = []

    def fail_after_source_creation(source: Engine) -> None:
        with source.connect() as connection:
            schema = connection.scalar(
                text("SELECT current_schema()")
            )
        assert isinstance(schema, str)
        created_schema.append(schema)
        raise RuntimeError("injected post-schema failure")

    with pytest.raises(RuntimeError, match="injected post-schema failure"):
        prepare_local(
            descriptor_path,
            _after_source_creation=fail_after_source_creation,
            _retain_failed_resources=True,
        )

    journal_path = _journal_path(descriptor_path)
    assert created_schema
    assert not descriptor_path.exists()
    assert journal_path.exists()

    admin = create_engine(postgres_base_url)
    try:
        with admin.connect() as connection:
            assert connection.scalar(
                text("SELECT count(*) FROM pg_namespace WHERE nspname = :schema"),
                {"schema": created_schema[0]},
            ) == 1
    finally:
        admin.dispose()

    cleanup_local(descriptor_path)

    admin = create_engine(postgres_base_url)
    try:
        with admin.connect() as connection:
            source_schema_rows = connection.scalar(
                text(
                    "SELECT count(*) FROM pg_namespace WHERE nspname = :schema"
                ),
                {"schema": created_schema[0]},
            )
    finally:
        admin.dispose()
    assert source_schema_rows == 0
    assert not list(tmp_path.glob("*.duckdb"))
    assert not list(tmp_path.glob("*.duckdb.lock"))
