from __future__ import annotations

from typing import Any

import pytest

from dr_dspy.platform import dbos_bootstrap


def test_resolve_database_url_prefers_explicit_arg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://env/db")

    assert (
        dbos_bootstrap.resolve_database_url("postgresql://explicit/db")
        == "postgresql://explicit/db"
    )


def test_resolve_database_url_reads_env_when_arg_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://env/db")

    assert dbos_bootstrap.resolve_database_url(None) == "postgresql://env/db"


def test_resolve_database_url_raises_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(
        ValueError,
        match="--database-url or DATABASE_URL is required",
    ):
        dbos_bootstrap.resolve_database_url(None)

    with pytest.raises(
        ValueError,
        match=(
            "--database-url or DATABASE_URL is required "
            "for platform graph workflow"
        ),
    ):
        dbos_bootstrap.resolve_database_url(
            None,
            error_suffix="for platform graph workflow",
        )


def test_build_eval_dbos_config_system_url_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://app/db")
    monkeypatch.setenv(
        "DBOS_SYSTEM_DATABASE_URL",
        "postgresql://system-env/db",
    )

    explicit = dbos_bootstrap.build_eval_dbos_config(
        database_url="postgresql://app/db",
        dbos_system_database_url="postgresql://system-explicit/db",
        generation_concurrency=2,
        scoring_concurrency=1,
    )
    assert explicit.dbos_system_database_url == "postgresql://system-explicit/db"

    from_env = dbos_bootstrap.build_eval_dbos_config(
        database_url="postgresql://app/db",
        dbos_system_database_url=None,
        generation_concurrency=2,
        scoring_concurrency=1,
    )
    assert from_env.dbos_system_database_url == "postgresql://system-env/db"

    monkeypatch.delenv("DBOS_SYSTEM_DATABASE_URL", raising=False)
    from_app = dbos_bootstrap.build_eval_dbos_config(
        database_url="postgresql://app/db",
        dbos_system_database_url=None,
        generation_concurrency=2,
        scoring_concurrency=1,
    )
    assert from_app.dbos_system_database_url == "postgresql://app/db"


def test_destroy_dbos_runtime_calls_dbos_destroy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []

    monkeypatch.setattr(
        dbos_bootstrap.DBOS,
        "destroy",
        lambda: calls.append("destroy"),
    )

    dbos_bootstrap.destroy_dbos_runtime()

    assert calls == ["destroy"]
