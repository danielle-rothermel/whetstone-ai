"""App-side EvalDbosConfig wiring (URL resolution tests live in
dr-platform)."""

from __future__ import annotations

import pytest

from whetstone.platform import dbos_bootstrap


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
    assert (
        explicit.dbos_system_database_url
        == "postgresql+psycopg://system-explicit/db"
    )
    assert explicit.generation_concurrency == 2
    assert explicit.scoring_concurrency == 1

    from_env = dbos_bootstrap.build_eval_dbos_config(
        database_url="postgresql://app/db",
        dbos_system_database_url=None,
        generation_concurrency=2,
        scoring_concurrency=1,
    )
    assert (
        from_env.dbos_system_database_url
        == "postgresql+psycopg://system-env/db"
    )

    monkeypatch.delenv("DBOS_SYSTEM_DATABASE_URL", raising=False)
    from_app = dbos_bootstrap.build_eval_dbos_config(
        database_url="postgresql://app/db",
        dbos_system_database_url=None,
        generation_concurrency=2,
        scoring_concurrency=1,
    )
    assert (
        from_app.dbos_system_database_url == "postgresql+psycopg://app/db"
    )


def test_build_dbos_config_uses_system_database_url() -> None:
    config = dbos_bootstrap.EvalDbosConfig(
        database_url="postgresql+psycopg://app/db",
        dbos_system_database_url="postgresql+psycopg://system/db",
        generation_concurrency=2,
        scoring_concurrency=1,
    )
    assert dbos_bootstrap.build_dbos_config(config, app_name="app") == {
        "name": "app",
        "system_database_url": "postgresql+psycopg://system/db",
    }
