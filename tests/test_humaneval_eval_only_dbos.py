from __future__ import annotations

from typing import Any

import pytest


class FakeCursor:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, statement: str, params: object = None) -> None:
        self.statements.append(statement)


class FakeConnection:
    def __init__(self) -> None:
        self.cursor_instance = FakeCursor()

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return self.cursor_instance


def test_build_eval_dbos_config_uses_database_url_for_dbos_default(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DBOS_SYSTEM_DATABASE_URL", raising=False)

    config = eval_dbos_harness.build_eval_dbos_config(
        database_url="postgresql:///app",
        dbos_system_database_url=None,
        generation_concurrency=7,
        scoring_concurrency=3,
    )

    assert config.database_url == "postgresql:///app"
    assert config.dbos_system_database_url == "postgresql:///app"
    assert config.generation_concurrency == 7
    assert config.scoring_concurrency == 3
    assert eval_dbos_harness.build_dbos_config(config) == {
        "name": eval_dbos_harness.DBOS_APP_NAME,
        "system_database_url": "postgresql:///app",
    }


def test_build_eval_dbos_config_requires_database_url(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ValueError, match="DATABASE_URL"):
        eval_dbos_harness.build_eval_dbos_config(
            database_url=None,
            dbos_system_database_url=None,
            generation_concurrency=7,
            scoring_concurrency=3,
        )


def test_create_eval_schema_executes_expected_statements(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    fake_conn = FakeConnection()

    def fake_connect(database_url: str) -> FakeConnection:
        assert database_url == "postgresql:///unit"
        return fake_conn

    monkeypatch.setattr(eval_dbos_harness.psycopg, "connect", fake_connect)

    eval_dbos_harness.create_eval_schema("postgresql:///unit")

    statements = fake_conn.cursor_instance.statements
    assert statements[0] == eval_dbos_harness.EXPERIMENTS_TABLE_SQL
    assert statements[1] == eval_dbos_harness.PREDICTIONS_TABLE_SQL
    assert statements[2:] == list(eval_dbos_harness.PREDICTION_INDEX_SQL)


def test_register_and_listen_to_eval_queues(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    registered: list[dict[str, Any]] = []
    listened: list[list[str]] = []

    def fake_register_queue(name: str, **kwargs: Any) -> None:
        registered.append({"name": name, **kwargs})

    def fake_listen_queues(queues: list[str]) -> None:
        listened.append(queues)

    monkeypatch.setattr(
        eval_dbos_harness.DBOS, "register_queue", fake_register_queue
    )
    monkeypatch.setattr(
        eval_dbos_harness.DBOS, "listen_queues", fake_listen_queues
    )

    config = eval_dbos_harness.EvalDbosConfig(
        database_url="postgresql:///app",
        dbos_system_database_url="postgresql:///dbos",
        generation_concurrency=11,
        scoring_concurrency=5,
    )
    eval_dbos_harness.register_eval_queues(config)
    eval_dbos_harness.listen_to_selected_queue(
        eval_dbos_harness.QueueSelection.BOTH
    )

    assert registered == [
        {
            "name": eval_dbos_harness.GENERATION_QUEUE_NAME,
            "worker_concurrency": 11,
        },
        {
            "name": eval_dbos_harness.SCORING_QUEUE_NAME,
            "worker_concurrency": 5,
        },
    ]
    assert listened == [
        [
            eval_dbos_harness.GENERATION_QUEUE_NAME,
            eval_dbos_harness.SCORING_QUEUE_NAME,
        ]
    ]
