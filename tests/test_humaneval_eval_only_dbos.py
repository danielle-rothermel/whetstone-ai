from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from rich.console import Console

from dr_dspy import analysis as shared_analysis
from dspy.utils.dummies import dotdict  # type: ignore[attr-defined]


class FakeCursor:
    def __init__(self) -> None:
        self.statements: list[str] = []
        self.params: Any = None
        self.rowcount = 0

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, statement: str, params: Any = None) -> None:
        self.statements.append(statement)
        if not isinstance(self.params, list):
            self.params = params

    def executemany(self, statement: str, params: Any) -> None:
        self.statements.append(statement)
        self.rowcount = len(list(params))


class FakeConnection:
    def __init__(self) -> None:
        self.cursor_instance = FakeCursor()

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return self.cursor_instance


class FakeCompletions:
    def __init__(
        self,
        *,
        content: str | None = "ok",
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.content = content

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return dotdict(
            model=kwargs["model"],
            choices=[
                dotdict(
                    message=dotdict(role="assistant", content=self.content),
                    finish_reason="stop",
                )
            ],
            usage=dotdict(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
                cost=0.01,
            ),
        )


class FakeClient:
    def __init__(
        self,
        *,
        content: str | None = "ok",
    ) -> None:
        self.chat = dotdict(
            completions=FakeCompletions(content=content)
        )


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
    assert eval_dbos_harness.shared_dbos.build_dbos_config(
        config, app_name=eval_dbos_harness.DBOS_APP_NAME
    ) == {
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
    migration_end = 2 + len(eval_dbos_harness.PREDICTION_MIGRATION_SQL)
    assert statements[2:migration_end] == list(
        eval_dbos_harness.PREDICTION_MIGRATION_SQL
    )
    assert statements[migration_end:] == list(
        eval_dbos_harness.PREDICTION_INDEX_SQL
    )


def test_unconfigured_connect_db_uses_psycopg_connect(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    fake_conn = FakeConnection()
    calls: list[str] = []
    eval_dbos_harness.close_db_connection_pools()

    def fake_connect(database_url: str) -> FakeConnection:
        calls.append(database_url)
        return fake_conn

    monkeypatch.setattr(eval_dbos_harness.psycopg, "connect", fake_connect)

    with eval_dbos_harness.connect_db("postgresql:///unit") as conn:
        assert conn is fake_conn

    assert calls == ["postgresql:///unit"]


def test_db_pool_auto_size_follows_queue_selection(eval_dbos_harness) -> None:
    assert (
        eval_dbos_harness.shared_dbos.auto_db_pool_max_size(
            queue=eval_dbos_harness.QueueSelection.GENERATION,
            generation_concurrency=11,
            scoring_concurrency=5,
        )
        == 19
    )
    assert (
        eval_dbos_harness.shared_dbos.auto_db_pool_max_size(
            queue=eval_dbos_harness.QueueSelection.SCORING,
            generation_concurrency=11,
            scoring_concurrency=5,
        )
        == 13
    )
    assert (
        eval_dbos_harness.shared_dbos.auto_db_pool_max_size(
            queue=eval_dbos_harness.QueueSelection.BOTH,
            generation_concurrency=11,
            scoring_concurrency=5,
        )
        == 24
    )


def test_worker_db_pool_setup_keys_by_distinct_database_url(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    created: list[dict[str, object]] = []
    eval_dbos_harness.close_db_connection_pools()

    class FakePool:
        def __init__(
            self,
            *,
            conninfo: str,
            min_size: int,
            max_size: int,
            open: bool,
        ) -> None:
            self.conninfo = conninfo
            created.append(
                {
                    "conninfo": conninfo,
                    "min_size": min_size,
                    "max_size": max_size,
                    "open": open,
                }
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(eval_dbos_harness, "ConnectionPool", FakePool)
    config = eval_dbos_harness.EvalDbosConfig(
        database_url="postgresql:///same",
        dbos_system_database_url="postgresql:///same",
        generation_concurrency=11,
        scoring_concurrency=5,
    )

    pool_config = eval_dbos_harness.configure_worker_db_connection_pools(
        config,
        queue=eval_dbos_harness.QueueSelection.BOTH,
        raw_max_size=eval_dbos_harness.DB_POOL_AUTO,
    )

    assert pool_config.max_size == 24
    assert created == [
        {
            "conninfo": "postgresql:///same",
            "min_size": 0,
            "max_size": 24,
            "open": True,
        }
    ]
    eval_dbos_harness.close_db_connection_pools()

    config = eval_dbos_harness.EvalDbosConfig(
        database_url="postgresql:///app",
        dbos_system_database_url="postgresql:///dbos",
        generation_concurrency=11,
        scoring_concurrency=5,
    )

    eval_dbos_harness.configure_worker_db_connection_pools(
        config,
        queue=eval_dbos_harness.QueueSelection.SCORING,
        raw_max_size="3",
    )

    assert [item["conninfo"] for item in created[1:]] == [
        "postgresql:///app",
        "postgresql:///dbos",
    ]
    assert [item["max_size"] for item in created[1:]] == [3, 3]
    eval_dbos_harness.close_db_connection_pools()


def test_raise_open_file_limit_raises_soft_limit(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    class FakeResource:
        RLIMIT_NOFILE = 7
        RLIM_INFINITY = -1

        def __init__(self) -> None:
            self.limit = (256, 8192)
            self.set_calls: list[tuple[int, tuple[int, int]]] = []

        def getrlimit(self, limit: int) -> tuple[int, int]:
            assert limit == self.RLIMIT_NOFILE
            return self.limit

        def setrlimit(self, limit: int, value: tuple[int, int]) -> None:
            assert limit == self.RLIMIT_NOFILE
            self.set_calls.append((limit, value))
            self.limit = value

    fake_resource = FakeResource()
    monkeypatch.setattr(eval_dbos_harness, "resource", fake_resource)

    result = eval_dbos_harness.raise_open_file_limit(8192)

    assert result.requested == 8192
    assert result.original_soft == 256
    assert result.original_hard == 8192
    assert result.active_soft == 8192
    assert result.active_hard == 8192
    assert result.changed is True
    assert fake_resource.set_calls == [(7, (8192, 8192))]


def test_raise_open_file_limit_clamps_to_hard_limit(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    class FakeResource:
        RLIMIT_NOFILE = 7
        RLIM_INFINITY = -1

        def __init__(self) -> None:
            self.limit = (256, 1024)

        def getrlimit(self, limit: int) -> tuple[int, int]:
            assert limit == self.RLIMIT_NOFILE
            return self.limit

        def setrlimit(self, limit: int, value: tuple[int, int]) -> None:
            assert limit == self.RLIMIT_NOFILE
            self.limit = value

    fake_resource = FakeResource()
    monkeypatch.setattr(eval_dbos_harness, "resource", fake_resource)

    result = eval_dbos_harness.raise_open_file_limit(8192)

    assert result.requested == 8192
    assert result.active_soft == 1024
    assert result.active_hard == 1024
    assert result.changed is True
    assert eval_dbos_harness.open_file_limit_style(result) == "yellow"
    assert eval_dbos_harness.open_file_limit_line(result) == (
        "Open Files     | requested= 8192 | soft=     1024 | "
        "hard=     1024 | changed=yes"
    )


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
    queue_names = eval_dbos_harness.eval_queue_names("exp")
    eval_dbos_harness.register_eval_queues(config, experiment_name="exp")
    eval_dbos_harness.listen_to_selected_queue(
        eval_dbos_harness.QueueSelection.BOTH, experiment_name="exp"
    )

    assert registered == [
        {
            "name": queue_names.generation,
            "worker_concurrency": 11,
            "on_conflict": "always_update",
        },
        {
            "name": queue_names.scoring,
            "worker_concurrency": 5,
            "on_conflict": "always_update",
        },
    ]
    assert listened == [[queue_names.generation, queue_names.scoring]]


def test_worker_queue_selection_and_log_path(eval_dbos_harness) -> None:
    queue_names = eval_dbos_harness.eval_queue_names("exp")
    assert eval_dbos_harness.queue_names_for_selection(
        eval_dbos_harness.QueueSelection.GENERATION, experiment_name="exp"
    ) == (queue_names.generation,)
    assert eval_dbos_harness.queue_names_for_selection(
        eval_dbos_harness.QueueSelection.SCORING, experiment_name="exp"
    ) == (queue_names.scoring,)
    assert eval_dbos_harness.queue_names_for_selection(
        eval_dbos_harness.QueueSelection.BOTH, experiment_name="exp"
    ) == (queue_names.generation, queue_names.scoring)

    experiment_name = "local mock/dbos smoke"
    path = eval_dbos_harness.shared_eval_logging.default_worker_log_path(
        log_root=eval_dbos_harness.DEFAULT_WORKER_LOG_ROOT,
        experiment_name=experiment_name,
        queue=eval_dbos_harness.QueueSelection.GENERATION,
        hash_length=eval_dbos_harness.EXPERIMENT_QUEUE_HASH_LENGTH,
        now=datetime(2026, 1, 2, 3, 4, 5),
        pid=123,
    )

    assert path == (
        eval_dbos_harness.DEFAULT_WORKER_LOG_ROOT
        / eval_dbos_harness.shared_eval_logging.hashed_experiment_log_name(
            experiment_name,
            hash_length=eval_dbos_harness.EXPERIMENT_QUEUE_HASH_LENGTH,
        )
        / "20260102-030405-generation-pid123.log"
    )


def test_sync_existing_dbos_queue_concurrency_updates_existing_rows(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    class QueueCursor(FakeCursor):
        def __init__(self) -> None:
            super().__init__()
            self.params: list[object] = []

        def execute(self, statement: str, params: object = None) -> None:
            super().execute(statement, params)
            self.params.append(params)
            self.rowcount = 1

    class QueueConnection(FakeConnection):
        def __init__(self) -> None:
            self.cursor_instance = QueueCursor()

    fake_conn = QueueConnection()
    monkeypatch.setattr(
        eval_dbos_harness.psycopg,
        "connect",
        lambda _database_url: fake_conn,
    )
    config = eval_dbos_harness.EvalDbosConfig(
        database_url="postgresql:///app",
        dbos_system_database_url="postgresql:///dbos",
        generation_concurrency=11,
        scoring_concurrency=5,
    )

    rowcount = eval_dbos_harness.sync_existing_dbos_queue_concurrency(
        config, experiment_name="exp"
    )

    queue_names = eval_dbos_harness.eval_queue_names("exp")
    assert rowcount == 2
    assert fake_conn.cursor_instance.params == [
        (11, queue_names.generation),
        (5, queue_names.scoring),
    ]
    assert "UPDATE dbos.queues" in fake_conn.cursor_instance.statements[0]


def test_sync_existing_dbos_queue_concurrency_skips_missing_dbos_table(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    class MissingQueueCursor(FakeCursor):
        def execute(self, statement: str, params: object = None) -> None:
            raise eval_dbos_harness.psycopg.errors.UndefinedTable(
                "missing dbos.queues"
            )

    class MissingQueueConnection(FakeConnection):
        def __init__(self) -> None:
            self.cursor_instance = MissingQueueCursor()

    monkeypatch.setattr(
        eval_dbos_harness.psycopg,
        "connect",
        lambda _database_url: MissingQueueConnection(),
    )
    config = eval_dbos_harness.EvalDbosConfig(
        database_url="postgresql:///app",
        dbos_system_database_url="postgresql:///dbos",
        generation_concurrency=11,
        scoring_concurrency=5,
    )

    rowcount = eval_dbos_harness.sync_existing_dbos_queue_concurrency(
        config, experiment_name="exp"
    )

    assert rowcount == 0


def test_worker_monitor_phase_counts_filter_experiment(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    class FetchCursor:
        def __init__(self) -> None:
            self.statement: str | None = None
            self.params: object = None

        def __enter__(self) -> FetchCursor:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, statement: str, params: object = None) -> None:
            self.statement = statement
            self.params = params

        def fetchall(self) -> list[tuple[str, int]]:
            return [("pending", 3)]

    class FetchConnection:
        def __init__(self) -> None:
            self.cursor_instance = FetchCursor()

        def __enter__(self) -> FetchConnection:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def cursor(self) -> FetchCursor:
            return self.cursor_instance

    fake_conn = FetchConnection()
    monkeypatch.setattr(
        eval_dbos_harness.psycopg,
        "connect",
        lambda _database_url: fake_conn,
    )

    counts = eval_dbos_harness.fetch_prediction_phase_counts(
        "postgresql:///unit",
        status_column="scoring_status",
        experiment_name="exp-1",
    )

    assert counts == {"pending": 3}
    assert fake_conn.cursor_instance.statement is not None
    assert "WHERE experiment_name = %s" in fake_conn.cursor_instance.statement
    assert fake_conn.cursor_instance.params == ("exp-1",)


def test_worker_monitor_line_has_aligned_metrics(
    eval_dbos_harness,
) -> None:
    snapshot = eval_dbos_harness.WorkerQueueSnapshot(
        dbos_status_counts={"ENQUEUED": 3, "SUCCESS": 2},
        generation_status_counts={"generated": 2, "started": 3},
    )
    line = eval_dbos_harness.shared_worker_monitor.worker_monitor_line(
        snapshot,
        was_active=None,
        initial_success_total=2,
        initial_failure_total=0,
        force_summary=False,
    )
    assert line is not None

    assert line.startswith("Queue Active | active=   3 | enqueued=   3")
    assert "completed=   0" in line
    assert "gen pend=   0 start=   3 done=   2 err=   0" in line
    assert "score pend=   -" in line


def test_worker_monitor_line_can_force_empty_summary(
    eval_dbos_harness,
) -> None:
    snapshot = eval_dbos_harness.WorkerQueueSnapshot(
        dbos_status_counts={"SUCCESS": 7},
        scoring_status_counts={"pending": 2, "scored": 5},
    )

    line = eval_dbos_harness.shared_worker_monitor.worker_monitor_line(
        snapshot,
        was_active=False,
        initial_success_total=2,
        initial_failure_total=0,
        force_summary=True,
    )

    assert line is not None
    assert line.startswith("Queue Empty  | active=   0")
    assert "completed=   5" in line
    assert "score pend=   2 queue=   0 start=   0 done=   5 err=   0" in line


def test_timestamped_operator_lines(eval_dbos_harness) -> None:
    line = eval_dbos_harness.shared_eval_logging.timestamped_line(
        "Queue Empty  | active=   0",
        now=datetime(2026, 1, 2, 3, 4, 5),
        timestamp_format=eval_dbos_harness.OPERATOR_TIMESTAMP_FORMAT,
    )

    assert line == "03:04:05 | Queue Empty  | active=   0"


def test_enqueue_scores_line_is_fixed_width(eval_dbos_harness) -> None:
    line = eval_dbos_harness.enqueue_scores_line(
        experiment_name="local-mock-dbos-smoke",
        selected_count=10,
        limit=1000,
        timeout=15.0,
    )

    assert line.startswith(
        "Enqueue Scores | selected=   10 | limit= 1000 | "
        "timeout=  15.0s |"
    )
    assert line.endswith("experiment=local-mock-dbos-smoke")
    assert eval_dbos_harness.enqueue_scores_style(0) == "yellow"
    assert eval_dbos_harness.enqueue_scores_style(10) == "green"


def test_fetch_started_generation_repair_candidates(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    class RepairCursor:
        def __init__(self, rows: list[tuple[Any, ...]]) -> None:
            self.rows = rows
            self.statement: str | None = None
            self.params: object = None

        def __enter__(self) -> RepairCursor:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, statement: str, params: object = None) -> None:
            self.statement = statement
            self.params = params

        def fetchall(self) -> list[tuple[Any, ...]]:
            return self.rows

    class RepairConnection:
        def __init__(self, rows: list[tuple[Any, ...]]) -> None:
            self.cursor_instance = RepairCursor(rows)

        def __enter__(self) -> RepairConnection:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def cursor(self) -> RepairCursor:
            return self.cursor_instance

    app_conn = RepairConnection(
        [
            ("pred-1", "HumanEval/1", 0, 0, "model/a", 0.1),
            ("pred-2", "HumanEval/2", 1, 0, "model/b", 0.2),
        ]
    )
    dbos_conn = RepairConnection([("generate:pred-2", "ERROR")])

    def fake_connect(database_url: str) -> RepairConnection:
        if database_url == "postgresql:///app":
            return app_conn
        if database_url == "postgresql:///dbos":
            return dbos_conn
        raise AssertionError(f"unexpected database_url: {database_url}")

    monkeypatch.setattr(eval_dbos_harness.psycopg, "connect", fake_connect)

    candidates = (
        eval_dbos_harness.shared_eval_repair
        .fetch_started_generation_repair_candidates(
            "postgresql:///app",
            dbos_system_database_url="postgresql:///dbos",
            prediction_table=eval_dbos_harness.PREDICTION_TABLE_NAME,
            experiment_name="exp",
            dimension_columns=eval_dbos_harness.REPAIR_DIMENSION_COLUMNS,
            order_columns=eval_dbos_harness.REPAIR_ORDER_COLUMNS,
        )
    )

    assert [candidate.prediction_id for candidate in candidates] == ["pred-2"]
    assert candidates[0].dbos_status == "ERROR"
    assert app_conn.cursor_instance.params == ("exp",)
    assert dbos_conn.cursor_instance.params == (
        ["generate:pred-1", "generate:pred-2"],
        list(eval_dbos_harness.shared_dbos.DBOS_FAILED_WORKFLOW_STATUSES),
    )


def test_mark_started_generations_as_repaired_errors(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    class RepairCursor(FakeCursor):
        def execute(self, statement: str, params: object = None) -> None:
            super().execute(statement, params)
            self.params = params
            self.rowcount = 2

    class RepairConnection(FakeConnection):
        def __init__(self) -> None:
            self.cursor_instance = RepairCursor()

    fake_conn = RepairConnection()
    monkeypatch.setattr(
        eval_dbos_harness.psycopg,
        "connect",
        lambda _database_url: fake_conn,
    )

    rowcount = eval_dbos_harness.mark_started_generations_as_repaired_errors(
        "postgresql:///unit",
        prediction_ids=["pred-1", "pred-2"],
    )

    assert rowcount == 2
    statement = fake_conn.cursor_instance.statements[0]
    assert "generation_status = 'generation_error'" in statement
    assert "AND generation_status = 'started'" in statement
    assert fake_conn.cursor_instance.params == (
        eval_dbos_harness.GENERATION_REPAIR_ERROR,
        ["pred-1", "pred-2"],
    )


def test_fetch_stranded_scoring_repair_candidates_finds_failed_and_missing(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    class RepairCursor:
        def __init__(self, rows: list[tuple[Any, ...]]) -> None:
            self.rows = rows
            self.statement: str | None = None
            self.params: object = None

        def __enter__(self) -> RepairCursor:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, statement: str, params: object = None) -> None:
            self.statement = statement
            self.params = params

        def fetchall(self) -> list[tuple[Any, ...]]:
            return self.rows

    class RepairConnection:
        def __init__(self, rows: list[tuple[Any, ...]]) -> None:
            self.cursor_instance = RepairCursor(rows)

        def __enter__(self) -> RepairConnection:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def cursor(self) -> RepairCursor:
            return self.cursor_instance

    app_conn = RepairConnection(
        [
            ("pred-1", "HumanEval/1", 0, 0, "queued", "model/a", 0.1),
            ("pred-2", "HumanEval/2", 1, 0, "started", "model/b", 0.2),
            ("pred-3", "HumanEval/3", 2, 0, "queued", "model/c", 0.3),
        ]
    )
    dbos_conn = RepairConnection(
        [
            ("score:pred-1", "ERROR"),
            ("score:pred-2", "ENQUEUED"),
        ]
    )

    def fake_connect(database_url: str) -> RepairConnection:
        if database_url == "postgresql:///app":
            return app_conn
        if database_url == "postgresql:///dbos":
            return dbos_conn
        raise AssertionError(f"unexpected database_url: {database_url}")

    monkeypatch.setattr(eval_dbos_harness.psycopg, "connect", fake_connect)

    candidates = (
        eval_dbos_harness.shared_eval_repair
        .fetch_stranded_scoring_repair_candidates(
            "postgresql:///app",
            dbos_system_database_url="postgresql:///dbos",
            prediction_table=eval_dbos_harness.PREDICTION_TABLE_NAME,
            experiment_name="exp",
            dimension_columns=eval_dbos_harness.REPAIR_DIMENSION_COLUMNS,
            order_columns=eval_dbos_harness.REPAIR_ORDER_COLUMNS,
            limit=1000,
        )
    )

    assert [candidate.prediction_id for candidate in candidates] == [
        "pred-1",
        "pred-3",
    ]
    assert [candidate.dbos_status for candidate in candidates] == [
        "ERROR",
        eval_dbos_harness.shared_dbos.MISSING_DBOS_WORKFLOW_STATUS,
    ]


def test_mark_stranded_scoring_as_errors(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    class RepairCursor(FakeCursor):
        def execute(self, statement: str, params: object = None) -> None:
            super().execute(statement, params)
            self.params = params
            self.rowcount = 2

    class RepairConnection(FakeConnection):
        def __init__(self) -> None:
            self.cursor_instance = RepairCursor()

    fake_conn = RepairConnection()
    monkeypatch.setattr(
        eval_dbos_harness.psycopg,
        "connect",
        lambda _database_url: fake_conn,
    )

    rowcount = eval_dbos_harness.mark_stranded_scoring_as_errors(
        "postgresql:///unit",
        prediction_ids=["pred-1", "pred-2"],
    )

    assert rowcount == 2
    statement = fake_conn.cursor_instance.statements[0]
    assert "scoring_status = 'score_error'" in statement
    assert "AND scoring_status IN ('started', 'queued')" in statement
    assert fake_conn.cursor_instance.params == (
        eval_dbos_harness.SCORING_REPAIR_ERROR,
        ["pred-1", "pred-2"],
    )


def test_reset_generation_errors_for_retry(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    class RetryCursor(FakeCursor):
        def execute(self, statement: str, params: object = None) -> None:
            super().execute(statement, params)
            self.params = params
            self.rowcount = 2

    class RetryConnection(FakeConnection):
        def __init__(self) -> None:
            self.cursor_instance = RetryCursor()

    fake_conn = RetryConnection()
    monkeypatch.setattr(
        eval_dbos_harness.psycopg,
        "connect",
        lambda _database_url: fake_conn,
    )

    rowcount = eval_dbos_harness.reset_generation_errors_for_retry(
        "postgresql:///unit",
        prediction_ids=["pred-1", "pred-2"],
    )

    assert rowcount == 2
    statement = fake_conn.cursor_instance.statements[0]
    assert "generation_status = 'pending'" in statement
    assert "scoring_status = 'pending'" in statement
    assert "AND generation_status = 'generation_error'" in statement
    assert fake_conn.cursor_instance.params == (["pred-1", "pred-2"],)


def test_direct_compatibility_helpers_are_removed(
    eval_dbos_harness,
) -> None:
    removed_names = [
        "GenerationRepairCandidate",
        "ScoringRepairCandidate",
        "RepairPlan",
        "RepairApplyResult",
        "fetch_started_generation_repair_candidates",
        "fetch_stranded_scoring_repair_candidates",
        "repair_generation_started_line",
        "repair_generation_started_style",
        "retry_generation_errors_line",
        "retry_generation_errors_style",
        "SOLVE_FIELDS",
        "SOLVE_INSTRUCTIONS",
        "DEFAULT_MODEL_CONFIGS",
        "DEFAULT_SAMPLE_COUNT",
        "DEFAULT_SEED",
        "DEFAULT_TEMPERATURE",
        "DEFAULT_TEMPERATURES",
        "DEFAULT_MAX_COMPLETION_TOKENS",
        "DEFAULT_SUBPROCESS_TIMEOUT",
        "DATASET_NAME",
        "DATASET_SPLIT",
        "app",
        "default_worker_log_path",
        "hashed_experiment_log_name",
        "timestamped_line",
        "format_cost",
        "format_cost_column",
        "format_float_column",
        "generation_workflow_id",
        "score_workflow_id",
        "worker_monitor_line",
        "worker_monitor_style",
        "build_dbos_config",
        "configure_db_connection_pools",
        "auto_db_pool_max_size",
        "resolve_db_pool_config",
    ]
    assert not [
        name
        for name in removed_names
        if hasattr(eval_dbos_harness, name)
    ]


def test_worker_detail_log_writes_prediction_context(
    eval_dbos_harness, tmp_path
) -> None:
    log_file = tmp_path / "worker.log"
    logger = eval_dbos_harness.configure_worker_file_logging(log_file)
    context = eval_dbos_harness.PredictionLogContext(
        prediction_id="pred-1",
        experiment_name="exp",
        task_id="HumanEval/1",
        sample_index=0,
        model="model/a",
        temperature=0.0,
        repetition_seed=0,
    )

    eval_dbos_harness.emit_prediction_log_event(
        "generation_started",
        context,
    )
    for handler in logger.handlers:
        handler.flush()

    text = log_file.read_text()
    assert '"event":"generation_started"' in text
    assert '"prediction_id":"pred-1"' in text
    assert '"task_id":"HumanEval/1"' in text


def test_configure_dbos_runtime_syncs_before_launch_and_registers_after(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    calls: list[str] = []

    class FakeDBOS:
        def __call__(self, *_args: object, **_kwargs: object) -> None:
            calls.append("init")

        def listen_queues(self, _queues: list[str]) -> None:
            calls.append("listen")

        def launch(self) -> None:
            calls.append("launch")

        def register_queue(self, _name: str, **_kwargs: object) -> None:
            calls.append("register")

    def fake_sync_existing_dbos_queue_concurrency(
        _config: object, *, experiment_name: str, **_kwargs: object
    ) -> int:
        assert experiment_name == "exp"
        calls.append("sync")
        return 2

    monkeypatch.setattr(eval_dbos_harness, "DBOS", FakeDBOS())
    monkeypatch.setattr(
        eval_dbos_harness.shared_dbos,
        "sync_existing_dbos_queue_concurrency",
        fake_sync_existing_dbos_queue_concurrency,
    )
    monkeypatch.setattr(
        eval_dbos_harness, "operator_log", lambda *_args, **_kwargs: None
    )

    config = eval_dbos_harness.EvalDbosConfig(
        database_url="postgresql:///app",
        dbos_system_database_url="postgresql:///dbos",
        generation_concurrency=11,
        scoring_concurrency=5,
    )

    eval_dbos_harness.configure_dbos_runtime(
        config, experiment_name="exp", consume_queues=False
    )

    assert calls == [
        "init",
        "sync",
        "listen",
        "launch",
        "register",
        "register",
    ]


def test_configure_pooled_worker_runtime_configures_pool_before_launch(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    calls: list[str] = []
    config = eval_dbos_harness.EvalDbosConfig(
        database_url="postgresql:///app",
        dbos_system_database_url="postgresql:///dbos",
        generation_concurrency=11,
        scoring_concurrency=5,
    )
    pool_config = eval_dbos_harness.DbPoolConfig(max_size=13)

    def fake_configure_worker_db_connection_pools(
        _config: object,
        *,
        queue: object,
        raw_max_size: str,
    ) -> object:
        assert queue is eval_dbos_harness.QueueSelection.SCORING
        assert raw_max_size == eval_dbos_harness.DB_POOL_AUTO
        calls.append("pool")
        return pool_config

    def fake_configure_dbos_runtime(
        _config: object, *, experiment_name: str, queue: object
    ) -> None:
        assert experiment_name == "exp"
        assert queue is eval_dbos_harness.QueueSelection.SCORING
        calls.append("launch")

    monkeypatch.setattr(
        eval_dbos_harness,
        "configure_worker_db_connection_pools",
        fake_configure_worker_db_connection_pools,
    )
    monkeypatch.setattr(
        eval_dbos_harness,
        "configure_dbos_runtime",
        fake_configure_dbos_runtime,
    )

    result = eval_dbos_harness.configure_pooled_worker_runtime(
        config,
        experiment_name="exp",
        queue=eval_dbos_harness.QueueSelection.SCORING,
        raw_db_pool_max_size=eval_dbos_harness.DB_POOL_AUTO,
    )

    assert result == pool_config
    assert calls == ["pool", "launch"]


def test_build_humaneval_samples_from_rows_is_seeded(
    eval_dbos_harness,
) -> None:
    rows = [
        {
            "task_id": f"task/{index}",
            "prompt": f"def f_{index}():\n",
            "canonical_solution": "\n    return 1\n",
            "test": (
                "def check(candidate):\n"
                "    inputs = [()]\n"
                "    results = [1]\n"
                "    for inp, exp in zip(inputs, results):\n"
                "        assert candidate(*inp) == exp\n"
            ),
            "entry_point": f"f_{index}",
        }
        for index in range(6)
    ]

    first = eval_dbos_harness.build_humaneval_samples_from_rows(
        rows, seed=3, sample_count=4
    )
    second = eval_dbos_harness.build_humaneval_samples_from_rows(
        rows, seed=3, sample_count=4
    )
    different = eval_dbos_harness.build_humaneval_samples_from_rows(
        rows, seed=4, sample_count=4
    )

    assert [sample.task_id for sample in first] == [
        sample.task_id for sample in second
    ]
    assert [sample.task_id for sample in first] != [
        sample.task_id for sample in different
    ]
    assert [sample.sample_index for sample in first] == [0, 1, 2, 3]


def test_build_prediction_jobs_uses_stable_ids(eval_dbos_harness) -> None:
    samples = [
        eval_dbos_harness.HumanEvalSample(
            task_id="task/add",
            sample_index=0,
            prompt="def add(a, b): pass",
            test="def check(candidate): pass",
            entry_point="add",
        )
    ]
    models = [
        eval_dbos_harness.ModelConfig(
            model="model/a", reasoning={"enabled": False}
        ),
        eval_dbos_harness.ModelConfig(
            model="model/b", reasoning={"effort": "low"}
        ),
    ]

    first = eval_dbos_harness.build_prediction_jobs(
        experiment_name="exp",
        submission_id="sub-1",
        samples=samples,
        model_configs=models,
        temperatures=[0.0, 0.2],
        repetitions=2,
    )
    second = eval_dbos_harness.build_prediction_jobs(
        experiment_name="exp",
        submission_id="sub-2",
        samples=samples,
        model_configs=models,
        temperatures=[0.0, 0.2],
        repetitions=2,
    )

    assert len(first) == 8
    assert {job.prediction_id for job in first} == {
        job.prediction_id for job in second
    }
    assert first[0].submission_id == "sub-1"
    assert second[0].submission_id == "sub-2"


def test_insert_prediction_jobs_does_not_overwrite_existing_rows(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    fake_conn = FakeConnection()

    def fake_connect(database_url: str) -> FakeConnection:
        assert database_url == "postgresql:///unit"
        return fake_conn

    monkeypatch.setattr(eval_dbos_harness.psycopg, "connect", fake_connect)

    jobs = eval_dbos_harness.build_prediction_jobs(
        experiment_name="exp",
        submission_id="sub",
        samples=[
            eval_dbos_harness.HumanEvalSample(
                task_id="task/add",
                sample_index=0,
                prompt="def add(a, b): pass",
                test="def check(candidate): pass",
                entry_point="add",
            )
        ],
        model_configs=[
            eval_dbos_harness.ModelConfig(model="model/a", reasoning={})
        ],
        temperatures=[0.0],
        repetitions=1,
    )

    eval_dbos_harness.insert_prediction_jobs("postgresql:///unit", jobs)

    statement = fake_conn.cursor_instance.statements[0]
    assert "ON CONFLICT (prediction_id) DO NOTHING" in statement


def test_enqueue_generation_jobs_uses_stable_workflow_ids(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    workflow_ids: list[str] = []
    enqueued: list[dict[str, Any]] = []

    class FakeSetWorkflowID:
        def __init__(self, workflow_id: str) -> None:
            self.workflow_id = workflow_id

        def __enter__(self) -> None:
            workflow_ids.append(self.workflow_id)

        def __exit__(self, *_args: object) -> None:
            return None

    def fake_enqueue_workflow(
        queue_name: str,
        workflow: object,
        database_url: str,
        prediction_id: str,
        experiment_name: str,
        score_timeout: float,
    ) -> None:
        enqueued.append(
            {
                "queue_name": queue_name,
                "workflow": workflow,
                "database_url": database_url,
                "prediction_id": prediction_id,
                "experiment_name": experiment_name,
                "score_timeout": score_timeout,
            }
        )

    monkeypatch.setattr(eval_dbos_harness, "SetWorkflowID", FakeSetWorkflowID)
    monkeypatch.setattr(
        eval_dbos_harness.DBOS, "enqueue_workflow", fake_enqueue_workflow
    )
    job = eval_dbos_harness.PredictionJob(
        prediction_id="abc",
        experiment_name="exp",
        submission_id="sub",
        task_id="task/add",
        sample_index=0,
        model="model/a",
        temperature=0.0,
        repetition_seed=0,
        prompt="def add(a, b): pass",
        test="def check(candidate): pass",
        entry_point="add",
    )

    eval_dbos_harness.enqueue_generation_jobs(
        "postgresql:///unit",
        [job],
        score_timeout=7.0,
    )

    assert workflow_ids == ["generate:abc"]
    assert enqueued == [
        {
            "queue_name": eval_dbos_harness.eval_queue_names(
                "exp"
            ).generation,
            "workflow": eval_dbos_harness.generate_prediction_workflow,
            "database_url": "postgresql:///unit",
            "prediction_id": "abc",
            "experiment_name": "exp",
            "score_timeout": 7.0,
        }
    ]


def test_enqueue_generation_jobs_can_use_retry_workflow_ids(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    workflow_ids: list[str] = []

    class FakeSetWorkflowID:
        def __init__(self, workflow_id: str) -> None:
            self.workflow_id = workflow_id

        def __enter__(self) -> None:
            workflow_ids.append(self.workflow_id)

        def __exit__(self, *_args: object) -> None:
            return None

    def fake_enqueue_workflow(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(eval_dbos_harness, "SetWorkflowID", FakeSetWorkflowID)
    monkeypatch.setattr(
        eval_dbos_harness.DBOS, "enqueue_workflow", fake_enqueue_workflow
    )
    job = eval_dbos_harness.PredictionJob(
        prediction_id="abc",
        experiment_name="exp",
        submission_id="sub",
        task_id="task/add",
        sample_index=0,
        model="model/a",
        temperature=0.0,
        repetition_seed=0,
        prompt="def add(a, b): pass",
        test="def check(candidate): pass",
        entry_point="add",
    )

    eval_dbos_harness.enqueue_generation_jobs(
        "postgresql:///unit",
        [job],
        score_timeout=7.0,
        retry_token="retry-1",
    )

    assert workflow_ids == ["generate-retry:retry-1:abc"]
    assert (
        eval_dbos_harness.shared_dbos.generation_workflow_id("abc")
        == "generate:abc"
    )


def test_generation_workflow_enqueues_scoring_after_success(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    calls: list[tuple[str, object]] = []
    result = eval_dbos_harness.GenerationResult(
        prediction_id="abc",
        raw_generation="def add(a, b): return a + b",
    )

    def fake_generate_prediction_step(
        database_url: str, prediction_id: str
    ) -> object:
        calls.append(
            (
                "generate",
                {
                    "database_url": database_url,
                    "prediction_id": prediction_id,
                },
            )
        )
        return result

    def fake_record_generation_success_step(
        database_url: str, generation_result: object
    ) -> None:
        calls.append(
            (
                "record_generation_success",
                {
                    "database_url": database_url,
                    "result": generation_result,
                },
            )
        )

    def fake_enqueue_score_job(
        database_url: str,
        prediction_id: str,
        *,
        experiment_name: str,
        timeout: float,
    ) -> None:
        calls.append(
            (
                "enqueue_score",
                {
                    "database_url": database_url,
                    "prediction_id": prediction_id,
                    "experiment_name": experiment_name,
                    "timeout": timeout,
                },
            )
        )

    def fake_mark_scoring_queued_step(
        database_url: str, prediction_id: str
    ) -> None:
        calls.append(
            (
                "mark_scoring_queued",
                {
                    "database_url": database_url,
                    "prediction_id": prediction_id,
                },
            )
        )

    def fail_record_generation_error_step(
        _database_url: str, _prediction_id: str, _error: str
    ) -> None:
        raise AssertionError("generation should not be marked failed")

    monkeypatch.setattr(
        eval_dbos_harness,
        "generate_prediction_step",
        fake_generate_prediction_step,
    )
    monkeypatch.setattr(
        eval_dbos_harness,
        "record_generation_success_step",
        fake_record_generation_success_step,
    )
    monkeypatch.setattr(
        eval_dbos_harness,
        "enqueue_score_job",
        fake_enqueue_score_job,
    )
    monkeypatch.setattr(
        eval_dbos_harness,
        "mark_scoring_queued_step",
        fake_mark_scoring_queued_step,
    )
    monkeypatch.setattr(
        eval_dbos_harness,
        "record_generation_error_step",
        fail_record_generation_error_step,
    )

    status = eval_dbos_harness.generate_prediction_workflow.__wrapped__(
        "postgresql:///unit",
        "abc",
        "exp",
        9.0,
    )

    assert status == "generated"
    assert calls == [
        (
            "generate",
            {
                "database_url": "postgresql:///unit",
                "prediction_id": "abc",
            },
        ),
        (
            "record_generation_success",
            {"database_url": "postgresql:///unit", "result": result},
        ),
        (
            "enqueue_score",
            {
                "database_url": "postgresql:///unit",
                "prediction_id": "abc",
                "experiment_name": "exp",
                "timeout": 9.0,
            },
        ),
        (
            "mark_scoring_queued",
            {"database_url": "postgresql:///unit", "prediction_id": "abc"},
        ),
    ]


def test_generation_workflow_does_not_enqueue_scoring_after_failure(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    calls: list[tuple[str, object]] = []

    def fake_generate_prediction_step(
        _database_url: str, _prediction_id: str
    ) -> object:
        raise RuntimeError("generation failed")

    def fake_record_generation_error_step(
        database_url: str, prediction_id: str, error: str
    ) -> None:
        calls.append(
            (
                "record_generation_error",
                {
                    "database_url": database_url,
                    "prediction_id": prediction_id,
                    "error": error,
                },
            )
        )

    def fail_enqueue_score_job(
        _database_url: str,
        _prediction_id: str,
        *,
        experiment_name: str,
        timeout: float,
    ) -> None:
        raise AssertionError("scoring should not be enqueued")

    def fail_mark_scoring_queued_step(
        _database_url: str, _prediction_id: str
    ) -> None:
        raise AssertionError("scoring should not be marked queued")

    monkeypatch.setattr(
        eval_dbos_harness,
        "generate_prediction_step",
        fake_generate_prediction_step,
    )
    monkeypatch.setattr(
        eval_dbos_harness,
        "record_generation_error_step",
        fake_record_generation_error_step,
    )
    monkeypatch.setattr(
        eval_dbos_harness,
        "enqueue_score_job",
        fail_enqueue_score_job,
    )
    monkeypatch.setattr(
        eval_dbos_harness,
        "mark_scoring_queued_step",
        fail_mark_scoring_queued_step,
    )

    status = eval_dbos_harness.generate_prediction_workflow.__wrapped__(
        "postgresql:///unit",
        "abc",
        "exp",
        9.0,
    )

    assert status == "generation_error"
    assert calls == [
        (
            "record_generation_error",
            {
                "database_url": "postgresql:///unit",
                "prediction_id": "abc",
                "error": "RuntimeError('generation failed')",
            },
        )
    ]


def test_build_generation_lm_passes_temperature_and_reasoning(
    eval_dbos_harness,
) -> None:
    job = eval_dbos_harness.PredictionJob(
        prediction_id="abc",
        experiment_name="exp",
        submission_id="sub",
        task_id="task/add",
        sample_index=0,
        model="openai/gpt-5-nano",
        temperature=0.2,
        repetition_seed=0,
        prompt="def add(a, b): pass",
        test="def check(candidate): pass",
        entry_point="add",
        reasoning={"effort": "minimal", "exclude": False},
    )
    client = FakeClient()
    buffer = eval_dbos_harness.LmEventBuffer()

    lm = eval_dbos_harness.build_generation_lm(
        job, event_buffer=buffer, client=client
    )
    lm.forward(messages=[{"role": "user", "content": "hello"}])

    assert client.chat.completions.calls == [
        {
                "model": "openai/gpt-5-nano",
                "messages": [{"role": "user", "content": "hello"}],
                "max_completion_tokens": (
                    eval_dbos_harness.experiment_config()
                    .default_max_completion_tokens
                ),
            "temperature": 0.2,
            "extra_body": {
                "reasoning": {"effort": "minimal", "exclude": False}
            },
        }
    ]


def test_lm_event_buffer_extracts_latest_response_text(
    eval_dbos_harness,
) -> None:
    buffer = eval_dbos_harness.LmEventBuffer()
    buffer.put_event(
        "lm.response",
        payload={
            "response": {
                "choices": [
                    {"message": {"content": "def add(a, b): return a + b"}}
                ]
            }
        },
    )

    assert buffer.latest_response_text() == "def add(a, b): return a + b"
    assert buffer.has_latest_response() is True


def test_generate_code_for_job_stores_empty_raw_generation_for_null_response(
    eval_dbos_harness,
) -> None:
    job = eval_dbos_harness.PredictionJob(
        prediction_id="abc",
        experiment_name="exp",
        submission_id="sub",
        task_id="task/add",
        sample_index=0,
        model="openai/gpt-5-nano",
        temperature=0.0,
        repetition_seed=0,
        prompt="def add(a, b):\n    pass\n",
        test="def check(candidate): pass",
        entry_point="add",
    )

    result = eval_dbos_harness.generate_code_for_job(
        job, client=FakeClient(content=None)
    )

    assert result.prediction_id == "abc"
    assert result.raw_generation == ""
    assert result.response_metadata["choices"][0]["message"]["content"] is None


def test_score_generated_code_records_pass_and_failure(
    eval_dbos_harness,
) -> None:
    test = (
        "def check(candidate):\n"
        "    inputs = [(1, 2)]\n"
        "    results = [3]\n"
        "    for inp, exp in zip(inputs, results):\n"
        "        assert candidate(*inp) == exp\n"
    )
    prompt = "def add(a, b):\n"
    canonical_solution = "    return a + b\n"
    ground_truth_code = prompt + canonical_solution
    passing = eval_dbos_harness.ScoringTarget(
        prediction_id="pass",
        task_id="task/add",
        prompt=prompt,
        canonical_solution=canonical_solution,
        ground_truth_code=ground_truth_code,
        raw_generation="def add(a, b):\n    return a + b\n",
        test=test,
        entry_point="add",
    )
    failing = eval_dbos_harness.ScoringTarget(
        prediction_id="fail",
        task_id="task/add",
        prompt=prompt,
        canonical_solution=canonical_solution,
        ground_truth_code=ground_truth_code,
        raw_generation="def add(a, b):\n    return a - b\n",
        test=test,
        entry_point="add",
    )
    differently_named = eval_dbos_harness.ScoringTarget(
        prediction_id="renamed",
        task_id="task/add",
        prompt=prompt,
        canonical_solution=canonical_solution,
        ground_truth_code=ground_truth_code,
        raw_generation="def solve(a, b):\n    return a + b\n",
        test=test,
        entry_point="add",
    )

    pass_result = eval_dbos_harness.score_generated_code(
        passing, timeout=5.0
    )
    fail_result = eval_dbos_harness.score_generated_code(
        failing, timeout=5.0
    )
    renamed_result = eval_dbos_harness.score_generated_code(
        differently_named, timeout=5.0
    )

    assert pass_result.score == 1.0
    assert pass_result.error is None
    assert pass_result.raw_code == "def add(a, b):\n    return a + b"
    assert pass_result.raw_compile_ok is True
    assert pass_result.extracted_compile_ok is True
    assert pass_result.compression_metrics
    assert pass_result.compression_metrics[0].representation_bytes == len(
        prompt.encode("utf-8")
    )
    assert pass_result.compression_metrics[0].ground_truth_bytes == len(
        ground_truth_code.encode("utf-8")
    )
    assert pass_result.best_compression_ratio is not None
    assert renamed_result.score == 1.0
    assert renamed_result.error is None
    assert renamed_result.raw_code == "def solve(a, b):\n    return a + b"
    assert fail_result.score == 0.0
    assert fail_result.error == "HumanEval tests failed"


def test_score_generated_code_recovers_raw_compile_failure_by_extraction(
    eval_dbos_harness,
) -> None:
    target = eval_dbos_harness.ScoringTarget(
        prediction_id="pass",
        task_id="task/add",
        raw_generation=(
            "Here is the code:\n"
            "```python\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "```"
        ),
        test=(
            "def check(candidate):\n"
            "    inputs = [(1, 2)]\n"
            "    results = [3]\n"
            "    for inp, exp in zip(inputs, results):\n"
            "        assert candidate(*inp) == exp\n"
        ),
        entry_point="add",
    )

    result = eval_dbos_harness.score_generated_code(target, timeout=5.0)

    assert result.score == 1.0
    assert result.raw_compile_ok is False
    assert result.raw_compile_error is not None
    assert result.extracted_compile_ok is True
    assert result.extraction_candidate_count == 1
    assert result.raw_code == "def add(a, b):\n    return a + b"


def test_score_generated_code_scores_zero_when_no_candidate_compiles(
    eval_dbos_harness,
) -> None:
    target = eval_dbos_harness.ScoringTarget(
        prediction_id="fail",
        task_id="task/add",
        raw_generation="def broken(:\n    pass\n",
        test=(
            "def check(candidate):\n"
            "    inputs = [(1, 2)]\n"
            "    results = [3]\n"
            "    for inp, exp in zip(inputs, results):\n"
            "        assert candidate(*inp) == exp\n"
        ),
        entry_point="add",
    )

    result = eval_dbos_harness.score_generated_code(target, timeout=5.0)

    assert result.score == 0.0
    assert result.error == "no compilable extracted candidate"
    assert result.raw_code is None
    assert result.raw_compile_ok is False
    assert result.extracted_compile_ok is False
    assert result.extracted_compile_error is not None
    assert result.extraction_error == "no compilable extracted candidate"


def test_score_generated_code_scores_zero_for_empty_raw_generation(
    eval_dbos_harness,
) -> None:
    target = eval_dbos_harness.ScoringTarget(
        prediction_id="empty",
        task_id="task/add",
        raw_generation="",
        test=(
            "def check(candidate):\n"
            "    inputs = [(1, 2)]\n"
            "    results = [3]\n"
            "    for inp, exp in zip(inputs, results):\n"
            "        assert candidate(*inp) == exp\n"
        ),
        entry_point="add",
    )

    result = eval_dbos_harness.score_generated_code(target, timeout=5.0)

    assert result.score == 0.0
    assert result.error == "empty raw generation"
    assert result.raw_code is None
    assert result.raw_compile_ok is False
    assert result.raw_compile_error == "empty raw generation"
    assert result.extraction_candidate_count == 0
    assert result.extracted_compile_ok is False
    assert result.extraction_error == "empty raw generation"


def test_record_score_success_persists_extraction_metrics(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    fake_conn = FakeConnection()
    monkeypatch.setattr(
        eval_dbos_harness.psycopg,
        "connect",
        lambda _database_url: fake_conn,
    )
    result = eval_dbos_harness.ScoreResult(
        prediction_id="pred-1",
        score=0.0,
        error="AssertionError",
        raw_code="def add(a, b):\n    return a - b",
        raw_compile_ok=False,
        raw_compile_error="SyntaxError: invalid syntax",
        extraction_candidate_count=1,
        extracted_compile_ok=True,
        extracted_compile_error=None,
        extraction_error=None,
        stdout="out",
        stderr="err",
        stdout_truncated=True,
        stderr_truncated=False,
    )

    eval_dbos_harness.record_score_success("postgresql:///unit", result)

    statement = fake_conn.cursor_instance.statements[0]
    assert "raw_code = %s" in statement
    assert "raw_compile_ok = %s" in statement
    assert "score_stdout = %s" in statement
    params = fake_conn.cursor_instance.params
    assert params[:10] == (
        0.0,
        "AssertionError",
        "def add(a, b):\n    return a - b",
        False,
        "SyntaxError: invalid syntax",
        1,
        None,
        True,
        None,
        None,
    )
    assert params[10].obj == []
    assert params[11:13] == (None, None)
    assert params[13].obj == {}
    assert params[14].obj == []
    assert params[15:17] == (None, None)
    assert params[17:] == ("out", "err", True, False, "pred-1")


def test_enqueue_score_jobs_uses_stable_workflow_ids(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    workflow_ids: list[str] = []
    enqueued: list[dict[str, Any]] = []

    class FakeSetWorkflowID:
        def __init__(self, workflow_id: str) -> None:
            self.workflow_id = workflow_id

        def __enter__(self) -> None:
            workflow_ids.append(self.workflow_id)

        def __exit__(self, *_args: object) -> None:
            return None

    def fake_enqueue_workflow(
        queue_name: str,
        workflow: object,
        database_url: str,
        prediction_id: str,
        timeout: float,
    ) -> None:
        enqueued.append(
            {
                "queue_name": queue_name,
                "workflow": workflow,
                "database_url": database_url,
                "prediction_id": prediction_id,
                "timeout": timeout,
            }
        )

    monkeypatch.setattr(eval_dbos_harness, "SetWorkflowID", FakeSetWorkflowID)
    monkeypatch.setattr(
        eval_dbos_harness.DBOS, "enqueue_workflow", fake_enqueue_workflow
    )

    eval_dbos_harness.enqueue_score_jobs(
        "postgresql:///unit",
        ["abc"],
        experiment_name="exp",
        timeout=7.0,
    )

    assert workflow_ids == ["score:abc"]
    assert enqueued == [
        {
            "queue_name": eval_dbos_harness.eval_queue_names("exp").scoring,
            "workflow": eval_dbos_harness.score_prediction_workflow,
            "database_url": "postgresql:///unit",
            "prediction_id": "abc",
            "timeout": 7.0,
        }
    ]


def test_enqueue_score_jobs_can_use_retry_workflow_ids(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    workflow_ids: list[str] = []

    class FakeSetWorkflowID:
        def __init__(self, workflow_id: str) -> None:
            self.workflow_id = workflow_id

        def __enter__(self) -> None:
            workflow_ids.append(self.workflow_id)

        def __exit__(self, *_args: object) -> None:
            return None

    def fake_enqueue_workflow(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(eval_dbos_harness, "SetWorkflowID", FakeSetWorkflowID)
    monkeypatch.setattr(
        eval_dbos_harness.DBOS, "enqueue_workflow", fake_enqueue_workflow
    )

    eval_dbos_harness.enqueue_score_jobs(
        "postgresql:///unit",
        ["abc"],
        experiment_name="exp",
        timeout=7.0,
        retry_token="repair-1",
    )

    assert workflow_ids == ["score-retry:repair-1:abc"]
    assert (
        eval_dbos_harness.shared_dbos.score_workflow_id("abc")
        == "score:abc"
    )


def test_apply_repair_reconciles_before_selecting_retries(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    calls: list[str] = []
    config = eval_dbos_harness.EvalDbosConfig(
        database_url="postgresql:///app",
        dbos_system_database_url="postgresql:///dbos",
        generation_concurrency=11,
        scoring_concurrency=5,
    )
    job = eval_dbos_harness.PredictionJob(
        prediction_id="gen-1",
        experiment_name="exp",
        submission_id="sub",
        task_id="task/add",
        sample_index=0,
        model="model/a",
        temperature=0.0,
        repetition_seed=0,
        prompt="def add(a, b): pass",
        test="def check(candidate): pass",
        entry_point="add",
    )

    shared_repair = eval_dbos_harness.shared_eval_repair

    def fake_fetch_started_generation_repair_candidates(
        *_args: object, **_kwargs: object
    ) -> list[Any]:
        return [
            shared_repair.RepairCandidate(
                prediction_id="gen-1",
                task_id="task/add",
                sample_index=0,
                repetition_seed=0,
                dbos_status="ERROR",
                dimensions={"model": "model/a", "temperature": 0.0},
            )
        ]

    def fake_mark_started_generations_as_repaired_errors(
        *_args: object, **_kwargs: object
    ) -> int:
        calls.append("mark_generation")
        return 1

    def fake_fetch_generation_error_prediction_ids(
        *_args: object, **_kwargs: object
    ) -> list[str]:
        assert calls == ["mark_generation"]
        calls.append("fetch_generation_errors")
        return ["gen-1"]

    def fake_fetch_prediction_job(*_args: object, **_kwargs: object) -> Any:
        return job

    def fake_configure_dbos_runtime(*_args: object, **_kwargs: object) -> None:
        calls.append("configure")

    def fake_enqueue_generation_jobs(
        *_args: object, retry_token: str | None = None, **_kwargs: object
    ) -> None:
        assert retry_token == "repair-token"
        calls.append("enqueue_generation_retry")

    def fake_reset_generation_errors_for_retry(
        *_args: object, **_kwargs: object
    ) -> int:
        calls.append("reset_generation")
        return 1

    def fake_fetch_stranded_scoring_repair_candidates(
        *_args: object, **_kwargs: object
    ) -> list[Any]:
        assert "reset_generation" in calls
        calls.append("fetch_stranded_scoring")
        return [
            shared_repair.RepairCandidate(
                prediction_id="score-1",
                task_id="task/add",
                sample_index=0,
                repetition_seed=0,
                scoring_status="queued",
                dbos_status="ERROR",
                dimensions={"model": "model/a", "temperature": 0.0},
            )
        ]

    def fake_mark_stranded_scoring_as_errors(
        *_args: object, **_kwargs: object
    ) -> int:
        calls.append("mark_scoring")
        return 1

    def fake_fetch_pending_scoring_prediction_ids(
        *_args: object, **_kwargs: object
    ) -> list[str]:
        assert calls[-1] == "mark_scoring"
        calls.append("fetch_pending_scoring")
        return ["score-2"]

    def fake_enqueue_score_jobs(
        *_args: object, retry_token: str | None = None, **_kwargs: object
    ) -> None:
        calls.append(
            "enqueue_scoring_retry"
            if retry_token == "repair-token"
            else "enqueue_pending_scoring"
        )

    def fake_mark_scoring_queued(*_args: object, **_kwargs: object) -> int:
        calls.append("mark_scoring_queued")
        return 1

    def fake_fetch_score_error_prediction_ids(
        *_args: object, **_kwargs: object
    ) -> list[str]:
        assert "mark_scoring" in calls
        calls.append("fetch_score_errors")
        return ["score-1"]

    monkeypatch.setattr(
        shared_repair,
        "fetch_started_generation_repair_candidates",
        fake_fetch_started_generation_repair_candidates,
    )
    monkeypatch.setattr(
        shared_repair,
        "mark_started_generations_as_repaired_errors",
        fake_mark_started_generations_as_repaired_errors,
    )
    monkeypatch.setattr(
        shared_repair,
        "fetch_generation_error_prediction_ids",
        fake_fetch_generation_error_prediction_ids,
    )
    monkeypatch.setattr(
        eval_dbos_harness, "fetch_prediction_job", fake_fetch_prediction_job
    )
    monkeypatch.setattr(
        eval_dbos_harness,
        "configure_dbos_runtime",
        fake_configure_dbos_runtime,
    )
    monkeypatch.setattr(
        eval_dbos_harness,
        "enqueue_generation_jobs",
        fake_enqueue_generation_jobs,
    )
    monkeypatch.setattr(
        eval_dbos_harness,
        "reset_generation_errors_for_retry",
        fake_reset_generation_errors_for_retry,
    )
    monkeypatch.setattr(
        shared_repair,
        "fetch_stranded_scoring_repair_candidates",
        fake_fetch_stranded_scoring_repair_candidates,
    )
    monkeypatch.setattr(
        shared_repair,
        "mark_stranded_scoring_as_errors",
        fake_mark_stranded_scoring_as_errors,
    )
    monkeypatch.setattr(
        shared_repair,
        "fetch_pending_scoring_prediction_ids",
        fake_fetch_pending_scoring_prediction_ids,
    )
    monkeypatch.setattr(
        eval_dbos_harness, "enqueue_score_jobs", fake_enqueue_score_jobs
    )
    monkeypatch.setattr(
        shared_repair, "mark_scoring_queued", fake_mark_scoring_queued
    )
    monkeypatch.setattr(
        shared_repair,
        "fetch_score_error_prediction_ids",
        fake_fetch_score_error_prediction_ids,
    )

    result = eval_dbos_harness.apply_repair(
        config,
        experiment_name="exp",
        generation_limit=1000,
        scoring_limit=1000,
        score_timeout=7.0,
        repair_token="repair-token",
    )

    assert calls == [
        "mark_generation",
        "fetch_generation_errors",
        "configure",
        "enqueue_generation_retry",
        "reset_generation",
        "fetch_stranded_scoring",
        "mark_scoring",
        "fetch_pending_scoring",
        "enqueue_pending_scoring",
        "mark_scoring_queued",
        "fetch_score_errors",
        "enqueue_scoring_retry",
        "mark_scoring_queued",
    ]
    assert result.repair_token == "repair-token"
    assert result.generation_retries_enqueued == 1
    assert result.scoring_retries_enqueued == 1


def test_mark_scoring_queued_only_updates_waiting_predictions(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    fake_conn = FakeConnection()
    monkeypatch.setattr(
        eval_dbos_harness.psycopg,
        "connect",
        lambda _database_url: fake_conn,
    )

    eval_dbos_harness.mark_scoring_queued("postgresql:///unit", ["abc"])

    statement = fake_conn.cursor_instance.statements[0]
    assert "scoring_status = 'queued'" in statement
    assert "AND scoring_status IN ('pending', 'score_error')" in statement


def test_summarize_analysis_records_includes_variance(
    eval_dbos_harness,
) -> None:
    records = [
        eval_dbos_harness.AnalysisRecord(
            model="model/a",
            temperature=0.0,
            task_id="task/1",
            repetition_seed=0,
            score=1.0,
            provider_cost=0.01,
            raw_compile_ok=False,
            extracted_compile_ok=True,
        ),
        eval_dbos_harness.AnalysisRecord(
            model="model/a",
            temperature=0.0,
            task_id="task/1",
            repetition_seed=1,
            score=0.0,
            provider_cost=0.03,
            raw_compile_ok=False,
            extracted_compile_ok=False,
        ),
        eval_dbos_harness.AnalysisRecord(
            model="model/a",
            temperature=0.0,
            task_id="task/2",
            repetition_seed=0,
            score=1.0,
            provider_cost=0.02,
            raw_compile_ok=True,
            extracted_compile_ok=True,
        ),
    ]

    summaries = eval_dbos_harness.summarize_analysis_records(records)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.model == "model/a"
    assert summary.sample_count == 2
    assert summary.scored_count == 3
    assert summary.total_price == pytest.approx(0.06)
    assert summary.avg_price_per_sample == pytest.approx(0.02)
    assert summary.avg_performance == pytest.approx(2 / 3)
    assert summary.price_variance == pytest.approx(0.0001)
    assert summary.performance_variance == pytest.approx(1 / 3)
    assert summary.avg_repetition_variance == pytest.approx(0.5)
    assert summary.raw_compile_pass_count == 1
    assert summary.extracted_compile_pass_count == 2
    assert summary.extraction_lift == 1


def test_analysis_markdown_and_csv(eval_dbos_harness, tmp_path) -> None:
    summaries = [
        eval_dbos_harness.AnalysisSummary(
            model="model/a",
            temperature=0.0,
            sample_count=2,
            scored_count=3,
            total_price=0.06,
            avg_price_per_sample=0.02,
            price_variance=0.0001,
            avg_performance=2 / 3,
            performance_variance=1 / 3,
            avg_repetition_variance=0.5,
            raw_compile_pass_count=1,
            extracted_compile_pass_count=2,
            extraction_lift=1,
        )
    ]

    markdown = eval_dbos_harness.analysis_markdown(
        experiment_name="exp", summaries=summaries
    )
    csv_path = tmp_path / "analysis.csv"
    eval_dbos_harness.write_analysis_csv(summaries, csv_path=csv_path)

    assert "# Eval Analysis: exp" in markdown
    assert "model/a" in markdown
    assert "Avg Price/1k Samples" in markdown
    assert "Extraction Lift" in markdown
    assert "0.06" in markdown
    assert "20" in markdown
    assert "0.666667" in markdown
    assert "| Total |  | 2 | 3 | 0.06 |  |  | 1 | 2 | 1 |  |  |  |" in markdown
    csv_text = csv_path.read_text()
    assert "model,temperature,sample_count" in csv_text
    assert "model/a" in csv_text


def test_cost_formatting_uses_fixed_decimal(eval_dbos_harness) -> None:
    assert shared_analysis.format_float_column(
        [1.0, 2 / 3, 0.0, None]
    ) == ["1.000000", "0.666667", "0.000000", ""]
    assert shared_analysis.format_float_column(
        [4.50062e-08, 0.333333, 5e-08]
    ) == ["    4.50062e-08", "    0.333333000", "5e-08          "]
    assert (
        shared_analysis.format_cost(5.0862e-05)
        == "0.000050862"
    )
    assert (
        shared_analysis.format_cost(0.0007023)
        == "0.0007023"
    )
    assert shared_analysis.format_cost(47.12) == "47.12"
    assert shared_analysis.format_cost(0.0) == "0"
    assert shared_analysis.format_cost(None) == ""
    assert shared_analysis.format_cost_column(
        [5.0862e-05, 0.0007023, 0.00011185, 5.492e-05, 4.712e-05]
    ) == [
        "0.000050862",
        "0.000702300",
        "0.000111850",
        "0.000054920",
        "0.000047120",
    ]
    assert shared_analysis.format_cost_column(
        [0.050862, 0.7023, 0.11185, 0.05492, 0.04712]
    ) == [
        "0.050862",
        "0.702300",
        "0.111850",
        "0.054920",
        "0.047120",
    ]
    assert shared_analysis.format_cost_column(
        [0.0, 0.1]
    ) == [
        "0.0",
        "0.1",
    ]
    assert shared_analysis.price_per_thousand_samples(
        5.0862e-05
    ) == pytest.approx(0.050862)


def test_status_and_analysis_tables_render(eval_dbos_harness) -> None:
    status_table = eval_dbos_harness.status_counts_table(
        [
            {
                "experiment_name": "exp",
                "model": "model/a",
                "temperature": 0.0,
                "generation_status": "generated",
                "scoring_status": "scored",
                "count": 2,
            }
        ],
        experiment_name="exp",
    )
    analysis_table = eval_dbos_harness.analysis_table(
        experiment_name="exp",
        summaries=[
            eval_dbos_harness.AnalysisSummary(
                model="model/a",
                temperature=0.0,
                sample_count=2,
                scored_count=2,
                total_price=None,
                avg_price_per_sample=None,
                price_variance=None,
                avg_performance=1.0,
                performance_variance=None,
                avg_repetition_variance=None,
                raw_compile_pass_count=1,
                extracted_compile_pass_count=2,
                extraction_lift=1,
            ),
            eval_dbos_harness.AnalysisSummary(
                model="model/b",
                temperature=0.0,
                sample_count=2,
                scored_count=2,
                total_price=5.0862e-05,
                avg_price_per_sample=5.0862e-05,
                price_variance=None,
                avg_performance=1.0,
                performance_variance=None,
                avg_repetition_variance=None,
                raw_compile_pass_count=2,
                extracted_compile_pass_count=2,
                extraction_lift=0,
            ),
            eval_dbos_harness.AnalysisSummary(
                model="model/c",
                temperature=0.0,
                sample_count=2,
                scored_count=2,
                total_price=0.0007023,
                avg_price_per_sample=0.0007023,
                price_variance=4.50062e-08,
                avg_performance=2 / 3,
                performance_variance=1 / 3,
                avg_repetition_variance=None,
                raw_compile_pass_count=0,
                extracted_compile_pass_count=2,
                extraction_lift=2,
            ),
        ],
    )
    console = Console(record=True, width=120)
    console.print(status_table)
    console.print(analysis_table)
    text = console.export_text()
    performance_table, cost_table, variance_table = analysis_table.renderables

    assert "Eval Status: exp" in text
    assert "Generation" in text
    assert "Scoring" in text
    assert status_table.row_styles == list(
        eval_dbos_harness.TABLE_ROW_STYLES
    )
    assert "Eval Analysis: exp" in text
    assert "Avg $/1k Samples" in text
    assert performance_table.row_styles == list(
        eval_dbos_harness.TABLE_ROW_STYLES
    )
    assert cost_table.row_styles == list(eval_dbos_harness.TABLE_ROW_STYLES)
    assert variance_table.row_styles == list(
        eval_dbos_harness.TABLE_ROW_STYLES
    )
    assert (
        performance_table.rows[-1].style
        == eval_dbos_harness.TABLE_TOTAL_ROW_STYLE
    )
    assert (
        cost_table.rows[-1].style
        == eval_dbos_harness.TABLE_TOTAL_ROW_STYLE
    )
    assert "0.000050862" in text
    assert "0.000702300" in text
    assert "0.050862" in text
    assert "0.702300" in text
    assert "1.000000" in text
    assert "0.666667" in text
    assert "4.50062e-08" in text
    assert "0.333333" in text
    assert "Total" in text
    assert "0.000753162" in text
    assert "Variance" in text
