from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from dr_dspy.graph import GraphSpec
from dr_dspy.platform import worker
from tests.support.jsonl_fixtures import write_prediction_specs_jsonl
from tests.support.platform_workflow_fixtures import (
    direct_node,
    prediction_spec,
)


def test_score_one_wires_scoring_workflow_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RuntimeConfig:
        database_url = "postgresql://example/db"

    calls: list[tuple[str, Any]] = []

    def load_env_file(env_file: Any = None) -> None:
        calls.append(("load_env", env_file))

    def configure_runtime(
        *,
        database_url: str | None,
        dbos_system_database_url: str | None,
        consume_generation_queue: bool,
        database_url_error_suffix: str = "",
    ) -> RuntimeConfig:
        calls.append(
            (
                "configure",
                (
                    database_url,
                    dbos_system_database_url,
                    consume_generation_queue,
                    database_url_error_suffix,
                ),
            )
        )
        return RuntimeConfig()

    def run_once(**kwargs: Any) -> SimpleNamespace:
        calls.append(("score_once", kwargs))
        return SimpleNamespace(
            score_attempt_id="score-1",
            insert_status="inserted",
        )

    def destroy_runtime() -> None:
        calls.append(("destroy", None))

    monkeypatch.setattr(worker, "load_env_file", load_env_file)
    monkeypatch.setattr(
        worker,
        "configure_platform_dbos_runtime",
        configure_runtime,
    )
    monkeypatch.setattr(
        worker,
        "run_score_generation_workflow_once",
        run_once,
    )
    monkeypatch.setattr(worker, "destroy_dbos_runtime", destroy_runtime)

    worker.score_one(
        generation_run_id="generation-run-1",
        score_attempt_index=2,
        scoring_profile_id="profile",
        scoring_profile_version="v2",
        dataset_name="dataset",
        dataset_split="split",
        database_url="postgresql://app/db",
        dbos_system_database_url="postgresql://dbos/db",
        env_file=None,
    )

    assert calls[0] == ("load_env", None)
    assert calls[1][0] == "configure"
    assert calls[1][1][2] is False
    assert calls[1][1][3] == "for platform scoring workflow"
    assert calls[2] == (
        "score_once",
        {
            "database_url": "postgresql://example/db",
            "generation_run_id": "generation-run-1",
            "score_attempt_index": 2,
            "scoring_profile_id": "profile",
            "scoring_profile_version": "v2",
            "dataset_name": "dataset",
            "dataset_split": "split",
        },
    )
    assert calls[-1] == ("destroy", None)


def test_submit_jsonl_wires_submission_and_tears_down(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    graph = GraphSpec(nodes=(direct_node(),), terminal_node_id="direct")
    graph_spec = prediction_spec(graph, task_id="HumanEval/0")
    specs_file = tmp_path / "specs.jsonl"
    write_prediction_specs_jsonl(specs_file, (graph_spec,))

    class RuntimeConfig:
        database_url = "postgresql://example/db"

    class FakeEngine:
        disposed = False

        def dispose(self) -> None:
            self.disposed = True

    calls: list[tuple[str, Any]] = []
    captured_engine = FakeEngine()

    def configure_runtime(
        *,
        database_url: str | None,
        dbos_system_database_url: str | None,
        worker_concurrency: int = 1,
        consume_generation_queue: bool = False,
        database_url_error_suffix: str = "",
    ) -> RuntimeConfig:
        calls.append(
            (
                "configure",
                (
                    database_url,
                    dbos_system_database_url,
                    worker_concurrency,
                    consume_generation_queue,
                ),
            )
        )
        return RuntimeConfig()

    def register_queue(*, worker_concurrency: int) -> None:
        calls.append(("register", worker_concurrency))

    def submit_jsonl(engine: FakeEngine, **kwargs: Any) -> SimpleNamespace:
        calls.append(("submit", (engine, kwargs)))
        return SimpleNamespace(model_dump=lambda mode: {"ok": True})

    def create_engine(database_url: str) -> FakeEngine:
        calls.append(("create_engine", database_url))
        return captured_engine

    monkeypatch.setattr(worker, "load_env_file", lambda env_file=None: None)
    monkeypatch.setattr(
        worker,
        "configure_platform_dbos_runtime",
        configure_runtime,
    )
    monkeypatch.setattr(
        worker,
        "register_platform_generation_queue",
        register_queue,
    )
    monkeypatch.setattr(worker, "submit_prediction_specs_jsonl", submit_jsonl)
    monkeypatch.setattr(worker, "create_engine", create_engine)
    monkeypatch.setattr(
        worker,
        "destroy_dbos_runtime",
        lambda: calls.append(("destroy", None)),
    )

    worker.submit_jsonl(
        specs_file=specs_file,
        operation_key="op-1",
        experiment_name="exp",
        chunk_size=11,
        attempt_index=2,
        queue_registration_concurrency=3,
        database_url="postgresql://app/db",
        dbos_system_database_url="postgresql://dbos/db",
        env_file=None,
    )

    assert (
        "configure",
        ("postgresql://app/db", "postgresql://dbos/db", 3, False),
    ) in calls
    assert ("register", 3) in calls
    assert ("create_engine", "postgresql://example/db") in calls
    submit_call = next(call for call in calls if call[0] == "submit")
    assert submit_call[1][0] is captured_engine
    assert submit_call[1][1]["operation_key"] == "op-1"
    assert submit_call[1][1]["experiment_name"] == "exp"
    assert submit_call[1][1]["chunk_size"] == 11
    assert submit_call[1][1]["attempt_index"] == 2
    assert submit_call[1][1]["specs_file"] == specs_file
    assert captured_engine.disposed is True
    assert ("destroy", None) in calls


def test_worker_command_runs_until_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RuntimeConfig:
        database_url = "postgresql://example/db"

    calls: list[tuple[str, Any]] = []

    def configure_runtime(
        *,
        database_url: str | None,
        dbos_system_database_url: str | None,
        worker_concurrency: int = 1,
        consume_generation_queue: bool = False,
        database_url_error_suffix: str = "",
    ) -> RuntimeConfig:
        calls.append(
            (
                "configure",
                (
                    database_url,
                    dbos_system_database_url,
                    worker_concurrency,
                    consume_generation_queue,
                ),
            )
        )
        return RuntimeConfig()

    def sleep(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(worker, "load_env_file", lambda env_file=None: None)
    monkeypatch.setattr(
        worker,
        "configure_platform_dbos_runtime",
        configure_runtime,
    )
    monkeypatch.setattr(worker, "time", SimpleNamespace(sleep=sleep))
    monkeypatch.setattr(
        worker,
        "destroy_dbos_runtime",
        lambda: calls.append(("destroy", None)),
    )
    monkeypatch.setattr(
        worker.CONSOLE,
        "print",
        lambda value: calls.append(("print", value)),
    )

    worker.worker(
        worker_concurrency=4,
        database_url="postgresql://app/db",
        dbos_system_database_url=None,
        env_file=None,
    )

    assert (
        "configure",
        ("postgresql://app/db", None, 4, True),
    ) in calls
    assert any(
        call[0] == "print"
        and call[1]["queue_name"] == worker.PLATFORM_GENERATION_QUEUE_NAME
        for call in calls
    )
    assert ("destroy", None) in calls


def test_rescore_non_dry_run_launches_dbos_and_calls_rescore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class RuntimeConfig:
        database_url = "postgresql://example/db"

    class FakeEngine:
        def dispose(self) -> None:
            captured["disposed"] = True

    def configure_runtime(
        *,
        database_url: str | None,
        dbos_system_database_url: str | None,
        consume_generation_queue: bool = False,
        database_url_error_suffix: str = "",
        worker_concurrency: int = 1,
    ) -> RuntimeConfig:
        captured["configure"] = {
            "database_url": database_url,
            "consume_generation_queue": consume_generation_queue,
        }
        return RuntimeConfig()

    def fake_rescore(engine: FakeEngine, **kwargs: Any) -> SimpleNamespace:
        captured["engine"] = engine
        captured["kwargs"] = kwargs
        return SimpleNamespace(model_dump=lambda mode: {"scheduled": 1})

    monkeypatch.setattr(worker, "load_env_file", lambda env_file=None: None)
    monkeypatch.setattr(
        worker,
        "configure_platform_dbos_runtime",
        configure_runtime,
    )
    monkeypatch.setattr(
        worker,
        "create_engine",
        lambda database_url: FakeEngine(),
    )
    monkeypatch.setattr(worker, "rescore_generation_runs", fake_rescore)
    monkeypatch.setattr(
        worker,
        "destroy_dbos_runtime",
        lambda: captured.setdefault("destroyed", True),
    )

    result = CliRunner().invoke(
        worker.APP,
        [
            "rescore",
            "--database-url",
            "postgresql://app/db",
            "--experiment-name",
            "exp",
            "--generation-status",
            "success",
        ],
    )

    assert result.exit_code == 0
    assert captured["configure"]["consume_generation_queue"] is False
    assert captured["kwargs"]["dry_run"] is False
    assert captured["kwargs"]["experiment_name"] == "exp"
    assert captured["disposed"] is True
    assert captured["destroyed"] is True


def test_rescore_rejects_invalid_generation_status() -> None:
    result = CliRunner().invoke(
        worker.APP,
        [
            "rescore",
            "--experiment-name",
            "exp",
            "--generation-status",
            "not-a-status",
        ],
    )

    assert result.exit_code != 0
    assert "generation-status must be one of" in result.output
