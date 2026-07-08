from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from dbos._error import (
    DBOSConflictingWorkflowError,
    DBOSQueueDeduplicatedError,
    DBOSWorkflowConflictIDError,
)
from dr_graph import (
    BindingRef,
    FieldRole,
    FieldSpec,
    GraphSpec,
    NodeConfig,
    NodeSpec,
    graph_digest,
)
from dr_platform import OperationProgress
from dr_providers import EndpointKind, ProviderKind
from typer.testing import CliRunner

from whetstone.platform import (
    graph_workflow,
    queue_worker,
    worker,
)
from whetstone.records import (
    DimensionsPayload,
    GenerationRunStatus,
    GraphSnapshotPayload,
    PredictionSpecRecord,
    ProviderConfigRef,
    TaskInputsPayload,
    TaskSnapshotPayload,
    dimensions_digest,
    fair_order_key,
    stable_generation_run_id,
    stable_prediction_id,
)

NOW = datetime(2026, 6, 29, 12, 0, tzinfo=UTC)


class DummyConnection:
    def __init__(self) -> None:
        self.statements: list[Any] = []

    def execute(self, statement: Any) -> Any:
        self.statements.append(statement)
        return ExecuteResult()


class ExecuteResult:
    rowcount = 1

    def mappings(self) -> ExecuteResult:
        return self

    def one_or_none(self) -> None:
        return None

    def scalar_one(self) -> int:
        return 1


class DummyTransaction:
    def __init__(self, engine: DummyEngine) -> None:
        self.engine = engine

    def __enter__(self) -> DummyConnection:
        self.engine.in_transaction = True
        self.engine.begin_count += 1
        return self.engine.connection

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        self.engine.in_transaction = False


class DummyEngine:
    def __init__(self) -> None:
        self.connection = DummyConnection()
        self.begin_count = 0
        self.in_transaction = False

    def begin(self) -> DummyTransaction:
        return DummyTransaction(self)


def _node() -> NodeSpec:
    return NodeSpec(
        id="direct",
        op="llm_call",
        config=NodeConfig(
            fields=(
                FieldSpec(name="prompt", role=FieldRole.INPUT),
                FieldSpec(name="output", role=FieldRole.OUTPUT),
            ),
            input_bindings={
                "prompt": BindingRef.model_validate("task.prompt")
            },
            output_field="output",
        ),
    )


def _spec(
    *,
    task_id: str,
    model: str = "gpt-test",
    temperature: float = 0.2,
) -> PredictionSpecRecord:
    graph = GraphSpec(nodes=(_node(),), terminal_node_id="direct")
    graph_id = graph_digest(graph)
    dimensions = DimensionsPayload(values={"temperature": temperature})
    dimensions_id = dimensions_digest(dimensions)
    provider = ProviderConfigRef(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.RESPONSES,
        model=model,
        throttle_key=f"openai:responses:{model}",
    )
    prediction_id = stable_prediction_id(
        experiment_name="exp",
        task_id=task_id,
        graph_digest=graph_id,
        dimensions_digest=dimensions_id,
        repetition_seed=0,
        provider_kind=provider.provider_kind.value,
        endpoint_kind=provider.endpoint_kind.value,
        model=provider.model,
        throttle_key=provider.throttle_key,
    )
    return PredictionSpecRecord(
        prediction_id=prediction_id,
        experiment_name="exp",
        task_id=task_id,
        repetition_seed=0,
        graph=GraphSnapshotPayload(
            graph=graph,
            graph_digest=graph_id,
            layout="direct",
        ),
        dimensions=dimensions,
        dimensions_digest=dimensions_id,
        task=TaskSnapshotPayload(
            task_id=task_id,
            inputs=TaskInputsPayload(values={"prompt": "write add"}),
        ),
        provider_configs=(provider,),
        provider_axis=provider,
        fair_order_seed="seed",
        fair_order_key=fair_order_key(
            experiment_seed="seed",
            prediction_id=prediction_id,
            provider=provider.provider_kind.value,
            endpoint_kind=provider.endpoint_kind.value,
            model=provider.model,
            throttle_key=provider.throttle_key,
            graph_layout="direct",
            task_id=task_id,
            repetition_seed=0,
            config_axis=dimensions_id,
        ),
        created_at=NOW,
    )




















def test_queue_enqueue_uses_stable_workflow_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        queue_worker.DBOS,
        "get_workflow_status",
        lambda workflow_id: None,
    )
    monkeypatch.setattr(
        queue_worker.DBOS,
        "enqueue_workflow",
        lambda queue_name, workflow, *args: captured.update(
            {"queue_name": queue_name, "args": args}
        ),
    )
    prediction_id = "prediction-1"

    result = queue_worker.enqueue_prediction_graph_workflow(
        database_url="postgresql://example/db",
        prediction_id=prediction_id,
        attempt_index=2,
    )

    generation_run_id = stable_generation_run_id(
        prediction_id=prediction_id,
        attempt_index=2,
    )
    assert result.enqueued is True
    assert result.generation_run_id == generation_run_id
    assert result.workflow_id == (
        f"platform-generate-v1:{generation_run_id}"
    )
    assert captured["queue_name"] == (
        queue_worker.PLATFORM_GENERATION_QUEUE_NAME
    )
    assert captured["args"] == (
        "postgresql://example/db",
        prediction_id,
        2,
    )


def test_queue_enqueue_skips_when_workflow_status_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueue_calls: list[Any] = []

    monkeypatch.setattr(
        queue_worker.DBOS,
        "get_workflow_status",
        lambda workflow_id: {"status": "PENDING"},
    )
    monkeypatch.setattr(
        queue_worker.DBOS,
        "enqueue_workflow",
        lambda *args: enqueue_calls.append(args),
    )
    prediction_id = "prediction-existing"

    result = queue_worker.enqueue_prediction_graph_workflow(
        database_url="postgresql://example/db",
        prediction_id=prediction_id,
    )

    generation_run_id = stable_generation_run_id(
        prediction_id=prediction_id,
        attempt_index=0,
    )
    assert result.enqueued is False
    assert result.generation_run_id == generation_run_id
    assert result.workflow_id == (
        f"platform-generate-v1:{generation_run_id}"
    )
    assert enqueue_calls == []


@pytest.mark.parametrize(
    "error",
    [
        DBOSWorkflowConflictIDError("platform-generate-v1:run-1"),
        DBOSQueueDeduplicatedError(
            "platform-generate-v1:run-1",
            "dr-dspy-platform-generation-v1",
            "dedup-1",
        ),
        DBOSConflictingWorkflowError("platform-generate-v1:run-1"),
    ],
)
def test_queue_enqueue_treats_start_race_as_existing(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    monkeypatch.setattr(
        queue_worker.DBOS,
        "get_workflow_status",
        lambda workflow_id: None,
    )

    def raise_race(*args: Any) -> None:
        raise error

    monkeypatch.setattr(queue_worker.DBOS, "enqueue_workflow", raise_race)
    prediction_id = "prediction-race"

    result = queue_worker.enqueue_prediction_graph_workflow(
        database_url="postgresql://example/db",
        prediction_id=prediction_id,
    )

    assert result.enqueued is False
    assert result.prediction_id == prediction_id


def test_enqueue_prediction_graph_workflows_aggregates_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing_status = {"status": "PENDING"}
    race_error = DBOSWorkflowConflictIDError("platform-generate-v1:run-race")
    existing_workflow_id = graph_workflow.platform_generation_workflow_id(
        stable_generation_run_id(
            prediction_id="prediction-existing",
            attempt_index=0,
        )
    )

    def status_for(workflow_id: str) -> dict[str, str] | None:
        if workflow_id == existing_workflow_id:
            return existing_status
        return None

    def enqueue_for(
        queue_name: str,
        workflow: Any,
        database_url: str,
        prediction_id: str,
        attempt_index: int,
    ) -> None:
        if prediction_id == "prediction-race":
            raise race_error

    monkeypatch.setattr(
        queue_worker.DBOS,
        "get_workflow_status",
        status_for,
    )
    monkeypatch.setattr(
        queue_worker.DBOS,
        "enqueue_workflow",
        enqueue_for,
    )

    result = queue_worker.enqueue_prediction_graph_workflows(
        database_url="postgresql://example/db",
        prediction_ids=(
            "prediction-new",
            "prediction-existing",
            "prediction-race",
        ),
    )

    assert result.enqueued_count == 1
    assert result.existing_count == 2
    assert [item.prediction_id for item in result.workflows] == [
        "prediction-new",
        "prediction-existing",
        "prediction-race",
    ]
    assert result.workflows[0].enqueued is True
    assert result.workflows[1].enqueued is False
    assert result.workflows[2].enqueued is False


def test_listen_to_platform_generation_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    monkeypatch.setattr(
        queue_worker.DBOS,
        "listen_queues",
        lambda queues: captured.append(list(queues)),
    )

    queue_worker.listen_to_platform_generation_queue()

    assert captured == [[queue_worker.PLATFORM_GENERATION_QUEUE_NAME]]


def test_queue_enqueue_surfaces_unrelated_start_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    starts: list[Any] = []

    monkeypatch.setattr(
        queue_worker.DBOS,
        "get_workflow_status",
        lambda workflow_id: None if not starts else {"status": "PENDING"},
    )

    def fail_enqueue(*args: Any) -> None:
        starts.append(args)
        raise RuntimeError("dbos unavailable")

    monkeypatch.setattr(
        queue_worker.DBOS,
        "enqueue_workflow",
        fail_enqueue,
    )

    with pytest.raises(RuntimeError, match="dbos unavailable"):
        queue_worker.enqueue_prediction_graph_workflow(
            database_url="postgresql://example/db",
            prediction_id="prediction-1",
        )


def test_register_platform_generation_queue_updates_existing_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def register_queue(
        queue_name: str,
        *,
        worker_concurrency: int,
        on_conflict: str,
    ) -> object:
        captured.update(
            {
                "queue_name": queue_name,
                "worker_concurrency": worker_concurrency,
                "on_conflict": on_conflict,
            }
        )
        return object()

    monkeypatch.setattr(queue_worker.DBOS, "register_queue", register_queue)

    queue_worker.register_platform_generation_queue(worker_concurrency=4)

    assert captured == {
        "queue_name": queue_worker.PLATFORM_GENERATION_QUEUE_NAME,
        "worker_concurrency": 4,
        "on_conflict": "always_update",
    }


def test_platform_worker_config_listens_to_v1_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    config = SimpleNamespace(database_url="postgresql://example/db")

    class FakeDbos:
        def __call__(self, *, config: dict[str, Any]) -> None:
            calls.append(("configure", config))

        def listen_queues(self, queues: list[str]) -> None:
            calls.append(("listen", queues))

        def launch(self) -> None:
            calls.append(("launch", None))

    monkeypatch.setattr(worker, "DBOS", FakeDbos())
    monkeypatch.setattr(
        worker,
        "build_eval_dbos_config",
        lambda **kwargs: config,
    )
    monkeypatch.setattr(
        worker,
        "build_dbos_config",
        lambda config, app_name: {"name": app_name},
    )
    monkeypatch.setattr(
        worker,
        "listen_to_platform_generation_queue",
        lambda: calls.append(("listen_v1", None)),
    )
    monkeypatch.setattr(
        worker,
        "register_platform_generation_queue",
        lambda worker_concurrency: calls.append(
            ("register", worker_concurrency)
        ),
    )

    worker.configure_platform_dbos_runtime(
        database_url=None,
        dbos_system_database_url=None,
        worker_concurrency=3,
        consume_generation_queue=True,
    )

    assert ("listen_v1", None) in calls
    assert ("register", 3) in calls


@pytest.mark.parametrize(
    ("failure_stage", "consume_generation_queue"),
    [
        ("launch", True),
        ("listen_queues", False),
        ("register", True),
    ],
)
def test_configure_platform_dbos_runtime_cleans_up_on_launch_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    consume_generation_queue: bool,
) -> None:
    destroy_calls: list[str] = []
    config = SimpleNamespace(database_url="postgresql://example/db")

    class FakeDbos:
        def __call__(self, *, config: dict[str, Any]) -> None:
            return None

        def listen_queues(self, queues: list[str]) -> None:
            if failure_stage == "listen_queues":
                raise RuntimeError("listen failed")

        def launch(self) -> None:
            if failure_stage == "launch":
                raise RuntimeError("launch failed")

    def fail_register(*, worker_concurrency: int) -> None:
        if failure_stage == "register":
            raise RuntimeError("register failed")

    monkeypatch.setattr(worker, "DBOS", FakeDbos())
    monkeypatch.setattr(
        worker,
        "build_eval_dbos_config",
        lambda **kwargs: config,
    )
    monkeypatch.setattr(
        worker,
        "build_dbos_config",
        lambda config, app_name: {"name": app_name},
    )
    monkeypatch.setattr(
        worker,
        "listen_to_platform_generation_queue",
        lambda: None,
    )
    monkeypatch.setattr(
        worker,
        "register_platform_generation_queue",
        fail_register,
    )
    monkeypatch.setattr(
        worker,
        "destroy_dbos_runtime",
        lambda: destroy_calls.append("destroy"),
    )

    with pytest.raises(RuntimeError):
        worker.configure_platform_dbos_runtime(
            database_url=None,
            dbos_system_database_url=None,
            worker_concurrency=3,
            consume_generation_queue=consume_generation_queue,
        )

    assert destroy_calls == ["destroy"]


def test_submit_jsonl_help_describes_queue_registration_concurrency() -> None:
    result = CliRunner().invoke(
        worker.APP,
        ["submit-jsonl", "--help"],
        terminal_width=160,
    )

    assert result.exit_code == 0
    assert "does not start a queue" in result.output
    assert "worker." in result.output


def test_rescore_cli_dry_run_wires_options_without_launching_dbos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeEngine:
        def dispose(self) -> None:
            captured["disposed"] = True

    def fake_rescore(engine: FakeEngine, **kwargs: Any) -> Any:
        captured["engine"] = engine
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            result=SimpleNamespace(
                model_dump=lambda mode: {"dry_run": kwargs["dry_run"]}
            ),
            workflow_handles=(),
        )

    def fail_configure_platform_dbos_runtime(**kwargs: Any) -> Any:
        raise AssertionError("dry-run should not launch DBOS")

    def fail_build_eval_dbos_config(**kwargs: Any) -> Any:
        raise AssertionError("dry-run should not resolve DBOS config")

    monkeypatch.setattr(
        worker,
        "build_eval_dbos_config",
        fail_build_eval_dbos_config,
    )
    monkeypatch.setattr(
        worker,
        "configure_platform_dbos_runtime",
        fail_configure_platform_dbos_runtime,
    )
    monkeypatch.setattr(
        worker,
        "create_engine",
        lambda database_url: FakeEngine(),
    )
    monkeypatch.setattr(worker, "rescore_generation_runs", fake_rescore)

    result = CliRunner().invoke(
        worker.APP,
        [
            "rescore",
            "--database-url",
            "postgresql://example/db",
            "--experiment-name",
            "exp",
            "--generation-status",
            "success",
            "--generation-attempt-index",
            "0",
            "--score-attempt-index",
            "1",
            "--scoring-profile-id",
            "humaneval",
            "--scoring-profile-version",
            "v1",
            "--dataset-name",
            "dataset",
            "--dataset-split",
            "split",
            "--chunk-size",
            "7",
            "--max-in-flight",
            "10",
            "--limit",
            "9",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert captured["disposed"] is True
    captured_kwargs = dict(captured["kwargs"])
    progress = captured_kwargs.pop("progress")
    assert isinstance(progress, OperationProgress)
    assert captured_kwargs == {
        "database_url": "postgresql+psycopg://example/db",
        "experiment_name": "exp",
        "generation_statuses": (GenerationRunStatus.SUCCESS,),
        "generation_attempt_index": 0,
        "scoring_profile_id": "humaneval",
        "scoring_profile_version": "v1",
        "score_attempt_index": 1,
        "dataset_name": "dataset",
        "dataset_split": "split",
        "chunk_size": 7,
        "max_in_flight": 10,
        "limit": 9,
        "dry_run": True,
        "recover_orphans": True,
    }
    assert "{'dry_run': True}" in result.output


def test_run_one_runtime_keeps_empty_queue_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    config = SimpleNamespace(database_url="postgresql://example/db")

    class FakeDbos:
        def __call__(self, *, config: dict[str, Any]) -> None:
            calls.append(("configure", config))

        def listen_queues(self, queues: list[str]) -> None:
            calls.append(("listen", queues))

        def launch(self) -> None:
            calls.append(("launch", None))

    monkeypatch.setattr(worker, "DBOS", FakeDbos())
    monkeypatch.setattr(
        worker,
        "build_eval_dbos_config",
        lambda **kwargs: config,
    )
    monkeypatch.setattr(
        worker,
        "build_dbos_config",
        lambda config, app_name: {"name": app_name},
    )

    worker.configure_platform_dbos_runtime(
        database_url=None,
        dbos_system_database_url=None,
        consume_generation_queue=False,
    )

    assert ("listen", []) in calls





