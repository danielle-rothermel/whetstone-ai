from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from sqlalchemy.dialects import postgresql

from dr_dspy.db import io as db_io
from dr_dspy.eval_failures import (
    FailureClass,
    PermanentFailureError,
    TransientFailureError,
)
from dr_dspy.graph import (
    BindingRef,
    FieldRole,
    FieldSpec,
    GraphRunStatus,
    GraphSpec,
    NodeConfig,
    NodeOutput,
    NodeSpec,
    graph_digest,
)
from dr_dspy.lm.boundary import (
    EndpointKind,
    ProviderConfig,
    ProviderKind,
)
from dr_dspy.platform import graph_workflow
from dr_dspy.platform.graph_workflow import execute_prediction_graph
from dr_dspy.platform.node_execution import (
    NodeStepResult,
    NodeStepStatus,
    execute_lm_node,
    provider_config_ref_for_node,
)
from dr_dspy.platform.persistence import (
    idempotent_insert_generation_run,
    idempotent_insert_node_attempt,
    persist_generation_result,
    prediction_spec_from_row,
)
from dr_dspy.records import (
    DimensionsPayload,
    FailureMetadataPayload,
    GenerationRunStatus,
    GraphSnapshotPayload,
    PredictionSpecRecord,
    ProviderConfigRef,
    TaskInputsPayload,
    TaskSnapshotPayload,
    dimensions_digest,
    fair_order_key,
    stable_generation_run_id,
    stable_node_attempt_id,
    stable_prediction_id,
)

NOW = datetime(2026, 6, 29, 12, 0, tzinfo=UTC)
LATER = NOW + timedelta(seconds=5)


def _node(
    node_id: str,
    *,
    bindings: dict[str, str] | None = None,
    output_field: str = "output",
    user_prompt_template: str = "{prompt}",
    system_prompt: str | None = None,
    provider_config_id: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> NodeSpec:
    input_bindings = {
        name: BindingRef.model_validate(ref)
        for name, ref in (bindings or {}).items()
    }
    fields = [
        FieldSpec(name=name, role=FieldRole.INPUT)
        for name in input_bindings
    ]
    fields.append(FieldSpec(name=output_field, role=FieldRole.OUTPUT))
    metadata: dict[str, Any] = {"user_prompt_template": user_prompt_template}
    if system_prompt is not None:
        metadata["system_prompt"] = system_prompt
    if provider_config_id is not None:
        metadata["provider_config_id"] = provider_config_id
    return NodeSpec(
        id=node_id,
        config=NodeConfig(
            fields=tuple(fields),
            input_bindings=input_bindings,
            output_field=output_field,
            parameters=parameters or {},
            metadata=metadata,
        ),
    )


def _provider(
    *,
    config_id: str | None = "main",
    model: str = "gpt-test",
) -> ProviderConfigRef:
    return ProviderConfigRef(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.RESPONSES,
        model=model,
        config_id=config_id,
        throttle_key=f"openai:responses:{model}",
        parameters={"temperature": 0.2},
    )


def _spec(
    graph: GraphSpec,
    *,
    providers: tuple[ProviderConfigRef, ...] | None = None,
    provider_axis: ProviderConfigRef | None = None,
    layout: str = "direct",
    task_inputs: dict[str, Any] | None = None,
) -> PredictionSpecRecord:
    providers = providers or (_provider(),)
    provider_axis = provider_axis or providers[0]
    graph_id = graph_digest(graph)
    dimensions = DimensionsPayload(values={"temperature": 0.2})
    dimensions_id = dimensions_digest(dimensions)
    prediction_id = stable_prediction_id(
        experiment_name="exp",
        task_id="HumanEval/0",
        graph_digest=graph_id,
        dimensions_digest=dimensions_id,
        repetition_seed=0,
        provider_kind=provider_axis.provider_kind.value,
        endpoint_kind=provider_axis.endpoint_kind.value,
        model=provider_axis.model,
        throttle_key=provider_axis.throttle_key,
    )
    return PredictionSpecRecord(
        prediction_id=prediction_id,
        experiment_name="exp",
        task_id="HumanEval/0",
        repetition_seed=0,
        graph=GraphSnapshotPayload(
            graph=graph,
            graph_digest=graph_id,
            layout=layout,
        ),
        dimensions=dimensions,
        dimensions_digest=dimensions_id,
        task=TaskSnapshotPayload(
            task_id="HumanEval/0",
            inputs=TaskInputsPayload(
                values=task_inputs or {"prompt": "write add"}
            ),
        ),
        provider_configs=providers,
        provider_axis=provider_axis,
        fair_order_seed="seed",
        fair_order_key=fair_order_key(
            experiment_seed="seed",
            prediction_id=prediction_id,
            provider=provider_axis.provider_kind.value,
            endpoint_kind=provider_axis.endpoint_kind.value,
            model=provider_axis.model,
            throttle_key=provider_axis.throttle_key,
            graph_layout=layout,
            task_id="HumanEval/0",
            repetition_seed=0,
            config_axis=dimensions_id,
        ),
        created_at=NOW,
    )


def _step_success(
    node: NodeSpec,
    value: str,
    *,
    provider: ProviderConfigRef | None = None,
) -> NodeStepResult:
    return NodeStepResult.success(
        node_id=node.id,
        provider_config=provider or _provider(),
        output=NodeOutput(values={node.config.output_field: value}),
        usage_metadata={"total_tokens": 3},
        provider_cost=0.01,
        response_metadata={"id": f"response-{node.id}"},
        started_at=NOW,
        completed_at=NOW,
    )


def _step_error(node: NodeSpec, message: str) -> NodeStepResult:
    return NodeStepResult(
        node_id=node.id,
        status=NodeStepStatus.ERROR,
        provider_config=_provider(),
        failure=FailureMetadataPayload(
            failure_class=FailureClass.PERMANENT,
            error_type="builtins.RuntimeError",
            message=message,
            metadata={"node_id": node.id},
        ),
        started_at=NOW,
        completed_at=NOW,
    )


def test_direct_graph_success_persists_generation_and_node_attempt() -> None:
    graph = GraphSpec(
        nodes=(_node("direct", bindings={"prompt": "task.prompt"}),),
        terminal_node_id="direct",
    )
    spec = _spec(graph)
    generation_run_id = stable_generation_run_id(
        prediction_id=spec.prediction_id,
        attempt_index=0,
    )

    execution = execute_prediction_graph(
        spec=spec,
        attempt_index=0,
        generation_run_id=generation_run_id,
        started_at=NOW,
        completed_at=LATER,
        run_node_step=lambda step_spec, node, inputs: _step_success(
            node,
            f"code for {inputs['prompt']}",
        ),
    )

    assert execution.generation_run.status is GenerationRunStatus.SUCCESS
    assert execution.generation_run.summary.terminal_output == (
        "code for write add"
    )
    assert len(execution.node_attempts) == 1
    attempt = execution.node_attempts[0]
    assert attempt.node_id == "direct"
    assert attempt.output is not None
    assert attempt.output.values == {"output": "code for write add"}
    assert attempt.usage_cost.provider_cost == 0.01
    assert attempt.response_metadata.response_metadata == {
        "id": "response-direct"
    }


def test_encdec_graph_success_persists_encoder_and_decoder_attempts() -> None:
    encoder = _node(
        "encoder",
        bindings={"prompt": "task.prompt"},
        output_field="description",
        user_prompt_template="Describe {prompt}",
    )
    decoder = _node(
        "decoder",
        bindings={"description": "encoder.description"},
        output_field="code",
        user_prompt_template="Write code from {description}",
    )
    graph = GraphSpec(nodes=(decoder, encoder), terminal_node_id="decoder")
    spec = _spec(graph, layout="encdec")

    def run_node_step(
        step_spec: PredictionSpecRecord,
        node: NodeSpec,
        inputs: Mapping[str, Any],
    ) -> NodeStepResult:
        if node.id == "encoder":
            return _step_success(node, "plain description")
        return _step_success(
            node,
            f"def f(): return {inputs['description']!r}",
        )

    execution = execute_prediction_graph(
        spec=spec,
        attempt_index=0,
        generation_run_id="run-1",
        started_at=NOW,
        completed_at=LATER,
        run_node_step=run_node_step,
    )

    assert execution.generation_run.status is GenerationRunStatus.SUCCESS
    assert [attempt.node_id for attempt in execution.node_attempts] == [
        "encoder",
        "decoder",
    ]
    assert execution.generation_run.summary.terminal_output == (
        "def f(): return 'plain description'"
    )


def test_failed_upstream_node_blocks_downstream_without_node_attempt() -> None:
    encoder = _node("encoder", bindings={"prompt": "task.prompt"})
    decoder = _node("decoder", bindings={"text": "encoder"})
    graph = GraphSpec(nodes=(decoder, encoder), terminal_node_id="decoder")
    spec = _spec(graph, layout="encdec")

    execution = execute_prediction_graph(
        spec=spec,
        attempt_index=0,
        generation_run_id="run-1",
        started_at=NOW,
        completed_at=LATER,
        run_node_step=lambda step_spec, node, inputs: _step_error(
            node,
            "provider failed",
        ),
    )

    assert execution.graph_result.status is GraphRunStatus.BLOCKED
    assert execution.generation_run.status is GenerationRunStatus.BLOCKED
    assert [attempt.node_id for attempt in execution.node_attempts] == [
        "encoder"
    ]
    assert execution.node_attempts[0].failure is not None
    assert execution.generation_run.summary.terminal_error is not None
    assert execution.generation_run.summary.terminal_error.blocked_by == (
        "encoder",
    )


def test_terminal_error_summary_preserves_node_failure_class() -> None:
    graph = GraphSpec(
        nodes=(_node("direct", bindings={"prompt": "task.prompt"}),),
        terminal_node_id="direct",
    )
    spec = _spec(graph)

    execution = execute_prediction_graph(
        spec=spec,
        attempt_index=0,
        generation_run_id="run-1",
        started_at=NOW,
        completed_at=LATER,
        run_node_step=lambda step_spec, node, inputs: _step_error(
            node,
            "provider failed",
        ),
    )

    node_failure = execution.node_attempts[0].failure
    terminal_error = execution.generation_run.summary.terminal_error
    assert node_failure is not None
    assert terminal_error is not None
    assert terminal_error.failure is not None
    assert node_failure.failure_class is FailureClass.PERMANENT
    assert terminal_error.failure.failure_class is FailureClass.PERMANENT


def test_run_prediction_graph_workflow_uses_dbos_step_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = GraphSpec(
        nodes=(_node("direct", bindings={"prompt": "task.prompt"}),),
        terminal_node_id="direct",
    )
    spec = _spec(graph)
    database_url = "postgresql://example/db"
    attempt_index = 2
    generation_run_id = stable_generation_run_id(
        prediction_id=spec.prediction_id,
        attempt_index=attempt_index,
    )
    calls: list[tuple[str, Any]] = []

    def load_step(
        step_database_url: str,
        prediction_id: str,
    ) -> dict[str, Any]:
        calls.append(("load", (step_database_url, prediction_id)))
        return spec.model_dump(mode="json")

    def started_step(step_generation_run_id: str) -> str:
        calls.append(("started", step_generation_run_id))
        return NOW.isoformat()

    def execute_step(
        spec_payload: dict[str, Any],
        node_payload: dict[str, Any],
        node_inputs: dict[str, Any],
    ) -> dict[str, Any]:
        step_spec = PredictionSpecRecord.model_validate(spec_payload)
        node = NodeSpec.model_validate(node_payload)
        calls.append(
            (
                "execute",
                (step_spec.prediction_id, node.id, node_inputs),
            )
        )
        return _step_success(
            node,
            f"workflow {node_inputs['prompt']}",
        ).model_dump(mode="json")

    def completed_step(step_generation_run_id: str) -> str:
        calls.append(("completed", step_generation_run_id))
        return LATER.isoformat()

    def persist_step(
        step_database_url: str,
        spec_payload: dict[str, Any],
        step_generation_run_id: str,
        step_attempt_index: int,
        graph_result_payload: dict[str, Any],
        node_step_result_payloads: list[dict[str, Any]],
        started_at: str,
        completed_at: str,
    ) -> None:
        persisted_spec = PredictionSpecRecord.model_validate(spec_payload)
        graph_result = graph_workflow.GraphRunResult.model_validate(
            graph_result_payload
        )
        calls.append(
            (
                "persist",
                (
                    step_database_url,
                    persisted_spec.prediction_id,
                    step_generation_run_id,
                    step_attempt_index,
                    graph_result.status,
                    len(node_step_result_payloads),
                    started_at,
                    completed_at,
                ),
            )
        )

    monkeypatch.setattr(
        graph_workflow,
        "load_prediction_spec_step",
        load_step,
    )
    monkeypatch.setattr(
        graph_workflow,
        "generation_started_at_step",
        started_step,
    )
    monkeypatch.setattr(graph_workflow, "execute_lm_node_step", execute_step)
    monkeypatch.setattr(
        graph_workflow,
        "generation_completed_at_step",
        completed_step,
    )
    monkeypatch.setattr(
        graph_workflow,
        "persist_generation_result_step",
        persist_step,
    )

    workflow = cast(
        Any,
        graph_workflow.run_prediction_graph_workflow,
    ).__wrapped__
    result = workflow(database_url, spec.prediction_id, attempt_index)

    assert result == generation_run_id
    assert calls == [
        ("load", (database_url, spec.prediction_id)),
        ("started", generation_run_id),
        (
            "execute",
            (
                spec.prediction_id,
                "direct",
                {"prompt": "write add"},
            ),
        ),
        ("completed", generation_run_id),
        (
            "persist",
            (
                database_url,
                spec.prediction_id,
                generation_run_id,
                attempt_index,
                GraphRunStatus.SUCCESS,
                1,
                NOW.isoformat(),
                LATER.isoformat(),
            ),
        ),
    ]


def test_independent_branch_failure_records_partial_generation() -> None:
    graph = GraphSpec(
        nodes=(_node("terminal"), _node("bad")),
        terminal_node_id="terminal",
    )
    spec = _spec(graph, task_inputs={})

    def run_node_step(
        step_spec: PredictionSpecRecord,
        node: NodeSpec,
        inputs: Mapping[str, Any],
    ) -> NodeStepResult:
        if node.id == "bad":
            return _step_error(node, "boom")
        return _step_success(node, "ok")

    execution = execute_prediction_graph(
        spec=spec,
        attempt_index=0,
        generation_run_id="run-1",
        started_at=NOW,
        completed_at=LATER,
        run_node_step=run_node_step,
    )

    assert execution.graph_result.status is GraphRunStatus.PARTIAL
    assert execution.generation_run.status is GenerationRunStatus.PARTIAL
    assert {attempt.node_id for attempt in execution.node_attempts} == {
        "bad",
        "terminal",
    }


def test_lm_node_executor_sends_exact_messages_and_metadata() -> None:
    node = _node(
        "direct",
        bindings={"prompt": "task.prompt"},
        output_field="code",
        user_prompt_template="Solve: {prompt}",
        system_prompt="Write Python.",
        parameters={"token_limit": 20},
    )
    graph = GraphSpec(nodes=(node,), terminal_node_id="direct")
    spec = _spec(graph)
    captured: dict[str, Any] = {}

    def client_factory(config: ProviderConfig) -> object:
        captured["config"] = config
        return object()

    def provider_caller(client: Any, request: Any) -> dict[str, Any]:
        captured["request"] = request
        return {
            "id": "resp-1",
            "model": "gpt-test",
            "status": "completed",
            "output_text": "def add(a, b): return a + b",
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
                "cost": 0.02,
            },
        }

    result = execute_lm_node(
        spec=spec,
        node=node,
        node_inputs={"prompt": "write add"},
        client_factory=client_factory,
        provider_caller=provider_caller,
    )

    assert result.status is NodeStepStatus.SUCCESS
    request = captured["request"]
    assert request.kwargs == {
        "model": "gpt-test",
        "instructions": "Write Python.",
        "input": [{"role": "user", "content": "Solve: write add"}],
        "temperature": 0.2,
        "max_output_tokens": 20,
    }
    assert result.output is not None
    assert result.output.values == {"code": "def add(a, b): return a + b"}
    assert result.usage_cost.usage_metadata["total_tokens"] == 3
    assert result.usage_cost.provider_cost == 0.02
    assert result.response_metadata.response_metadata["id"] == "resp-1"


def test_lm_node_executor_reraises_retryable_failures_for_dbos_retry() -> None:
    node = _node("direct", bindings={"prompt": "task.prompt"})
    graph = GraphSpec(nodes=(node,), terminal_node_id="direct")
    spec = _spec(graph)

    def provider_caller(client: Any, request: Any) -> None:
        raise TransientFailureError("temporary provider failure")

    with pytest.raises(TransientFailureError):
        execute_lm_node(
            spec=spec,
            node=node,
            node_inputs={"prompt": "write add"},
            client_factory=lambda config: object(),
            provider_caller=provider_caller,
            raise_retryable=True,
        )


def test_lm_node_executor_rejects_unsupported_node_op() -> None:
    node = _node("direct", bindings={"prompt": "task.prompt"})
    unsupported_node = NodeSpec.model_construct(
        id=node.id,
        config=node.config,
        op=cast(Any, "python_call"),
    )
    graph = GraphSpec(nodes=(node,), terminal_node_id="direct")
    spec = _spec(graph)

    def provider_caller(client: Any, request: Any) -> Any:
        raise AssertionError("unsupported node op should not call provider")

    result = execute_lm_node(
        spec=spec,
        node=unsupported_node,
        node_inputs={"prompt": "write add"},
        client_factory=lambda config: object(),
        provider_caller=provider_caller,
    )

    assert result.status is NodeStepStatus.ERROR
    assert result.failure is not None
    assert result.failure.failure_class is FailureClass.PERMANENT
    assert result.failure.message == (
        "unsupported node operation for LM executor"
    )
    assert result.failure.metadata == {
        "node_id": "direct",
        "node_op": "python_call",
    }


def test_multiple_provider_configs_require_node_provider_config_id() -> None:
    node = _node("direct", bindings={"prompt": "task.prompt"})
    graph = GraphSpec(nodes=(node,), terminal_node_id="direct")
    spec = _spec(
        graph,
        providers=(
            _provider(config_id="encoder", model="encoder-model"),
            _provider(config_id="decoder", model="decoder-model"),
        ),
    )

    with pytest.raises(PermanentFailureError, match="provider_config_id"):
        provider_config_ref_for_node(spec=spec, node=node)


def test_deterministic_generation_and_node_attempt_ids() -> None:
    generation_run_id = stable_generation_run_id(
        prediction_id="prediction-1",
        attempt_index=2,
    )

    assert generation_run_id == stable_generation_run_id(
        prediction_id="prediction-1",
        attempt_index=2,
    )
    assert generation_run_id != stable_generation_run_id(
        prediction_id="prediction-1",
        attempt_index=3,
    )
    assert stable_node_attempt_id(
        generation_run_id=generation_run_id,
        node_id="decoder",
        attempt_index=0,
    ) == stable_node_attempt_id(
        generation_run_id=generation_run_id,
        node_id="decoder",
        attempt_index=0,
    )


def test_prediction_spec_from_row_round_trips_db_io_shape() -> None:
    graph = GraphSpec(
        nodes=(_node("direct", bindings={"prompt": "task.prompt"}),),
        terminal_node_id="direct",
    )
    spec = _spec(graph)

    parsed = prediction_spec_from_row(db_io.prediction_spec_row(spec))

    assert parsed.model_dump(mode="json") == spec.model_dump(mode="json")


def test_persist_generation_result_uses_idempotent_inserts() -> None:
    graph = GraphSpec(
        nodes=(_node("direct", bindings={"prompt": "task.prompt"}),),
        terminal_node_id="direct",
    )
    spec = _spec(graph)
    generation_run_id = stable_generation_run_id(
        prediction_id=spec.prediction_id,
        attempt_index=0,
    )
    execution = execute_prediction_graph(
        spec=spec,
        attempt_index=0,
        generation_run_id=generation_run_id,
        started_at=NOW,
        completed_at=LATER,
        run_node_step=lambda step_spec, node, inputs: _step_success(
            node,
            "ok",
        ),
    )

    generation_sql = _postgres_sql(
        idempotent_insert_generation_run(execution.generation_run)
    )
    node_sql = _postgres_sql(
        idempotent_insert_node_attempt(execution.node_attempts[0])
    )

    assert "ON CONFLICT (generation_run_id) DO NOTHING" in generation_sql
    assert "ON CONFLICT (node_attempt_id) DO NOTHING" in node_sql


def test_persist_generation_result_executes_idempotent_statements() -> None:
    graph = GraphSpec(
        nodes=(_node("direct", bindings={"prompt": "task.prompt"}),),
        terminal_node_id="direct",
    )
    spec = _spec(graph)
    execution = execute_prediction_graph(
        spec=spec,
        attempt_index=0,
        generation_run_id="run-1",
        started_at=NOW,
        completed_at=LATER,
        run_node_step=lambda step_spec, node, inputs: _step_success(
            node,
            "ok",
        ),
    )
    connection = _RecordingConnection()

    persist_generation_result(
        cast(Any, connection),
        generation_run=execution.generation_run,
        node_attempts=execution.node_attempts,
    )

    compiled = [_postgres_sql(statement) for statement in connection.calls]
    assert len(compiled) == 2
    assert all("ON CONFLICT" in statement for statement in compiled)


def test_platform_worker_import_registers_entrypoint() -> None:
    from dr_dspy.platform import worker

    assert worker.APP is not None


def test_platform_worker_run_one_uses_shared_workflow_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_dspy.platform import worker

    class RuntimeConfig:
        database_url = "postgresql://example/db"

    calls: list[tuple[str, Any]] = []

    def load_env_file(env_file: Any = None) -> None:
        calls.append(("load_env", env_file))

    def configure_runtime(
        *,
        database_url: str | None,
        dbos_system_database_url: str | None,
    ) -> RuntimeConfig:
        calls.append(
            (
                "configure",
                (database_url, dbos_system_database_url),
            )
        )
        return RuntimeConfig()

    def run_once(
        *,
        database_url: str,
        prediction_id: str,
        attempt_index: int,
    ) -> str:
        calls.append(
            (
                "run_once",
                (database_url, prediction_id, attempt_index),
            )
        )
        return "generation-run-1"

    def destroy_runtime() -> None:
        calls.append(("destroy", None))

    monkeypatch.setattr(worker, "load_env_file", load_env_file)
    monkeypatch.setattr(
        worker,
        "configure_platform_dbos_runtime",
        configure_runtime,
    )
    monkeypatch.setattr(worker, "run_prediction_graph_workflow_once", run_once)
    monkeypatch.setattr(
        worker.shared_dbos,
        "destroy_dbos_runtime",
        destroy_runtime,
    )

    worker.run_one(
        prediction_id="prediction-1",
        attempt_index=3,
        database_url="postgresql://app/db",
        dbos_system_database_url="postgresql://dbos/db",
        env_file=None,
    )

    assert calls == [
        ("load_env", None),
        ("configure", ("postgresql://app/db", "postgresql://dbos/db")),
        ("run_once", ("postgresql://example/db", "prediction-1", 3)),
        ("destroy", None),
    ]


def test_generation_clock_steps_have_distinct_dbos_names() -> None:
    step_names = {
        graph_workflow.GENERATION_STARTED_AT_STEP_NAME,
        graph_workflow.GENERATION_COMPLETED_AT_STEP_NAME,
    }

    assert len(step_names) == 2


class _RecordingConnection:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def execute(self, statement: Any) -> None:
        self.calls.append(statement)


def _postgres_sql(statement: Any) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))
