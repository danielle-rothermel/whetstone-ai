"""Reshape legacy v0 prediction rows into v1 platform records."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_dspy.eval_failures.types import FailureClass
from dr_dspy.graph import (
    BindingRef,
    FieldRole,
    FieldSpec,
    GraphSpec,
    NodeConfig,
    NodeSpec,
    graph_digest,
)
from dr_dspy.lm.boundary import EndpointKind, ProviderKind
from dr_dspy.records import (
    DimensionsPayload,
    FailureMetadataPayload,
    GenerationRunRecord,
    GenerationRunStatus,
    GenerationRunSummaryPayload,
    GenerationTerminalErrorPayload,
    GraphSnapshotPayload,
    NodeAttemptRecord,
    NodeAttemptStatus,
    NodeOutputPayload,
    PredictionSpecRecord,
    ProviderConfigRef,
    ResponseMetadataPayload,
    TaskInputsPayload,
    TaskSnapshotPayload,
    UsageCostPayload,
    dimensions_digest,
    fair_order_key,
    stable_generation_run_id,
    stable_node_attempt_id,
    stable_prediction_id,
)

V0_TERMINAL_GENERATION_STATUSES = frozenset({"generated", "generation_error"})
V0_SOURCE_METADATA_KEY = "v0_source"
DEFAULT_FAIR_ORDER_SEED = "legacy-v0-migration"
DEFAULT_PROVIDER_KIND = ProviderKind.OPENROUTER
DEFAULT_ENDPOINT_KIND = EndpointKind.CHAT_COMPLETIONS


class V0ReshapeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: PredictionSpecRecord
    generation_run: GenerationRunRecord | None = None
    node_attempts: tuple[NodeAttemptRecord, ...] = Field(default_factory=tuple)
    source_metadata: dict[str, Any] = Field(default_factory=dict)


def reshape_v0_direct_row(row: Mapping[str, Any]) -> V0ReshapeResult:
    graph = _direct_graph()
    dimensions = DimensionsPayload(
        values={
            "model": row["model"],
            "temperature": row.get("temperature"),
            "reasoning": row.get("reasoning") or {},
        }
    )
    provider = _provider_ref(
        model=row["model"],
        config_id="main",
        parameters={"temperature": row.get("temperature")},
    )
    spec = _prediction_spec_from_row(
        row=row,
        graph=graph,
        layout="direct",
        dimensions=dimensions,
        provider_configs=(provider,),
        provider_axis=provider,
    )
    source_metadata = _source_metadata(row, layout="direct")
    if row.get("generation_status") not in V0_TERMINAL_GENERATION_STATUSES:
        return V0ReshapeResult(
            spec=spec,
            source_metadata=source_metadata,
        )
    generation_run_id = stable_generation_run_id(
        prediction_id=spec.prediction_id,
        attempt_index=0,
    )
    started_at, completed_at = _generation_timestamps(row)
    if row["generation_status"] == "generated":
        terminal_output = row.get("raw_code")
        generation_run = GenerationRunRecord(
            generation_run_id=generation_run_id,
            prediction_id=spec.prediction_id,
            attempt_index=0,
            status=GenerationRunStatus.SUCCESS,
            terminal_node_id="direct",
            terminal_output_node_id="direct",
            summary=GenerationRunSummaryPayload(
                execution_order=("direct",),
                terminal_node_id="direct",
                terminal_output=terminal_output,
                metadata={V0_SOURCE_METADATA_KEY: source_metadata},
            ),
            started_at=started_at,
            completed_at=completed_at,
        )
        node_attempts = (
            NodeAttemptRecord(
                node_attempt_id=stable_node_attempt_id(
                    generation_run_id=generation_run_id,
                    node_id="direct",
                    attempt_index=0,
                ),
                generation_run_id=generation_run_id,
                prediction_id=spec.prediction_id,
                node_id="direct",
                attempt_index=0,
                status=NodeAttemptStatus.SUCCESS,
                provider_config=provider,
                output=NodeOutputPayload(
                    values={"output": terminal_output or ""},
                ),
                usage_cost=UsageCostPayload(
                    usage_metadata=row.get("usage_metadata") or {},
                    provider_cost=row.get("provider_cost"),
                ),
                response_metadata=ResponseMetadataPayload(
                    response_metadata=row.get("response_metadata") or {},
                ),
                started_at=started_at,
                completed_at=completed_at,
            ),
        )
        return V0ReshapeResult(
            spec=spec,
            generation_run=generation_run,
            node_attempts=node_attempts,
            source_metadata=source_metadata,
        )

    failure = _failure_from_v0_generation(row)
    generation_run = GenerationRunRecord(
        generation_run_id=generation_run_id,
        prediction_id=spec.prediction_id,
        attempt_index=0,
        status=GenerationRunStatus.ERROR,
        terminal_node_id="direct",
        summary=GenerationRunSummaryPayload(
            execution_order=("direct",),
            terminal_node_id="direct",
            terminal_error=GenerationTerminalErrorPayload(
                node_id="direct",
                status=GenerationRunStatus.ERROR,
                failure=failure,
            ),
            metadata={V0_SOURCE_METADATA_KEY: source_metadata},
        ),
        started_at=started_at,
        completed_at=completed_at,
    )
    node_attempts = (
        NodeAttemptRecord(
            node_attempt_id=stable_node_attempt_id(
                generation_run_id=generation_run_id,
                node_id="direct",
                attempt_index=0,
            ),
            generation_run_id=generation_run_id,
            prediction_id=spec.prediction_id,
            node_id="direct",
            attempt_index=0,
            status=NodeAttemptStatus.ERROR,
            failure=failure,
            started_at=started_at,
            completed_at=completed_at,
        ),
    )
    return V0ReshapeResult(
        spec=spec,
        generation_run=generation_run,
        node_attempts=node_attempts,
        source_metadata=source_metadata,
    )


def reshape_v0_encdec_row(row: Mapping[str, Any]) -> V0ReshapeResult:
    graph = _encdec_graph()
    dimensions = DimensionsPayload(
        values={
            "encoder_model": row["encoder_model"],
            "decoder_model": row["decoder_model"],
            "encoder_temperature": row.get("encoder_temperature"),
            "decoder_temperature": row.get("decoder_temperature"),
            "budget_ratio": row.get("budget_ratio"),
            "encoder_reasoning": row.get("encoder_reasoning") or {},
            "decoder_reasoning": row.get("decoder_reasoning") or {},
        }
    )
    encoder = _provider_ref(
        model=row["encoder_model"],
        config_id="encoder",
        parameters={"temperature": row.get("encoder_temperature")},
    )
    decoder = _provider_ref(
        model=row["decoder_model"],
        config_id="decoder",
        parameters={"temperature": row.get("decoder_temperature")},
    )
    spec = _prediction_spec_from_row(
        row=row,
        graph=graph,
        layout="encdec",
        dimensions=dimensions,
        provider_configs=(encoder, decoder),
        provider_axis=decoder,
    )
    source_metadata = _source_metadata(row, layout="encdec")
    if row.get("generation_status") not in V0_TERMINAL_GENERATION_STATUSES:
        return V0ReshapeResult(
            spec=spec,
            source_metadata=source_metadata,
        )
    generation_run_id = stable_generation_run_id(
        prediction_id=spec.prediction_id,
        attempt_index=0,
    )
    started_at, completed_at = _generation_timestamps(row)
    status = row["generation_status"]
    if status == "generated":
        terminal_output = row.get("raw_code") or row.get("decoded_generation")
        generation_run = GenerationRunRecord(
            generation_run_id=generation_run_id,
            prediction_id=spec.prediction_id,
            attempt_index=0,
            status=GenerationRunStatus.SUCCESS,
            terminal_node_id="decoder",
            terminal_output_node_id="decoder",
            summary=GenerationRunSummaryPayload(
                execution_order=("encoder", "decoder"),
                terminal_node_id="decoder",
                terminal_output=terminal_output,
                metadata={
                    V0_SOURCE_METADATA_KEY: source_metadata,
                    "extraction_error": row.get("extraction_error"),
                },
            ),
            started_at=started_at,
            completed_at=completed_at,
        )
        node_attempts = (
            _encdec_encoder_attempt(
                row=row,
                spec=spec,
                generation_run_id=generation_run_id,
                provider=encoder,
                started_at=started_at,
                completed_at=completed_at,
            ),
            _encdec_decoder_attempt(
                row=row,
                spec=spec,
                generation_run_id=generation_run_id,
                provider=decoder,
                started_at=started_at,
                completed_at=completed_at,
                success=True,
            ),
        )
        return V0ReshapeResult(
            spec=spec,
            generation_run=generation_run,
            node_attempts=node_attempts,
            source_metadata=source_metadata,
        )

    failure = _failure_from_v0_generation(row)
    has_encoder_output = bool(row.get("encoded_description"))
    if has_encoder_output:
        run_status = GenerationRunStatus.PARTIAL
        execution_order = ("encoder", "decoder")
        node_attempts = (
            _encdec_encoder_attempt(
                row=row,
                spec=spec,
                generation_run_id=generation_run_id,
                provider=encoder,
                started_at=started_at,
                completed_at=completed_at,
            ),
            _encdec_decoder_attempt(
                row=row,
                spec=spec,
                generation_run_id=generation_run_id,
                provider=decoder,
                started_at=started_at,
                completed_at=completed_at,
                success=False,
                failure=failure,
            ),
        )
        terminal_error = GenerationTerminalErrorPayload(
            node_id="decoder",
            status=GenerationRunStatus.ERROR,
            failure=failure,
        )
    else:
        run_status = GenerationRunStatus.ERROR
        execution_order = ("encoder",)
        node_attempts = (
            NodeAttemptRecord(
                node_attempt_id=stable_node_attempt_id(
                    generation_run_id=generation_run_id,
                    node_id="encoder",
                    attempt_index=0,
                ),
                generation_run_id=generation_run_id,
                prediction_id=spec.prediction_id,
                node_id="encoder",
                attempt_index=0,
                status=NodeAttemptStatus.ERROR,
                failure=failure,
                started_at=started_at,
                completed_at=completed_at,
            ),
        )
        terminal_error = GenerationTerminalErrorPayload(
            node_id="encoder",
            status=GenerationRunStatus.ERROR,
            failure=failure,
        )
    generation_run = GenerationRunRecord(
        generation_run_id=generation_run_id,
        prediction_id=spec.prediction_id,
        attempt_index=0,
        status=run_status,
        terminal_node_id=terminal_error.node_id,
        summary=GenerationRunSummaryPayload(
            execution_order=execution_order,
            terminal_node_id=terminal_error.node_id,
            terminal_error=terminal_error,
            metadata={V0_SOURCE_METADATA_KEY: source_metadata},
        ),
        started_at=started_at,
        completed_at=completed_at,
    )
    return V0ReshapeResult(
        spec=spec,
        generation_run=generation_run,
        node_attempts=node_attempts,
        source_metadata=source_metadata,
    )


def _direct_graph() -> GraphSpec:
    node = _llm_node(
        "direct",
        bindings={"prompt": "task.prompt"},
        output_field="output",
        user_prompt_template="{prompt}",
    )
    return GraphSpec(nodes=(node,), terminal_node_id="direct")


def _encdec_graph() -> GraphSpec:
    encoder = _llm_node(
        "encoder",
        bindings={"prompt": "task.prompt"},
        output_field="description",
        user_prompt_template="Describe {prompt}",
        provider_config_id="encoder",
    )
    decoder = _llm_node(
        "decoder",
        bindings={"description": "encoder.description"},
        output_field="code",
        user_prompt_template="Write code from {description}",
        provider_config_id="decoder",
    )
    return GraphSpec(nodes=(decoder, encoder), terminal_node_id="decoder")


def _llm_node(
    node_id: str,
    *,
    bindings: dict[str, str],
    output_field: str,
    user_prompt_template: str,
    provider_config_id: str | None = None,
) -> NodeSpec:
    input_bindings = {
        name: BindingRef.model_validate(ref)
        for name, ref in bindings.items()
    }
    fields = [
        FieldSpec(name=name, role=FieldRole.INPUT)
        for name in input_bindings
    ]
    fields.append(FieldSpec(name=output_field, role=FieldRole.OUTPUT))
    metadata: dict[str, Any] = {
        "user_prompt_template": user_prompt_template,
    }
    if provider_config_id is not None:
        metadata["provider_config_id"] = provider_config_id
    return NodeSpec(
        id=node_id,
        config=NodeConfig(
            fields=tuple(fields),
            input_bindings=input_bindings,
            output_field=output_field,
            metadata=metadata,
        ),
    )


def _provider_ref(
    *,
    model: str,
    config_id: str | None,
    parameters: dict[str, Any] | None = None,
) -> ProviderConfigRef:
    params = {
        key: value
        for key, value in (parameters or {}).items()
        if value is not None
    }
    throttle_key = (
        f"{DEFAULT_PROVIDER_KIND.value}:"
        f"{DEFAULT_ENDPOINT_KIND.value}:{model}"
    )
    return ProviderConfigRef(
        provider_kind=DEFAULT_PROVIDER_KIND,
        endpoint_kind=DEFAULT_ENDPOINT_KIND,
        model=model,
        config_id=config_id,
        throttle_key=throttle_key,
        parameters=params,
    )


def _prediction_spec_from_row(
    *,
    row: Mapping[str, Any],
    graph: GraphSpec,
    layout: str,
    dimensions: DimensionsPayload,
    provider_configs: tuple[ProviderConfigRef, ...],
    provider_axis: ProviderConfigRef,
) -> PredictionSpecRecord:
    graph_id = graph_digest(graph)
    dimensions_id = dimensions_digest(dimensions)
    experiment_name = row["experiment_name"]
    task_id = row["task_id"]
    repetition_seed = int(row["repetition_seed"])
    prediction_id = stable_prediction_id(
        experiment_name=experiment_name,
        task_id=task_id,
        graph_digest=graph_id,
        dimensions_digest=dimensions_id,
        repetition_seed=repetition_seed,
        provider_kind=provider_axis.provider_kind.value,
        endpoint_kind=provider_axis.endpoint_kind.value,
        model=provider_axis.model,
        throttle_key=provider_axis.throttle_key,
    )
    return PredictionSpecRecord(
        prediction_id=prediction_id,
        experiment_name=experiment_name,
        task_id=task_id,
        repetition_seed=repetition_seed,
        graph=GraphSnapshotPayload(
            graph=graph,
            graph_digest=graph_id,
            layout=layout,
        ),
        dimensions=dimensions,
        dimensions_digest=dimensions_id,
        task=TaskSnapshotPayload(
            task_id=task_id,
            inputs=TaskInputsPayload(
                values={
                    "prompt": row["prompt"],
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                }
            ),
            metadata={
                "canonical_solution": row.get("canonical_solution") or "",
                "ground_truth_code": row.get("ground_truth_code") or "",
            },
        ),
        provider_configs=provider_configs,
        provider_axis=provider_axis,
        fair_order_seed=DEFAULT_FAIR_ORDER_SEED,
        fair_order_key=fair_order_key(
            experiment_seed=DEFAULT_FAIR_ORDER_SEED,
            prediction_id=prediction_id,
            provider=provider_axis.provider_kind.value,
            endpoint_kind=provider_axis.endpoint_kind.value,
            model=provider_axis.model,
            throttle_key=provider_axis.throttle_key,
            graph_layout=layout,
            task_id=task_id,
            repetition_seed=repetition_seed,
            config_axis=dimensions_id,
        ),
        created_at=_parse_timestamp(row.get("created_at")),
    )


def _encdec_encoder_attempt(
    *,
    row: Mapping[str, Any],
    spec: PredictionSpecRecord,
    generation_run_id: str,
    provider: ProviderConfigRef,
    started_at: datetime,
    completed_at: datetime,
) -> NodeAttemptRecord:
    return NodeAttemptRecord(
        node_attempt_id=stable_node_attempt_id(
            generation_run_id=generation_run_id,
            node_id="encoder",
            attempt_index=0,
        ),
        generation_run_id=generation_run_id,
        prediction_id=spec.prediction_id,
        node_id="encoder",
        attempt_index=0,
        status=NodeAttemptStatus.SUCCESS,
        provider_config=provider,
        output=NodeOutputPayload(
            values={"description": row.get("encoded_description") or ""},
        ),
        usage_cost=UsageCostPayload(
            usage_metadata=row.get("encoder_usage_metadata") or {},
            provider_cost=row.get("encoder_provider_cost"),
        ),
        response_metadata=ResponseMetadataPayload(
            response_metadata=row.get("encoder_response_metadata") or {},
        ),
        started_at=started_at,
        completed_at=completed_at,
    )


def _encdec_decoder_attempt(
    *,
    row: Mapping[str, Any],
    spec: PredictionSpecRecord,
    generation_run_id: str,
    provider: ProviderConfigRef,
    started_at: datetime,
    completed_at: datetime,
    success: bool,
    failure: FailureMetadataPayload | None = None,
) -> NodeAttemptRecord:
    if success:
        code = row.get("raw_code") or row.get("decoded_generation") or ""
        return NodeAttemptRecord(
            node_attempt_id=stable_node_attempt_id(
                generation_run_id=generation_run_id,
                node_id="decoder",
                attempt_index=0,
            ),
            generation_run_id=generation_run_id,
            prediction_id=spec.prediction_id,
            node_id="decoder",
            attempt_index=0,
            status=NodeAttemptStatus.SUCCESS,
            provider_config=provider,
            output=NodeOutputPayload(values={"code": code}),
            usage_cost=UsageCostPayload(
                usage_metadata=row.get("decoder_usage_metadata") or {},
                provider_cost=row.get("decoder_provider_cost"),
            ),
            response_metadata=ResponseMetadataPayload(
                response_metadata=row.get("decoder_response_metadata") or {},
            ),
            started_at=started_at,
            completed_at=completed_at,
        )
    assert failure is not None
    return NodeAttemptRecord(
        node_attempt_id=stable_node_attempt_id(
            generation_run_id=generation_run_id,
            node_id="decoder",
            attempt_index=0,
        ),
        generation_run_id=generation_run_id,
        prediction_id=spec.prediction_id,
        node_id="decoder",
        attempt_index=0,
        status=NodeAttemptStatus.ERROR,
        failure=failure,
        started_at=started_at,
        completed_at=completed_at,
    )


def _failure_from_v0_generation(
    row: Mapping[str, Any],
) -> FailureMetadataPayload:
    failure_class = row.get("generation_failure_class")
    parsed_class = None
    if failure_class is not None:
        try:
            parsed_class = FailureClass(failure_class)
        except ValueError:
            parsed_class = FailureClass.PERMANENT
    return FailureMetadataPayload(
        failure_class=parsed_class or FailureClass.PERMANENT,
        error_type=(
            row.get("generation_failure_exception_type") or "GenerationError"
        ),
        message=(
            row.get("generation_exception_message")
            or row.get("generation_error")
            or "generation failed"
        ),
        metadata=dict(row.get("generation_failure_metadata") or {}),
    )


def _source_metadata(row: Mapping[str, Any], *, layout: str) -> dict[str, Any]:
    return {
        "v0_prediction_id": row["prediction_id"],
        "v0_layout": layout,
        "v0_generation_status": row.get("generation_status"),
        "v0_scoring_status": row.get("scoring_status"),
    }


def _generation_timestamps(
    row: Mapping[str, Any],
) -> tuple[datetime, datetime]:
    started_at = _parse_timestamp(
        row.get("generated_at") or row.get("created_at"),
    )
    completed_source = (
        row.get("scored_at")
        or row.get("generated_at")
        or row.get("created_at")
    )
    completed_at = _parse_timestamp(completed_source)
    if completed_at < started_at:
        completed_at = started_at
    return started_at, completed_at


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed
    return datetime.now(UTC)
