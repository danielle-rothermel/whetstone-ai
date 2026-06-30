from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from dr_dspy.eval_failures import (
    PermanentFailureError,
    find_classified_exception,
    should_retry_step,
    summarize_exception,
)
from dr_dspy.graph import NodeOp, NodeOutput, NodeSpec
from dr_dspy.lm.boundary import (
    EndpointKind,
    ProviderConfig,
    ProviderKind,
    ProviderRequest,
    build_chat_completions_request,
    build_responses_request,
    call_provider_request,
    openai_chat_config,
    openai_responses_config,
    openrouter_chat_config,
    parse_provider_response,
)
from dr_dspy.platform.prompts import build_node_messages, node_prompt_spec
from dr_dspy.records import (
    FailureMetadataPayload,
    NodeAttemptStatus,
    NodeOutputPayload,
    PredictionSpecRecord,
    ProviderConfigRef,
    ResponseMetadataPayload,
    UsageCostPayload,
)

TEMPERATURE_PARAMETER = "temperature"
TOKEN_LIMIT_PARAMETER = "token_limit"
REASONING_PARAMETER = "reasoning"
EXTRA_BODY_PARAMETER = "extra_body"
EXTRA_KWARGS_PARAMETER = "extra_kwargs"
DEFAULT_PROVIDER_TIMEOUT_SECONDS = 120.0
NODE_STEP_STARTED_AT_METADATA_KEY = "node_step_started_at"
NODE_STEP_COMPLETED_AT_METADATA_KEY = "node_step_completed_at"

type ProviderClientFactory = Callable[[ProviderConfig], Any]
type ProviderCaller = Callable[[Any, ProviderRequest], Any]


class NodeStepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: StrictStr
    status: NodeAttemptStatus
    provider_config: ProviderConfigRef | None = None
    output: NodeOutputPayload | None = None
    usage_cost: UsageCostPayload = Field(default_factory=UsageCostPayload)
    response_metadata: ResponseMetadataPayload = Field(
        default_factory=ResponseMetadataPayload
    )
    failure: FailureMetadataPayload | None = None
    started_at: datetime
    completed_at: datetime

    @classmethod
    def success(
        cls,
        *,
        node_id: str,
        provider_config: ProviderConfigRef,
        output: NodeOutput,
        usage_metadata: Mapping[str, Any],
        provider_cost: float | None,
        response_metadata: Mapping[str, Any],
        started_at: datetime,
        completed_at: datetime,
    ) -> NodeStepResult:
        return cls(
            node_id=node_id,
            status=NodeAttemptStatus.SUCCESS,
            provider_config=provider_config,
            output=NodeOutputPayload(
                values=output.values,
                metadata=output.metadata,
            ),
            usage_cost=UsageCostPayload(
                usage_metadata=dict(usage_metadata),
                provider_cost=provider_cost,
            ),
            response_metadata=ResponseMetadataPayload(
                response_metadata=dict(response_metadata)
            ),
            started_at=started_at,
            completed_at=completed_at,
        )

    @classmethod
    def error(
        cls,
        *,
        node_id: str,
        provider_config: ProviderConfigRef | None,
        error: BaseException,
        started_at: datetime,
        completed_at: datetime,
    ) -> NodeStepResult:
        return cls(
            node_id=node_id,
            status=NodeAttemptStatus.ERROR,
            provider_config=provider_config,
            failure=failure_metadata_from_exception(error),
            started_at=started_at,
            completed_at=completed_at,
        )

    def graph_output(self) -> NodeOutput:
        if self.status is NodeAttemptStatus.ERROR:
            raise NodeStepFailure.from_result(self)
        if self.output is None:
            raise PermanentFailureError(
                "successful node step result has no output",
                metadata={"node_id": self.node_id},
            )
        return NodeOutput(
            values=self.output.values,
            metadata=self.output.metadata,
        )


class NodeStepFailure(Exception):
    def __init__(self, failure: FailureMetadataPayload) -> None:
        super().__init__(failure.message)
        self.error_type = failure.error_type
        self.failure_class = (
            failure.failure_class.value
            if failure.failure_class is not None
            else None
        )
        self.metadata = failure.metadata

    @classmethod
    def from_result(cls, result: NodeStepResult) -> NodeStepFailure:
        if result.failure is None:
            raise PermanentFailureError(
                "error node step result has no failure payload",
                metadata={"node_id": result.node_id},
            )
        return cls(result.failure)


def execute_lm_node(
    *,
    spec: PredictionSpecRecord,
    node: NodeSpec,
    node_inputs: Mapping[str, Any],
    client_factory: ProviderClientFactory | None = None,
    provider_caller: ProviderCaller = call_provider_request,
    raise_retryable: bool = False,
) -> NodeStepResult:
    started_at = datetime.now(UTC)
    provider_ref: ProviderConfigRef | None = None
    resolved_client_factory = client_factory or create_provider_client
    try:
        if node.op is not NodeOp.LLM_CALL:
            raise PermanentFailureError(
                "unsupported node operation for LM executor",
                metadata={
                    "node_id": node.id,
                    "node_op": str(node.op),
                },
            )
        provider_ref = provider_config_ref_for_node(spec=spec, node=node)
        runtime_config = runtime_provider_config(provider_ref)
        messages = build_node_messages(node=node, node_inputs=node_inputs)
        request = build_provider_request(
            config=runtime_config,
            messages=messages,
            parameters=merged_node_parameters(
                provider_ref=provider_ref,
                node=node,
            ),
        )
        response = provider_caller(
            resolved_client_factory(runtime_config),
            request,
        )
        result = parse_provider_response(
            response,
            config=runtime_config,
            output_field=node.config.output_field,
        )
        output = NodeOutput(
            values={node.config.output_field: result.text},
            metadata={
                key: value
                for key, value in {
                    "response_id": result.response_id,
                    "model": result.model,
                    "finish_reason": result.finish_reason,
                }.items()
                if value is not None
            },
        )
        completed_at = datetime.now(UTC)
        return NodeStepResult.success(
            node_id=node.id,
            provider_config=provider_ref,
            output=output,
            usage_metadata=result.usage_metadata,
            provider_cost=result.provider_cost,
            response_metadata=result.response_metadata,
            started_at=started_at,
            completed_at=completed_at,
        )
    except Exception as error:
        completed_at = datetime.now(UTC)
        if raise_retryable and should_retry_step(error):
            attach_node_step_timing_to_exception(
                error,
                started_at=started_at,
                completed_at=completed_at,
            )
            raise
        return NodeStepResult.error(
            node_id=node.id,
            provider_config=provider_ref,
            error=error,
            started_at=started_at,
            completed_at=completed_at,
        )


def attach_node_step_timing_to_exception(
    error: BaseException,
    *,
    started_at: datetime,
    completed_at: datetime,
) -> None:
    timing_metadata = {
        NODE_STEP_STARTED_AT_METADATA_KEY: started_at.isoformat(),
        NODE_STEP_COMPLETED_AT_METADATA_KEY: completed_at.isoformat(),
    }
    classified = find_classified_exception(error)
    if classified is not None:
        classified.metadata.update(timing_metadata)
        return
    error.__dict__["node_step_started_at"] = started_at
    error.__dict__["node_step_completed_at"] = completed_at


def node_step_timing_from_exception(
    error: BaseException,
) -> tuple[datetime, datetime] | None:
    classified = find_classified_exception(error)
    if classified is not None:
        started_at = classified.metadata.get(
            NODE_STEP_STARTED_AT_METADATA_KEY
        )
        completed_at = classified.metadata.get(
            NODE_STEP_COMPLETED_AT_METADATA_KEY
        )
        if isinstance(started_at, str) and isinstance(completed_at, str):
            return (
                datetime.fromisoformat(started_at),
                datetime.fromisoformat(completed_at),
            )
    started_at = error.__dict__.get("node_step_started_at")
    completed_at = error.__dict__.get("node_step_completed_at")
    if isinstance(started_at, datetime) and isinstance(completed_at, datetime):
        return started_at, completed_at
    return None


def node_step_error_result_from_failure(
    *,
    spec: PredictionSpecRecord,
    node: NodeSpec,
    failure: FailureMetadataPayload,
    started_at: datetime,
    completed_at: datetime,
) -> NodeStepResult:
    try:
        provider_ref = provider_config_ref_for_node(spec=spec, node=node)
    except Exception:
        provider_ref = None
    return NodeStepResult(
        node_id=node.id,
        status=NodeAttemptStatus.ERROR,
        provider_config=provider_ref,
        failure=failure,
        started_at=started_at,
        completed_at=completed_at,
    )


def provider_config_ref_for_node(
    *,
    spec: PredictionSpecRecord,
    node: NodeSpec,
) -> ProviderConfigRef:
    prompt_spec = node_prompt_spec(node)
    if prompt_spec.provider_config_id is not None:
        for provider_config in spec.provider_configs:
            if provider_config.config_id == prompt_spec.provider_config_id:
                return provider_config
        raise PermanentFailureError(
            "node references unknown provider_config_id",
            metadata={
                "node_id": node.id,
                "provider_config_id": prompt_spec.provider_config_id,
            },
        )

    if len(spec.provider_configs) == 1:
        return spec.provider_configs[0]

    raise PermanentFailureError(
        "node must declare provider_config_id when spec has multiple configs",
        metadata={"node_id": node.id},
    )


def runtime_provider_config(provider_ref: ProviderConfigRef) -> ProviderConfig:
    """Build the runtime provider config supported by today's spec record.

    ``ProviderConfigRef`` currently persists provider kind, endpoint kind,
    model, throttle key, and request parameters only. Custom provider runtime
    details such as ``base_url``, ``api_key_env``, and capability flags remain
    template-owned until the provider config contract is expanded.
    """

    if (
        provider_ref.provider_kind is ProviderKind.OPENROUTER
        and provider_ref.endpoint_kind is EndpointKind.CHAT_COMPLETIONS
    ):
        config = openrouter_chat_config(model=provider_ref.model)
    elif (
        provider_ref.provider_kind is ProviderKind.OPENAI
        and provider_ref.endpoint_kind is EndpointKind.CHAT_COMPLETIONS
    ):
        config = openai_chat_config(model=provider_ref.model)
    elif (
        provider_ref.provider_kind is ProviderKind.OPENAI
        and provider_ref.endpoint_kind is EndpointKind.RESPONSES
    ):
        config = openai_responses_config(model=provider_ref.model)
    else:
        raise PermanentFailureError(
            "unsupported provider endpoint for platform graph workflow",
            metadata={
                "provider_kind": provider_ref.provider_kind.value,
                "endpoint_kind": provider_ref.endpoint_kind.value,
                "model": provider_ref.model,
            },
        )
    return config.model_copy(
        update={"throttle_key": provider_ref.throttle_key}
    )


def merged_node_parameters(
    *,
    provider_ref: ProviderConfigRef,
    node: NodeSpec,
) -> dict[str, Any]:
    return {
        **provider_ref.parameters,
        **node.config.parameters,
    }


def build_provider_request(
    *,
    config: ProviderConfig,
    messages: Any,
    parameters: Mapping[str, Any],
) -> ProviderRequest:
    request_kwargs = {
        "config": config,
        "messages": messages,
        "temperature": parameters.get(TEMPERATURE_PARAMETER),
        "token_limit": parameters.get(TOKEN_LIMIT_PARAMETER),
        "reasoning": parameters.get(REASONING_PARAMETER),
        "extra_body": parameters.get(EXTRA_BODY_PARAMETER),
        "extra_kwargs": parameters.get(EXTRA_KWARGS_PARAMETER),
    }
    if config.endpoint_kind is EndpointKind.CHAT_COMPLETIONS:
        return build_chat_completions_request(**request_kwargs)
    if config.endpoint_kind is EndpointKind.RESPONSES:
        return build_responses_request(**request_kwargs)
    raise PermanentFailureError(
        "unsupported provider endpoint kind",
        metadata={"endpoint_kind": config.endpoint_kind.value},
    )


def create_provider_client(config: ProviderConfig) -> OpenAI:
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise PermanentFailureError(
            "provider API key environment variable is not set",
            metadata={
                "api_key_env": config.api_key_env,
                "provider_kind": config.provider_kind.value,
            },
        )
    return OpenAI(
        api_key=api_key,
        base_url=config.base_url,
        timeout=DEFAULT_PROVIDER_TIMEOUT_SECONDS,
        max_retries=0,
    )


def failure_metadata_from_exception(
    error: BaseException,
) -> FailureMetadataPayload:
    summary = summarize_exception(error)
    return FailureMetadataPayload(
        failure_class=summary.failure_class,
        error_type=summary.failure_exception_type,
        underlying_exception_type=summary.underlying_exception_type,
        message=summary.message,
        metadata=summary.failure_metadata,
    )
