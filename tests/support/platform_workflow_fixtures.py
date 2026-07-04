"""Shared prediction-graph fixtures for unit and integration tests."""

from __future__ import annotations

from datetime import UTC, datetime

from whetstone.graph import NodeOutput, NodeSpec
from whetstone.platform.node_execution import NodeStepResult
from whetstone.platform.spec_builder import (
    decoder_node,
    direct_node,
    encdec_graph,
    encdec_spec,
    encoder_node,
    prediction_spec,
    provider_ref,
)
from whetstone.records import NodeAttemptStatus, ProviderConfigRef

NOW = datetime(2026, 6, 29, 12, 0, tzinfo=UTC)

__all__ = [
    "NOW",
    "decoder_node",
    "direct_node",
    "encdec_graph",
    "encdec_spec",
    "encoder_node",
    "prediction_spec",
    "provider_ref",
    "step_error",
    "step_success",
]


def step_success(
    node: NodeSpec,
    value: str,
    *,
    provider: ProviderConfigRef | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> NodeStepResult:
    started = started_at or NOW
    completed = completed_at or NOW
    return NodeStepResult.success(
        node_id=node.id,
        provider_config=provider or provider_ref(),
        output=NodeOutput(values={node.config.output_field: value}),
        usage_metadata={"total_tokens": 3},
        provider_cost=0.01,
        response_metadata={"id": f"response-{node.id}"},
        started_at=started,
        completed_at=completed,
    )


def step_error(
    node: NodeSpec,
    message: str,
    *,
    provider: ProviderConfigRef | None = None,
) -> NodeStepResult:
    from whetstone.eval_failures import FailureClass
    from whetstone.records import FailureMetadataPayload

    return NodeStepResult(
        node_id=node.id,
        status=NodeAttemptStatus.ERROR,
        provider_config=provider or provider_ref(),
        failure=FailureMetadataPayload(
            failure_class=FailureClass.PERMANENT,
            error_type="PermanentFailureError",
            message=message,
            metadata={"node_id": node.id},
        ),
        started_at=NOW,
        completed_at=NOW,
    )
