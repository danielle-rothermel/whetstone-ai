from __future__ import annotations

from datetime import UTC, datetime

import pytest
from dr_graph import NodeOutput
from dr_providers import EndpointKind, ProviderKind

from whetstone.eval_failures import (
    PermanentFailureError,
    TransientFailureError,
)
from whetstone.platform.node_execution import (
    NodeStepFailure,
    NodeStepResult,
    attach_node_step_timing_to_exception,
    node_step_timing_from_exception,
)
from whetstone.records import NodeAttemptStatus, ProviderConfigRef

NOW = datetime(2026, 6, 29, 12, 0, tzinfo=UTC)
LATER = datetime(2026, 6, 29, 12, 1, tzinfo=UTC)


def _provider() -> ProviderConfigRef:
    return ProviderConfigRef(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.RESPONSES,
        model="gpt-test",
        config_id="default",
        throttle_key="openai:responses:gpt-test",
        parameters={"temperature": 0.2},
    )


def test_node_step_timing_round_trips_through_classified_exception() -> None:
    started_at = datetime(2026, 6, 29, 12, 0, tzinfo=UTC)
    completed_at = datetime(2026, 6, 29, 12, 1, tzinfo=UTC)
    error = TransientFailureError(
        "temporary provider failure",
        metadata={},
    )

    attach_node_step_timing_to_exception(
        error,
        started_at=started_at,
        completed_at=completed_at,
    )

    timing = node_step_timing_from_exception(error)
    assert timing == (started_at, completed_at)


def test_node_step_timing_round_trips_through_plain_exception() -> None:
    started_at = datetime(2026, 6, 29, 12, 0, tzinfo=UTC)
    completed_at = datetime(2026, 6, 29, 12, 1, tzinfo=UTC)
    error = RuntimeError("plain failure")

    attach_node_step_timing_to_exception(
        error,
        started_at=started_at,
        completed_at=completed_at,
    )

    timing = node_step_timing_from_exception(error)
    assert timing == (started_at, completed_at)


def test_node_step_result_graph_output_raises_for_error_status() -> None:
    result = NodeStepResult.error(
        node_id="direct",
        provider_config=_provider(),
        error=RuntimeError("failed"),
        started_at=NOW,
        completed_at=LATER,
    )

    with pytest.raises(NodeStepFailure, match="failed"):
        result.graph_output()


def test_graph_output_raises_when_success_has_no_output() -> None:
    result = NodeStepResult(
        node_id="direct",
        status=NodeAttemptStatus.SUCCESS,
        output=None,
        started_at=NOW,
        completed_at=LATER,
    )

    with pytest.raises(PermanentFailureError, match="no output"):
        result.graph_output()


def test_node_step_result_graph_output_returns_output_on_success() -> None:
    result = NodeStepResult.success(
        node_id="direct",
        provider_config=_provider(),
        output=NodeOutput(values={"code": "def f(): pass"}),
        usage_metadata={"total_tokens": 1},
        provider_cost=0.01,
        response_metadata={"id": "resp-1"},
        started_at=NOW,
        completed_at=LATER,
    )

    assert result.graph_output().values == {"code": "def f(): pass"}


def test_execute_lm_node_end_to_end_with_scripted_provider() -> None:
    """Acceptance: full node execution against ScriptedProvider, no network.

    Exercises spec → runtime config → LlmRequest (with idempotency key)
    → ScriptedProvider.complete → ProviderResult → NodeStepResult.
    """
    from dr_graph import GraphSpec
    from dr_providers import (
        CostInfo,
        ScriptedOutcome,
        ScriptedProvider,
        TokenUsage,
    )

    from tests.test_platform_graph_workflow import _node, _spec
    from whetstone.platform.node_execution import execute_lm_node

    node = _node("direct", bindings={"prompt": "task.prompt"})
    spec = _spec(GraphSpec(nodes=(node,), terminal_node_id="direct"))
    provider = ScriptedProvider(
        [
            ScriptedOutcome(
                text="def add(a, b):\n    return a + b\n",
                usage=TokenUsage(total_tokens=7),
                cost=CostInfo(total_cost=0.003),
                provider_metadata={"usage": {"total_tokens": 7}},
            )
        ]
    )

    result = execute_lm_node(
        spec=spec,
        node=node,
        node_inputs={"prompt": "write add"},
        provider=provider,
        idempotency_key="node-attempt-42",
    )

    assert result.status is NodeAttemptStatus.SUCCESS
    assert result.output is not None
    assert result.output.values == {
        node.config.output_field: "def add(a, b):\n    return a + b\n"
    }
    assert result.usage_cost.provider_cost == 0.003
    assert result.usage_cost.usage_metadata == {"total_tokens": 7}
    served = provider.requests[0]
    assert served.idempotency_key == "node-attempt-42"
    assert provider.payloads[0]["model"] == served.provider_config.model
