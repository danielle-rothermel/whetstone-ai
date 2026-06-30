from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dr_dspy.eval_failures import (
    PermanentFailureError,
    TransientFailureError,
)
from dr_dspy.graph import NodeOutput
from dr_dspy.lm.boundary import EndpointKind, ProviderKind
from dr_dspy.platform.node_execution import (
    NodeStepFailure,
    NodeStepResult,
    attach_node_step_timing_to_exception,
    node_step_timing_from_exception,
)
from dr_dspy.records import NodeAttemptStatus, ProviderConfigRef

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
