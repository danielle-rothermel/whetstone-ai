"""The durable stage body's pure (DBOS-free) behavior.

Exercises ``ExecutorContext.run_stage`` directly (no DBOS runtime): terminal
assembly from the checkpointed provider result, immutable persistence, and
authoritative Result Store binding for both semantic success and exhausted
failure; the reconstructed-Node-Outcomes / Graph-Run-Result determinism; the
reference-not-duplicate provider-body seam; and the result-based terminality
rule. The DBOS step / durable sleep / replay wiring is proved in the
integration test.
"""

from __future__ import annotations

from dr_providers import FailureClass, ProviderTransportFailure

from whetstone.result import (
    ROLLOUT_RESULT_SCHEMA,
    ResultBindStatus,
    RolloutResult,
    encode_rollout_execution_key,
)

from .support import (
    Harness,
    build_harness,
    execution_key,
    failure_outcome,
    response_outcome,
)


def _response(text: str = "the answer"):
    return response_outcome(text=text)


def _failure(
    *, failure_class: FailureClass = FailureClass.TRANSIENT
) -> ProviderTransportFailure:
    return failure_outcome(failure_class=failure_class, message="boom")


def test_semantic_success_persists_and_binds_a_terminal_result() -> None:
    harness = build_harness(outcomes=[_response("hello world")])
    outcome = harness.context.run_stage(harness.input_ref)

    assert outcome.semantic_success is True
    assert outcome.bind_status is ResultBindStatus.BOUND
    assert outcome.reference_schema == "whetstone.rollout_result"
    # The Rollout Execution Key is now authoritatively bound.
    assert (
        harness.result_store.resolve(
            harness.request.rollout_execution_key
        )
        is not None
    )
    # Exactly one physical provider call.
    assert harness.transport.calls == 1


def test_exhausted_failure_is_a_succeeded_stage_with_a_bound_result() -> None:
    """Semantic exhaustion is expected output, not operational failure.

    A retryable failure repeated to the attempt bound produces an exhausted
    causal failure Rollout Result that is still persisted and bound — a
    SUCCEEDED Stage.
    """
    harness = build_harness(
        outcomes=[_failure()],
        max_attempts=3,
    )
    outcome = harness.context.run_stage(harness.input_ref)

    assert outcome.semantic_success is False
    assert outcome.bind_status is ResultBindStatus.BOUND
    # Bounded retries were spent (three logical attempts).
    assert harness.transport.calls == 3
    reference = harness.result_store.resolve(
        harness.request.rollout_execution_key
    )
    assert reference is not None


def test_reconstruction_is_deterministic_across_runs() -> None:
    """Same recorded outcomes -> byte-identical terminal reference.

    Two independent executors over the same scripted outcomes bind the same
    content-addressed Rollout Result reference: the Node Outcomes and Graph Run
    Result reconstruction is deterministic.
    """
    a = build_harness(outcomes=[_response("stable")])
    b = build_harness(outcomes=[_response("stable")])
    ref_a = a.context.run_stage(a.input_ref).output_reference
    ref_b = b.context.run_stage(b.input_ref).output_reference
    assert ref_a == ref_b


def test_graph_run_result_references_provider_bodies_without_duplicating() -> (
    None
):
    harness = build_harness(outcomes=[_response("ok")])
    harness.context.run_stage(harness.input_ref)
    result = _resolve_result(harness)

    # Every attempt-evidence ref the nested Graph Run Result carries resolves
    # to a Provider Call Attempt observation held on the Rollout Result.
    held = {
        obs.evidence_ref for obs in result.provider_call_attempts
    }
    for ref in result.graph_run_result.attempt_evidence_refs:
        assert ref in held
    # The provider bodies live on the observation, not duplicated in the graph.
    assert result.provider_call_attempts
    assert result.provider_call_attempts[0].provider_invocation_evidence


def test_success_result_has_no_exhausted_failure() -> None:
    harness = build_harness(outcomes=[_response("ok")])
    harness.context.run_stage(harness.input_ref)
    result = _resolve_result(harness)
    assert result.exhausted_failure is None
    assert result.scores  # a success carries a measurement


def test_exhausted_result_carries_the_causal_failure() -> None:
    harness = build_harness(outcomes=[_failure()], max_attempts=2)
    harness.context.run_stage(harness.input_ref)
    result = _resolve_result(harness)
    assert result.exhausted_failure is not None
    assert result.exhausted_failure.failure_class == "transport-error"
    assert not result.scores


def test_stage_records_the_dbos_workflow_evidence_slot() -> None:
    """Off a DBOS runtime the workflow-id evidence is simply absent.

    The evidence slot is present (a Platform Stage Attempt evidence object);
    outside a workflow the dbos_workflow_id is None. No Platform Stage *state*
    is stored on the Rollout Result.
    """
    harness = build_harness(outcomes=[_response("ok")])
    harness.context.run_stage(harness.input_ref)
    result = _resolve_result(harness)
    assert result.stage_attempt_evidence.dbos_workflow_id is None
    assert result.stage_attempt_evidence.durability_replay_count == 0


def test_logical_call_id_is_deterministic_in_the_execution_key() -> None:
    from whetstone.orchestration.executor import ExecutorContext

    from .support import work_request

    key = execution_key()
    request = work_request(key=key)
    expected = "llm:" + encode_rollout_execution_key(key)
    assert ExecutorContext._logical_call_id(request) == expected


def _resolve_result(harness: Harness) -> RolloutResult:
    reference = harness.result_store.resolve(
        harness.request.rollout_execution_key
    )
    assert reference is not None
    content = harness.result_store.store.get(reference)
    assert reference.schema == ROLLOUT_RESULT_SCHEMA
    return RolloutResult.model_validate(content)
