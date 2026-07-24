"""The Operator Retry gate and the retried-executor binding recheck.

Proves the gateway invokes ``retry_stage`` only for a FAILED Stage whose
Rollout Execution Key is unbound, and that the retried executor rechecks the
authoritative binding before any new semantic effect — including the race where
the key is bound between the gateway check and the executor start, which the
executor must stop via idempotent-or-conflict, never overwriting.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from dr_platform.staging.states import StageExecutionState

from whetstone.orchestration import (
    OperatorRetryRefused,
    RetriedExecutorStop,
    RetryRefusalReason,
    assert_unbound_before_effect,
    operator_retry,
)
from whetstone.result import ResultBindStatus

from .platform_support import (
    NOW,
    admit,
    configure_default_and_quota_capacities,
    drive_to_failed,
    migrate,
    register_pipeline,
    stage_execution_id_for,
    submit_one,
)
from .support import (
    build_harness,
    execution_key,
    in_memory_result_store,
    quota,
    response_outcome,
)


def _response(text: str = "answer"):
    return response_outcome(text=text)


def _failed_stage(engine):
    """Submit one item, admit it, and drive it FAILED; return (id, key)."""
    schema = migrate(engine)
    registry = register_pipeline()
    key = execution_key()
    configure_default_and_quota_capacities(
        engine, default_capacity=4, quota_capacities=()
    )
    from .support import work_request

    input_ref = build_harness(outcomes=[_response()], key=key).input_ref
    submit_one(
        engine,
        registry,
        execution_key=key,
        input_ref=input_ref,
        quotas=(quota(),),
    )
    admit(engine, registry)
    stage_id = stage_execution_id_for(engine, execution_key=key, schema=schema)
    drive_to_failed(engine, stage_execution_id=stage_id, schema=schema)
    _ = work_request  # referenced for symmetry with the harness builder
    return stage_id, key


def test_gateway_permits_retry_for_failed_unbound_stage(
    pg_engine,
) -> None:
    stage_id, key = _failed_stage(pg_engine)
    store = in_memory_result_store()
    result = operator_retry(
        stage_execution_id=stage_id,
        execution_key=key,
        result_store=store,
        engine=pg_engine,
        clock=lambda: NOW + timedelta(seconds=2),
    )
    # A new Platform Stage Attempt was prepared and the stage is READY again.
    assert result.stage_retry.stage_execution.state is (
        StageExecutionState.READY
    )
    assert result.stage_retry.new_attempt.attempt_number == 2


def test_gateway_refuses_when_stage_is_not_failed(pg_engine) -> None:
    schema = migrate(pg_engine)
    registry = register_pipeline()
    key = execution_key()
    configure_default_and_quota_capacities(
        pg_engine, default_capacity=4, quota_capacities=()
    )
    input_ref = build_harness(outcomes=[_response()], key=key).input_ref
    submit_one(
        pg_engine,
        registry,
        execution_key=key,
        input_ref=input_ref,
        quotas=(quota(),),
    )
    admit(pg_engine, registry)
    stage_id = stage_execution_id_for(
        pg_engine, execution_key=key, schema=schema
    )
    # The stage is ADMITTED, not FAILED.
    store = in_memory_result_store()
    with pytest.raises(OperatorRetryRefused) as caught:
        operator_retry(
            stage_execution_id=stage_id,
            execution_key=key,
            result_store=store,
            engine=pg_engine,
        )
    assert caught.value.reason is RetryRefusalReason.STAGE_NOT_FAILED


def test_gateway_refuses_when_key_already_bound(pg_engine) -> None:
    stage_id, key = _failed_stage(pg_engine)
    # Bind the key first (a terminal Result already exists).
    harness = build_harness(outcomes=[_response("winner")], key=key)
    harness.context.run_stage(harness.input_ref)

    with pytest.raises(OperatorRetryRefused) as caught:
        operator_retry(
            stage_execution_id=stage_id,
            execution_key=key,
            result_store=harness.result_store,
            engine=pg_engine,
        )
    assert caught.value.reason is RetryRefusalReason.KEY_ALREADY_BOUND
    assert caught.value.existing_binding is not None


def test_gateway_refuses_for_unknown_stage(pg_engine) -> None:
    migrate(pg_engine)
    store = in_memory_result_store()
    with pytest.raises(OperatorRetryRefused) as caught:
        operator_retry(
            stage_execution_id=999_999,
            execution_key=execution_key(),
            result_store=store,
            engine=pg_engine,
        )
    assert caught.value.reason is RetryRefusalReason.STAGE_UNKNOWN


# ---------------------------------------------------------------------------
# The race: a bind lands between the gateway check and the executor start
# ---------------------------------------------------------------------------


def test_recheck_stops_when_key_bound_after_gateway_check() -> None:
    """The core race proof (pure Result Store level).

    The gateway precheck sees the key unbound and permits retry. Before the
    retried executor issues any new semantic effect, a concurrent actor binds
    the key. The executor recheck (``assert_unbound_before_effect``) then stops
    with :class:`RetriedExecutorStop`; it never overwrites the winner.
    """
    store = in_memory_result_store()
    key = execution_key()

    # Gateway precheck: unbound.
    assert store.resolve(key) is None

    # Concurrent winner binds between the gateway check and the executor start.
    winner = build_harness(
        outcomes=[_response("winner")], key=key, result_store=store
    )
    winner.context.run_stage(winner.input_ref)
    existing = store.resolve(key)
    assert existing is not None

    # The retried executor's recheck refuses to proceed to a new effect.
    with pytest.raises(RetriedExecutorStop) as caught:
        assert_unbound_before_effect(execution_key=key, result_store=store)
    assert caught.value.existing_binding == existing


def test_retried_executor_stops_via_idempotency_never_overwrites() -> None:
    """End-to-end race through the durable stage body.

    Two executors share one Result Store and one key. The first binds a
    ``winner`` Result. The second (the retried executor) runs its full stage
    body: its recheck finds the key already bound, so it issues NO new provider
    call and returns the existing authoritative binding idempotently — the
    winner is never overwritten.
    """
    store = in_memory_result_store()
    key = execution_key()

    winner = build_harness(
        outcomes=[_response("winner")], key=key, result_store=store
    )
    winner_outcome = winner.context.run_stage(winner.input_ref)
    assert winner_outcome.bind_status is ResultBindStatus.BOUND
    winning_reference = store.resolve(key)

    # A second executor whose transport would return a DIFFERENT generation.
    loser = build_harness(
        outcomes=[_response("loser")], key=key, result_store=store
    )
    loser_outcome = loser.context.run_stage(loser.input_ref)

    # The retried executor stopped idempotently: no new provider call issued.
    assert loser.transport.calls == 0
    assert loser_outcome.bind_status is ResultBindStatus.IDEMPOTENT
    # The winner's binding is preserved unchanged; never overwritten.
    assert store.resolve(key) == winning_reference
