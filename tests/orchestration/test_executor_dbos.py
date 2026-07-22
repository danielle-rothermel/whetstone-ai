"""End-to-end durable executor over dr-platform + DBOS.

Launches the real DBOS runtime and runs the concrete Whetstone Orchestration
Pipeline through submission → admission → the durable rollout-execution stage
body → a persisted-and-bound terminal Rollout Result → an operationally
SUCCEEDED Stage. Also proves each Provider Call Attempt runs in one
checkpointed DBOS step and that a completed logical attempt is checkpointed
(the wire transport is invoked exactly once per logical attempt across a run).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import uuid4

from dbos import DBOS, DBOSClient, DBOSConfig, Queue
from dr_platform.staging import PipelineRegistry
from dr_platform.staging.admission import run_admission_pass
from dr_platform.staging.states import StageExecutionState
from sqlalchemy import func, select

from whetstone.orchestration import (
    ROLLOUT_EXECUTION_STAGE_QUEUE,
    ExecutorContext,
    configure_provider_concurrency,
    encode_work_request_ref,
    orchestration_pipeline,
    rollout_work_input,
    work_key_for_execution_key,
)
from whetstone.orchestration.pipeline import orchestration_pipeline_identity
from whetstone.provider.policy import BackoffSchedule, ProviderExecutionPolicy
from whetstone.result import (
    ROLLOUT_RESULT_SCHEMA,
    ResultStore,
    RolloutResult,
    encode_rollout_execution_key,
)

from .platform_support import NOW, engine_dsn, migrate
from .support import (
    ScriptedTransport,
    build_request,
    execution_key,
    failure_outcome,
    in_memory_result_store,
    quota,
    response_outcome,
    transport_policy,
    work_request,
)

if TYPE_CHECKING:
    from dr_providers import ProviderTransportOutcome
    from sqlalchemy import Engine

    from whetstone.graph.rollout import RolloutExecutionKey
    from whetstone.orchestration import RolloutWorkRequest


def _response(text: str = "durable answer") -> ProviderTransportOutcome:
    return response_outcome(text=text)


def _rate_limit() -> ProviderTransportOutcome:
    from dr_providers import FailureClass

    return failure_outcome(
        failure_class=FailureClass.RATE_LIMITED, message="429 slow down"
    )


def _launch_dbos(database_url: str, *, suffix: str) -> None:
    config: DBOSConfig = {
        "name": f"whetstone-exec-{suffix}",
        "system_database_url": database_url,
        "application_database_url": database_url,
        "application_version": f"exec-{suffix}",
        "run_admin_server": False,
        "use_listen_notify": False,
        "notification_listener_polling_interval_sec": 0.01,
    }
    DBOS(config=config)
    DBOS.launch()


def _wait_for(predicate, *, timeout_seconds: float = 8.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out waiting for the durable stage workflow")


def _succeeded_count(engine: Engine, schema) -> int:
    with engine.connect() as connection:
        return connection.execute(
            select(func.count())
            .select_from(schema.stage_executions)
            .where(
                schema.stage_executions.c.state
                == StageExecutionState.SUCCEEDED.value
            )
        ).scalar_one()


def _resolve_rollout_result(
    store: ResultStore, key: RolloutExecutionKey
) -> RolloutResult:
    reference = store.resolve(key)
    assert reference is not None
    assert reference.schema == ROLLOUT_RESULT_SCHEMA
    return RolloutResult.model_validate(store.store.get(reference))


def _run_pipeline(
    *,
    engine: Engine,
    outcomes: list[ProviderTransportOutcome],
    max_attempts: int = 2,
) -> tuple[
    ResultStore, ScriptedTransport, RolloutWorkRequest, RolloutExecutionKey
]:
    """Drive one Work Item through the durable pipeline; return artifacts.

    Each run uses a unique task identity so its derived work key and DBOS
    workflow id never collide with a prior run's rows in the shared ``dbos``
    schema (which ``clean_pg`` does not reset).
    """
    schema = migrate(engine)
    suffix = uuid4().hex[:10]
    key = execution_key(task_identity=f"task-{suffix}")
    request = work_request(key=key)
    store = in_memory_result_store()
    scripted = ScriptedTransport(policy=transport_policy(), outcomes=outcomes)

    def resolve(input_ref: str) -> RolloutWorkRequest:
        assert input_ref == encode_work_request_ref(request)
        return request

    context = ExecutorContext(
        result_store=store,
        transport=scripted,
        policy=ProviderExecutionPolicy(
            transport_policy=transport_policy(),
            max_attempts=max_attempts,
            backoff=BackoffSchedule(
                base_seconds=0.05, multiplier=2.0, max_seconds=0.2
            ),
        ),
        resolve_work_request=resolve,
        build_request=lambda _r: build_request(),
        use_durable_sleep=True,
    )
    pipeline = orchestration_pipeline(context.stage_callable())
    registry = PipelineRegistry()
    registry.register(pipeline)
    configure_provider_concurrency(
        engine=engine,
        default_capacity=4,
        quota_capacities=(),
        clock=lambda: NOW,
    )

    from dr_platform.staging.submission import submit

    submit(
        campaign_key="camp-1",
        run_key="run-1",
        pipeline=orchestration_pipeline_identity(),
        config_ref="config:1",
        items=(
            rollout_work_input(
                execution_key=key,
                input_ref=encode_work_request_ref(request),
                quotas=(quota(),),
            ),
        ),
        registry=registry,
        engine=engine,
        clock=lambda: NOW,
    )
    Queue(ROLLOUT_EXECUTION_STAGE_QUEUE, polling_interval_sec=0.02)

    client: DBOSClient | None = None
    try:
        _launch_dbos(engine_dsn(engine), suffix=suffix)
        client = DBOSClient(system_database_url=engine_dsn(engine))
        run_admission_pass(
            engine, client=client, registry=registry, clock=lambda: NOW
        )
        _wait_for(lambda: _succeeded_count(engine, schema) == 1)
        return store, scripted, request, key
    finally:
        if client is not None:
            client.destroy()
        DBOS.destroy(destroy_registry=True)


def test_durable_success_binds_terminal_result_and_succeeds_stage(
    clean_pg: str,
    pg_engine: Engine,
) -> None:
    store, scripted, _request, key = _run_pipeline(
        engine=pg_engine,
        outcomes=[_response("hello durable")],
    )
    # The terminal Rollout Result is persisted and authoritatively bound.
    result = _resolve_rollout_result(store, key)
    assert result.scores
    assert result.exhausted_failure is None
    # Exactly one physical provider call for one logical attempt.
    assert scripted.calls == 1
    # The bound reference is the record's own content-addressed reference.
    from whetstone.result import rollout_result_reference

    assert store.resolve(key) == rollout_result_reference(result)


def test_durable_retry_then_success_records_two_attempts(
    clean_pg: str,
    pg_engine: Engine,
) -> None:
    """One retryable failure then a success: two logical attempts, durable
    backoff sleep between them, one bound success Result."""
    store, scripted, _request, key = _run_pipeline(
        engine=pg_engine,
        outcomes=[_rate_limit(), _response("recovered")],
        max_attempts=3,
    )
    result = _resolve_rollout_result(store, key)
    assert result.exhausted_failure is None
    assert result.scores
    # Two logical attempts were checkpointed (retry then success).
    assert len(result.provider_call_attempts) == 2
    assert scripted.calls == 2


def test_durable_exhaustion_binds_failure_result_and_succeeds_stage(
    clean_pg: str,
    pg_engine: Engine,
) -> None:
    """Bounded retries all fail: an exhausted-failure Result is still bound and
    the Stage is operationally SUCCEEDED (result-based terminality)."""
    store, scripted, _request, key = _run_pipeline(
        engine=pg_engine,
        outcomes=[_rate_limit()],
        max_attempts=2,
    )
    result = _resolve_rollout_result(store, key)
    assert result.exhausted_failure is not None
    assert result.exhausted_failure.failure_class == "rate-limit"
    assert scripted.calls == 2  # bounded attempts spent


def test_stage_attempt_evidence_records_the_dbos_workflow_id(
    clean_pg: str,
    pg_engine: Engine,
) -> None:
    store, _scripted, _request, key = _run_pipeline(
        engine=pg_engine,
        outcomes=[_response("with evidence")],
    )
    result = _resolve_rollout_result(store, key)
    # Under a real workflow identity the evidence slot carries the workflow id.
    assert result.stage_attempt_evidence.dbos_workflow_id is not None
    # The logical call id is deterministic in the execution key.
    assert result.provider_call_attempts[0].logical_call_id == (
        "llm:" + encode_rollout_execution_key(key)
    )


def test_work_key_is_derived_from_the_execution_key(
    clean_pg: str,
    pg_engine: Engine,
) -> None:
    key = execution_key()
    work_key = work_key_for_execution_key(key)
    # Deterministic and platform-key-safe.
    assert work_key.value.startswith("rxk-")
    assert work_key_for_execution_key(key) == work_key
