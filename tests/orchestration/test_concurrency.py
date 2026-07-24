"""Provider Concurrency Control configuration and admission.

Proves Whetstone configures the mandatory default empty-selector capacity
plus one exact-label capacity per Provider Quota Identity; that admission
applies all matching controls together and holds capacity for the full
ADMITTED Stage lifetime; and that a missing default leaves work
READY/unadmitted even when provider-specific controls exist.
"""

from __future__ import annotations

from dr_platform.staging.admission import run_admission_pass
from dr_platform.staging.states import StageExecutionState

from whetstone.orchestration import (
    ROLLOUT_EXECUTION_STAGE_KEY,
    QuotaCapacity,
    configure_provider_concurrency,
    orchestration_pipeline_identity,
    quota_label_value,
)

from .platform_support import (
    NOW,
    RecordingClient,
    _as_client,
    migrate,
    register_pipeline,
    stage_execution_id_for,
    submit_one,
)
from .support import (
    build_harness,
    execution_key,
    quota,
    response_outcome,
)


def _response():
    return response_outcome(text="ok")


def _state(engine, *, execution_key, schema) -> StageExecutionState:
    from dr_platform.staging.stage_executions import get_stage_execution

    stage_id = stage_execution_id_for(
        engine, execution_key=execution_key, schema=schema
    )
    with engine.connect() as connection:
        execution = get_stage_execution(
            connection, stage_execution_id=stage_id, schema=schema
        )
    assert execution is not None
    return execution.state


def _submit_item(engine, registry, *, key, quotas, run_key, campaign_key):
    input_ref = build_harness(outcomes=[_response()], key=key).input_ref
    submit_one(
        engine,
        registry,
        execution_key=key,
        input_ref=input_ref,
        quotas=quotas,
        run_key=run_key,
        campaign_key=campaign_key,
    )


def test_configuration_writes_default_and_per_quota_capacities(
    pg_engine,
) -> None:
    migrate(pg_engine)
    encoder = quota(model="encoder-model")
    decoder = quota(model="decoder-model")
    config = configure_provider_concurrency(
        engine=pg_engine,
        default_capacity=8,
        quota_capacities=(
            QuotaCapacity(quota=encoder, capacity=2),
            QuotaCapacity(quota=decoder, capacity=3),
        ),
        clock=lambda: NOW,
    )
    # The mandatory default empty-selector control exists.
    assert config.default_control.selector == {}
    assert config.default_control.capacity == 8
    # One exact-label control per quota, keyed by its collision-free value.
    assert set(config.per_quota) == {
        quota_label_value(encoder),
        quota_label_value(decoder),
    }
    assert config.per_quota[quota_label_value(encoder)].capacity == 2
    assert config.per_quota[quota_label_value(decoder)].capacity == 3


def test_missing_default_leaves_work_ready_and_unadmitted(
    pg_engine,
) -> None:
    """Without the mandatory default, work stays READY despite a quota control.

    Configure only a provider-specific exact-label control (no default). The
    admission pass reports the stage unconfigured and leaves the matching work
    READY/unadmitted.
    """
    schema = migrate(pg_engine)
    registry = register_pipeline()
    route = quota(model="only-route")
    key = execution_key()

    # A per-label control but NO default empty-selector control.
    from dr_platform.staging.operations import set_selector_capacity

    from whetstone.orchestration import quota_selector

    set_selector_capacity(
        pipeline=orchestration_pipeline_identity(),
        stage_key=ROLLOUT_EXECUTION_STAGE_KEY,
        labels=dict(quota_selector(route)),
        capacity=5,
        engine=pg_engine,
        clock=lambda: NOW,
    )
    _submit_item(
        pg_engine,
        registry,
        key=key,
        quotas=(route,),
        run_key="run-1",
        campaign_key="camp-1",
    )
    client = RecordingClient()
    summary = run_admission_pass(
        pg_engine,
        client=_as_client(client),
        registry=registry,
        clock=lambda: NOW,
    )
    assert summary.admitted_total == 0
    assert summary.unconfigured_stages  # reported unconfigured
    assert not client.enqueued
    assert (
        _state(pg_engine, execution_key=key, schema=schema)
        is StageExecutionState.READY
    )


def test_all_matching_controls_apply_together(pg_engine) -> None:
    """A tight per-quota control caps admission even under a generous default.

    default capacity 10; the route's per-label capacity is 1. Two items share
    the route. Admission admits at most one (the tighter matching control
    wins), proving all matching controls apply together, not just the default.
    """
    migrate(pg_engine)
    registry = register_pipeline()
    route = quota(model="tight-route")
    configure_provider_concurrency(
        engine=pg_engine,
        default_capacity=10,
        quota_capacities=(QuotaCapacity(quota=route, capacity=1),),
        clock=lambda: NOW,
    )
    key_a = execution_key(task_identity="task-a")
    key_b = execution_key(task_identity="task-b")
    for key in (key_a, key_b):
        _submit_item(
            pg_engine,
            registry,
            key=key,
            quotas=(route,),
            run_key="run-1",
            campaign_key="camp-1",
        )
    summary = admit_and_summary(pg_engine, registry)
    # Only one admitted: the per-label capacity of 1 caps the shared route.
    assert summary.admitted_total == 1
    assert summary.skipped_for_capacity >= 1


def test_multi_route_work_matches_every_control(pg_engine) -> None:
    """Multi-route work carries every label; each matching control applies.

    Work routing through two quotas carries both labels. With the encoder
    capacity at 0, that single control alone blocks admission (all matching
    controls must be satisfied), even though the default and decoder controls
    have room.
    """
    schema = migrate(pg_engine)
    registry = register_pipeline()
    encoder = quota(model="encoder")
    decoder = quota(model="decoder")
    configure_provider_concurrency(
        engine=pg_engine,
        default_capacity=10,
        quota_capacities=(
            QuotaCapacity(quota=encoder, capacity=0),
            QuotaCapacity(quota=decoder, capacity=10),
        ),
        clock=lambda: NOW,
    )
    key = execution_key()
    _submit_item(
        pg_engine,
        registry,
        key=key,
        quotas=(encoder, decoder),
        run_key="run-1",
        campaign_key="camp-1",
    )
    summary = admit_and_summary(pg_engine, registry)
    assert summary.admitted_total == 0
    assert summary.skipped_for_capacity >= 1
    assert (
        _state(pg_engine, execution_key=key, schema=schema)
        is StageExecutionState.READY
    )


def admit_and_summary(engine, registry):
    client = RecordingClient()
    return run_admission_pass(
        engine,
        client=_as_client(client),
        registry=registry,
        clock=lambda: NOW,
    )
