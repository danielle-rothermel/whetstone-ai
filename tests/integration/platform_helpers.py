"""Shared helpers for Postgres + DBOS platform integration tests."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from dbos import DBOS, Queue
from sqlalchemy import Engine, select

from dr_platform._core.ledger.schema import LedgerSchema
from dr_platform._core.ledger.states import StageExecutionState
from dr_platform.admission.controls import set_stage_capacity
from dr_platform.admission.runner import run_admission_pass
from dr_platform.completion.execution import inspect_run_completion
from dr_platform.inspection.work_items import (
    get_work_item_stages,
    list_predecessor_stage_outputs,
)
from dr_platform.pipeline.registry import PipelineRegistry
from dr_platform.recovery.live_identity import LiveDbosIdentity
from dr_platform.runtime.database.migrate import upgrade_platform_schema
from dr_platform.runtime.dbos import (
    DEFAULT_POOL_SIZE,
    PlatformDbosConfig,
    initialize_dbos_runtime,
)
from dr_platform.runtime.dispatcher import register_scheduled_dispatcher
from dr_store.content_addressing import parse_object_reference

from whetstone.coordination.eval_service import EvalDispatchMode
from whetstone.coordination.harness_run_controller import OptimRunLaunch
from whetstone.coordination.runtime_bootstrap import (
    RegisteredRuntime,
    build_toy_copro_control,
    prepare_copro_run,
    register_runtime,
)
from whetstone.core.blocking_store import BlockingObjectStore
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.contracts import OPTIM_RESULT_SCHEMA, OptimResult
from whetstone.platform.contracts import STAGE_EVAL_FANIN, STAGE_EVAL_ROW
from whetstone.platform.pipeline import EVAL_ROW_QUEUE_CONCURRENCY, register_optim_pipeline
from whetstone.platform.submit import submit_optim_run

if TYPE_CHECKING:
    from dr_platform.runtime.dispatcher import DispatcherRegistration


NOW = datetime(2026, 7, 17, 12, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class PlatformIntegrationContext:
    suffix: str
    campaign_key: str
    run_key: str
    work_key: str
    runtime: RegisteredRuntime
    launch: OptimRunLaunch
    registry: PipelineRegistry
    registration: DispatcherRegistration
    store: BlockingObjectStore


def migrate_platform_schema(engine: Engine) -> LedgerSchema:
    upgrade_platform_schema(engine.url.render_as_string(hide_password=False))
    return LedgerSchema()


def queue_concurrency(stage_key: str) -> int:
    if stage_key == STAGE_EVAL_ROW:
        return EVAL_ROW_QUEUE_CONCURRENCY
    return 1


def lookup_work_item_id(
    engine: Engine,
    *,
    campaign_key: str,
    work_key: str,
) -> int:
    schema = LedgerSchema()
    with engine.connect() as connection:
        work_item_id = connection.execute(
            select(schema.work_items.c.work_item_id).where(
                schema.work_items.c.campaign_key == campaign_key,
                schema.work_items.c.work_key == work_key,
            )
        ).scalar_one()
    return work_item_id


def succeeded_stage_counts(
    engine: Engine,
    work_item_id: int,
) -> dict[str, int]:
    stages = get_work_item_stages(work_item_id, engine=engine)
    counts: dict[str, int] = {}
    for summary in stages:
        if summary.execution.state is not StageExecutionState.SUCCEEDED:
            continue
        stage_key = summary.execution.stage_key.value
        counts[stage_key] = counts.get(stage_key, 0) + 1
    return counts


def assert_stage_coverage(
    engine: Engine,
    work_item_id: int,
    expected_counts: dict[str, int],
) -> None:
    actual = succeeded_stage_counts(engine, work_item_id)
    for stage_key, expected in expected_counts.items():
        observed = actual.get(stage_key, 0)
        assert observed == expected, (
            f"{stage_key}: expected {expected} SUCCEEDED executions, got {observed}; "
            f"actual={actual}"
        )


def assert_fanin_barrier_predecessors(
    engine: Engine,
    work_item_id: int,
    *,
    fanin_stage_index: int,
    expected_eval_row_count: int,
) -> None:
    stages = get_work_item_stages(work_item_id, engine=engine)
    fanin_executions = [
        summary.execution
        for summary in stages
        if summary.execution.stage_key.value == STAGE_EVAL_FANIN
        and summary.execution.state is StageExecutionState.SUCCEEDED
    ]
    assert fanin_executions, "expected a succeeded eval_fanin execution"
    assert fanin_executions[0].barrier is True
    predecessors = list_predecessor_stage_outputs(
        work_item_id,
        below_stage_index=fanin_stage_index,
        engine=engine,
    )
    eval_row_predecessors = [
        predecessor
        for predecessor in predecessors
        if predecessor.stage_key.value == STAGE_EVAL_ROW
    ]
    assert len(eval_row_predecessors) == expected_eval_row_count


def await_dbos_result(workflow_id: str, *, registration, timeout: float = 60):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            registration.client.retrieve_workflow(workflow_id).get_result,
            polling_interval_sec=0.01,
        )
        return future.result(timeout=timeout)


def stage_workflow_ids(engine: Engine) -> tuple[str, ...]:
    schema = LedgerSchema()
    with engine.connect() as connection:
        return tuple(
            connection.execute(
                select(schema.stage_attempts.c.workflow_id).order_by(
                    schema.stage_attempts.c.stage_attempt_id
                )
            ).scalars()
        )


def run_until_quiescent(
    *,
    pg_engine: Engine,
    registry: PipelineRegistry,
    registration,
    now: datetime,
    deadline_seconds: float = 120,
) -> None:
    seen: set[str] = set()
    deadline = time.monotonic() + deadline_seconds
    schema = LedgerSchema()
    while time.monotonic() < deadline:
        registration.workflow(now, now)
        for workflow_id in stage_workflow_ids(pg_engine):
            if workflow_id in seen:
                continue
            seen.add(workflow_id)
            await_dbos_result(workflow_id, registration=registration)
        run_admission_pass(
            pg_engine,
            client=registration.client,
            registry=registry,
            clock=lambda: now,
        )
        with pg_engine.connect() as connection:
            pending_stage = connection.execute(
                select(schema.stage_executions.c.stage_execution_id).where(
                    schema.stage_executions.c.state.in_(
                        ("READY", "ADMITTED", "ENQUEUED")
                    )
                )
            ).first()
        if pending_stage is None:
            return
        time.sleep(0.05)
    raise TimeoutError("platform integration run did not quiesce before deadline")


def await_run_completion(
    *,
    run_key: str,
    pg_engine: Engine,
    registration,
    now: datetime,
) -> str:
    registration.barrier_workflow(now, now)
    execution = inspect_run_completion(run_key, engine=pg_engine)
    terminal_result_ref = await_dbos_result(
        execution.workflow_id,
        registration=registration,
    )
    completed = inspect_run_completion(run_key, engine=pg_engine)
    assert completed.output_reference == terminal_result_ref
    return terminal_result_ref


def bootstrap_platform_runtime(
    *,
    store: BlockingObjectStore,
    pg_engine: Engine,
    clean_pg: str,
    now: datetime,
    dispatch_mode: EvalDispatchMode,
    breadth: int = 2,
    depth: int = 1,
) -> PlatformIntegrationContext:
    migrate_platform_schema(pg_engine)
    suffix = uuid4().hex[:10]
    campaign_key = f"campaign-{suffix}"
    run_key = f"run-{suffix}"
    work_key = f"work-{suffix}"

    runtime = register_runtime(store=store, ledger_engine=pg_engine)
    registry = PipelineRegistry()
    pipeline = register_optim_pipeline(
        registry,
        runtime,
        max_recovery_attempts=1,
    )
    for stage in pipeline.stages:
        Queue(
            stage.queue_name,
            concurrency=queue_concurrency(stage.key.value),
        )
        set_stage_capacity(
            pipeline=pipeline.identity,
            stage_key=stage.key,
            capacity=queue_concurrency(stage.key.value),
            engine=pg_engine,
            clock=lambda: now,
        )
    assert pipeline.run_completion is not None
    Queue(pipeline.run_completion.queue_name, concurrency=1)

    eval_engine = ReferenceEvalRuntimeConfig().build_engine(store)
    control = build_toy_copro_control(breadth=breadth, depth=depth, engine=eval_engine)
    launch = prepare_copro_run(
        runtime,
        run_id=f"integration-run-{suffix}",
        control=control,
        terminal_top_k=1,
    )
    submit_optim_run(
        runtime=runtime,
        registry=registry,
        engine=pg_engine,
        campaign_key=campaign_key,
        run_key=run_key,
        work_key=work_key,
        launch=launch,
        controller_identity_hash=runtime.controller.runtime_hash,
        execution_config_reference=f"exec-config-{suffix}",
        dispatch_mode=dispatch_mode,
    )

    config = PlatformDbosConfig(
        database_url=clean_pg,
        system_database_url=clean_pg,
        max_recovery_attempts=1,
        pool_size=DEFAULT_POOL_SIZE,
    )
    initialize_dbos_runtime(config, app_name=f"whetstone-platform-{suffix}")
    registration = register_scheduled_dispatcher(
        live_dbos_identity=LiveDbosIdentity(
            app_version=f"whetstone-{suffix}",
            resolve_executor_ids=lambda: frozenset({DBOS.application_version}),
        ),
        config=config,
        engine=pg_engine,
        registry=registry,
        sweep_cron=None,
    )
    DBOS.launch()
    DBOS.set_latest_application_version(DBOS.application_version)

    return PlatformIntegrationContext(
        suffix=suffix,
        campaign_key=campaign_key,
        run_key=run_key,
        work_key=work_key,
        runtime=runtime,
        launch=launch,
        registry=registry,
        registration=registration,
        store=store,
    )


def load_terminal_optim_result(
    context: PlatformIntegrationContext,
    terminal_result_ref: str,
) -> OptimResult:
    parsed = parse_object_reference(terminal_result_ref)
    assert parsed.schema == OPTIM_RESULT_SCHEMA
    result = OptimResult.model_validate(context.store.get(parsed))
    assert result.run.record.run_id == context.launch.run.run_id
    assert result.proposals
    return result


def shutdown_platform_runtime(context: PlatformIntegrationContext) -> None:
    context.registration.close()
    DBOS.destroy(destroy_registry=True)
