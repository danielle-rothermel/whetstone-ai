from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import select

from whetstone.platform.contracts import STAGE_EVAL_ROW
from whetstone.platform.pipeline import (
    EVAL_ROW_QUEUE_CONCURRENCY,
    register_optim_pipeline,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from dr_platform.pipeline.registry import PipelineRegistry
    from dr_platform.runtime.dispatcher import DispatcherRegistration
    from whetstone.coordination.runtime_bootstrap import RegisteredRuntime


def _require_platform_extra() -> None:
    try:
        import dr_platform  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Platform runtime requires the optional platform extra: "
            "pip install 'whetstone-ai[platform]'"
        ) from exc


def upgrade_platform_schema(database_url: str) -> None:
    _require_platform_extra()
    from dr_platform.runtime.database.migrate import (
        upgrade_platform_schema as _upgrade,
    )

    _upgrade(database_url)


def queue_concurrency(stage_key: str) -> int:
    if stage_key == STAGE_EVAL_ROW:
        return EVAL_ROW_QUEUE_CONCURRENCY
    return 1


@dataclass(frozen=True, slots=True)
class PlatformDeployment:
    registry: PipelineRegistry
    engine: Engine
    registration: DispatcherRegistration

    def shutdown(self) -> None:
        from dbos import DBOS

        self.registration.close()
        DBOS.destroy(destroy_registry=True)


def deploy_platform(
    *,
    runtime: RegisteredRuntime,
    engine: Engine,
    database_url: str,
    application_version: str,
    executor_id: str,
    now: datetime,
    app_name: str,
    max_recovery_attempts: int = 1,
    sweep_cron: str | None = None,
    system_database_url: str | None = None,
) -> PlatformDeployment:
    """Register the optim pipeline, queues, dispatcher, and launch DBOS."""
    _require_platform_extra()
    from dbos import DBOS, Queue
    from dr_platform.admission.controls import set_stage_capacity
    from dr_platform.pipeline.registry import PipelineRegistry
    from dr_platform.recovery.live_identity import LiveDbosIdentity
    from dr_platform.runtime.dbos import (
        DEFAULT_POOL_SIZE,
        PlatformDbosConfig,
        initialize_dbos_runtime,
    )
    from dr_platform.runtime.dispatcher import register_scheduled_dispatcher

    if runtime.ledger_engine is None:
        raise ValueError("deploy_platform requires a ledger engine")
    upgrade_platform_schema(database_url)
    registry = PipelineRegistry()
    pipeline = register_optim_pipeline(
        registry,
        runtime,
        max_recovery_attempts=max_recovery_attempts,
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
            engine=engine,
            clock=lambda: now,
        )
    assert pipeline.run_completion is not None
    Queue(pipeline.run_completion.queue_name, concurrency=1)

    resolved_system_url = system_database_url or database_url
    config = PlatformDbosConfig(
        database_url=database_url,
        system_database_url=resolved_system_url,
        max_recovery_attempts=max_recovery_attempts,
        pool_size=DEFAULT_POOL_SIZE,
        application_version=application_version,
        executor_id=executor_id,
    )
    try:
        initialize_dbos_runtime(config, app_name=app_name)
        registration = register_scheduled_dispatcher(
            live_dbos_identity=LiveDbosIdentity(),
            config=config,
            engine=engine,
            registry=registry,
            sweep_cron=sweep_cron,
        )
        DBOS.launch()
        # Pin this process as the current deployment. Without this, workflows
        # left in a reused Postgres database by a *different* prior application
        # version are treated as recoverable work and stall the run.
        DBOS.set_latest_application_version(DBOS.application_version)
    except Exception:
        DBOS.destroy(destroy_registry=True)
        raise
    return PlatformDeployment(
        registry=registry,
        engine=engine,
        registration=registration,
    )


def load_pipeline_run(engine: Engine, *, run_key: str):
    _require_platform_extra()
    from dr_platform.submission.runs import get_pipeline_run

    with engine.connect() as connection:
        record = get_pipeline_run(connection, run_key=run_key)
    if record is None:
        raise LookupError(f"pipeline run does not exist: {run_key!r}")
    return record


def wait_for_run_released(
    engine: Engine,
    *,
    run_key: str,
    timeout: float = 120,
) -> None:
    _require_platform_extra()
    from dr_platform._core.ledger.schema import LedgerSchema

    schema = LedgerSchema()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with engine.connect() as connection:
            released_at = connection.execute(
                select(schema.pipeline_runs.c.released_at).where(
                    schema.pipeline_runs.c.run_key == run_key
                )
            ).scalar_one_or_none()
        if released_at is not None:
            return
        time.sleep(0.05)
    raise TimeoutError(f"run {run_key!r} was not released before deadline")


def _stage_workflow_ids(
    engine: Engine,
    *,
    run_key: str | None = None,
) -> tuple[str, ...]:
    from dr_platform._core.ledger.schema import LedgerSchema

    schema = LedgerSchema()
    statement = select(schema.stage_attempts.c.workflow_id).order_by(
        schema.stage_attempts.c.stage_attempt_id
    )
    if run_key is not None:
        statement = (
            statement.select_from(
                schema.stage_attempts.join(
                    schema.stage_executions,
                    schema.stage_attempts.c.stage_execution_id
                    == schema.stage_executions.c.stage_execution_id,
                ).join(
                    schema.work_items,
                    schema.stage_executions.c.work_item_id
                    == schema.work_items.c.work_item_id,
                )
            ).where(schema.work_items.c.origin_run_key == run_key)
        )
    with engine.connect() as connection:
        return tuple(connection.execute(statement).scalars())


def _await_dbos_result(workflow_id: str, *, registration, timeout: float = 60):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            registration.client.retrieve_workflow(workflow_id).get_result,
            polling_interval_sec=0.01,
        )
        return future.result(timeout=timeout)


def drive_until_quiescent(
    *,
    engine: Engine,
    registry: PipelineRegistry,
    registration,
    now: datetime,
    deadline_seconds: float = 120,
    run_key: str | None = None,
) -> None:
    _require_platform_extra()
    from dr_platform._core.ledger.schema import LedgerSchema
    from dr_platform._core.ledger.states import StageExecutionState
    from dr_platform.admission.runner import run_admission_pass

    seen: set[str] = set()
    deadline = time.monotonic() + deadline_seconds
    schema = LedgerSchema()
    idle_passes = 0
    pending_states = (
        StageExecutionState.READY.value,
        StageExecutionState.ADMITTED.value,
    )
    while time.monotonic() < deadline:
        registration.workflow(now, now)
        processed_new = False
        for workflow_id in _stage_workflow_ids(engine, run_key=run_key):
            if workflow_id in seen:
                continue
            seen.add(workflow_id)
            processed_new = True
            _await_dbos_result(workflow_id, registration=registration)
        run_admission_pass(
            engine,
            client=registration.client,
            registry=registry,
            clock=lambda: now,
        )
        pending = select(schema.stage_executions.c.stage_execution_id).where(
            schema.stage_executions.c.state.in_(pending_states)
        )
        if run_key is not None:
            pending = pending.join(
                schema.work_items,
                schema.stage_executions.c.work_item_id
                == schema.work_items.c.work_item_id,
            ).where(schema.work_items.c.origin_run_key == run_key)
        with engine.connect() as connection:
            pending_stage = connection.execute(pending).first()
        if pending_stage is not None:
            idle_passes = 0
            continue
        if processed_new:
            idle_passes = 0
            continue
        idle_passes += 1
        if idle_passes >= 3:
            return
        time.sleep(0.05)
    raise TimeoutError("platform run did not quiesce before deadline")


def await_run_completion(
    *,
    run_key: str,
    engine: Engine,
    registration,
    registry: PipelineRegistry,
    now: datetime,
    timeout: float = 120,
) -> str:
    _require_platform_extra()
    from dr_platform.admission.runner import run_admission_pass
    from dr_platform.completion.execution import inspect_run_completion

    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        registration.barrier_workflow(now, now)
        run_admission_pass(
            engine,
            client=registration.client,
            registry=registry,
            clock=lambda: now,
        )
        for workflow_id in _stage_workflow_ids(engine, run_key=run_key):
            _await_dbos_result(workflow_id, registration=registration)
        try:
            wait_for_run_released(
                engine,
                run_key=run_key,
                timeout=min(5.0, deadline - time.monotonic()),
            )
            execution = inspect_run_completion(run_key, engine=engine)
            terminal_result_ref = _await_dbos_result(
                execution.workflow_id,
                registration=registration,
            )
            completed = inspect_run_completion(run_key, engine=engine)
            assert completed.output_reference == terminal_result_ref
            return terminal_result_ref
        except (LookupError, TimeoutError, AssertionError) as error:
            last_error = error
            time.sleep(0.05)
    raise TimeoutError(
        f"run completion for {run_key!r} did not finish before deadline"
    ) from last_error


__all__ = [
    "PlatformDeployment",
    "await_run_completion",
    "deploy_platform",
    "drive_until_quiescent",
    "load_pipeline_run",
    "queue_concurrency",
    "upgrade_platform_schema",
    "wait_for_run_released",
]
