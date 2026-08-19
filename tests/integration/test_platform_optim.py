"""Tier 2 platform integration tests (Postgres + DBOS + dr-platform).

Run locally:
    uv sync --extra platform
    WHETSTONE_PLATFORM_INTEGRATION=1 uv run pytest -m integration tests/integration/
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from dbos import DBOS, Queue
from sqlalchemy import Engine, select

from dr_platform._core.ledger.schema import LedgerSchema
from dr_platform.admission.controls import set_stage_capacity
from dr_platform.admission.runner import run_admission_pass
from dr_platform.completion.execution import inspect_run_completion
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

from whetstone.core.blocking_store import open_blocking_sqlite_store
from whetstone.coordination.runtime_bootstrap import (
    build_toy_copro_control,
    prepare_copro_run,
    register_runtime,
)
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.contracts import OPTIM_RESULT_SCHEMA, OptimResult
from whetstone.platform.contracts import STAGE_EVAL_ROW
from whetstone.platform.pipeline import EVAL_ROW_QUEUE_CONCURRENCY, register_optim_pipeline
from whetstone.platform.submit import submit_optim_run

pytestmark = pytest.mark.integration

NOW = datetime(2026, 7, 17, 12, tzinfo=UTC)


def _migrate_platform_schema(engine: Engine) -> LedgerSchema:
    upgrade_platform_schema(engine.url.render_as_string(hide_password=False))
    return LedgerSchema()


def _await_dbos_result(workflow_id: str, *, registration, timeout: float = 60):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            registration.client.retrieve_workflow(workflow_id).get_result,
            polling_interval_sec=0.01,
        )
        return future.result(timeout=timeout)


def _stage_workflow_ids(engine: Engine) -> tuple[str, ...]:
    schema = LedgerSchema()
    with engine.connect() as connection:
        return tuple(
            connection.execute(
                select(schema.stage_attempts.c.workflow_id).order_by(
                    schema.stage_attempts.c.stage_attempt_id
                )
            ).scalars()
        )


def _queue_concurrency(stage_key: str) -> int:
    if stage_key == STAGE_EVAL_ROW:
        return EVAL_ROW_QUEUE_CONCURRENCY
    return 1


@pytest.mark.skipif(
    os.environ.get("WHETSTONE_PLATFORM_INTEGRATION") != "1",
    reason="set WHETSTONE_PLATFORM_INTEGRATION=1 with Postgres+DBOS configured",
)
def test_inline_platform_copro_submit_to_result(
    pg_engine: Engine,
    clean_pg: str,
    tmp_path,
) -> None:
    """Full submit → admission → DBOS stages → run completion → OptimResult."""
    _migrate_platform_schema(pg_engine)
    suffix = uuid4().hex[:10]
    store_path = tmp_path / "integration-store.sqlite"
    registration = None
    run_key = f"run-{suffix}"

    with open_blocking_sqlite_store(str(store_path)) as store:
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
                concurrency=_queue_concurrency(stage.key.value),
            )
            set_stage_capacity(
                pipeline=pipeline.identity,
                stage_key=stage.key,
                capacity=_queue_concurrency(stage.key.value),
                engine=pg_engine,
                clock=lambda: NOW,
            )
        assert pipeline.run_completion is not None
        Queue(pipeline.run_completion.queue_name, concurrency=1)

        eval_engine = ReferenceEvalRuntimeConfig().build_engine(store)
        control = build_toy_copro_control(breadth=2, depth=1, engine=eval_engine)
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
            campaign_key=f"campaign-{suffix}",
            run_key=run_key,
            work_key=f"work-{suffix}",
            launch=launch,
            controller_identity_hash=runtime.controller.runtime_hash,
            execution_config_reference=f"exec-config-{suffix}",
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

        seen: set[str] = set()
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            registration.workflow(NOW, NOW)
            for workflow_id in _stage_workflow_ids(pg_engine):
                if workflow_id in seen:
                    continue
                seen.add(workflow_id)
                _await_dbos_result(workflow_id, registration=registration)
            run_admission_pass(
                pg_engine,
                client=registration.client,
                registry=registry,
                clock=lambda: NOW,
            )
            schema = LedgerSchema()
            with pg_engine.connect() as connection:
                pending_stage = connection.execute(
                    select(schema.stage_executions.c.stage_execution_id).where(
                        schema.stage_executions.c.state.in_(
                            ("READY", "ADMITTED", "ENQUEUED")
                        )
                    )
                ).first()
            if pending_stage is None:
                break
            time.sleep(0.05)

        registration.barrier_workflow(NOW, NOW)
        execution = inspect_run_completion(run_key, engine=pg_engine)
        terminal_result_ref = _await_dbos_result(
            execution.workflow_id,
            registration=registration,
        )
        completed = inspect_run_completion(run_key, engine=pg_engine)
        assert completed.output_reference == terminal_result_ref

        parsed = parse_object_reference(terminal_result_ref)
        assert parsed.schema == OPTIM_RESULT_SCHEMA
        result = OptimResult.model_validate(store.get(parsed))
        assert result.run.record.run_id == launch.run.run_id
        assert result.proposals

    if registration is not None:
        registration.close()
    DBOS.destroy(destroy_registry=True)
