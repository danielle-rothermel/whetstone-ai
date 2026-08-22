"""Shared helpers for Postgres + DBOS platform integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from sqlalchemy import Engine, select

from dr_platform._core.ledger.schema import LedgerSchema
from dr_platform._core.ledger.states import StageExecutionState
from dr_platform.inspection.work_items import (
    get_work_item_stages,
    list_predecessor_stage_outputs,
)
from dr_platform.pipeline.registry import PipelineRegistry
from dr_store.content_addressing import parse_object_reference

from whetstone.coordination.eval_service import EvalDispatchMode
from whetstone.coordination.harness_run_controller import OptimRunLaunch
from whetstone.coordination.runtime_bootstrap import RegisteredRuntime
from dr_store.sync import BlockingObjectStore
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.contracts import OPTIM_RESULT_SCHEMA, OptimResult
from whetstone.platform.contracts import (
    STAGE_EVAL_FANIN,
    STAGE_EVAL_ROW,
    load_run_result,
)
from whetstone.platform.deploy import (
    PlatformDeployment,
    await_run_completion as _await_run_completion,
    deploy_platform,
    drive_until_quiescent,
    upgrade_platform_schema,
    wait_for_run_released,
)
from whetstone.platform.submit import OptimRunMemberSpec, submit_optim_run
from whetstone.optim.gepa.harness_adapter import GEPA_ADAPTER_KEY
from whetstone.testing.runtime import (
    build_toy_copro_control,
    build_toy_gepa_adapter,
    build_toy_gepa_control,
    prepare_toy_copro_run,
    prepare_toy_gepa_run,
    register_toy_runtime,
)
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    build_toy_experiment,
)

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
    deployment: PlatformDeployment


def migrate_platform_schema(engine: Engine) -> LedgerSchema:
    upgrade_platform_schema(engine.url.render_as_string(hide_password=False))
    return LedgerSchema()


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
    deferral_origin_stage_index: int,
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
        stage_key=STAGE_EVAL_ROW,
        min_stage_index=deferral_origin_stage_index,
    )
    assert len(predecessors) == expected_eval_row_count


def diagnose_work_item(engine: Engine, work_item_id: int) -> str:
    stages = get_work_item_stages(work_item_id, engine=engine)
    counts: dict[str, dict[str, int]] = {}
    for summary in stages:
        stage_key = summary.execution.stage_key.value
        state = summary.execution.state.value
        counts.setdefault(stage_key, {})
        counts[stage_key][state] = counts[stage_key].get(state, 0) + 1
    return f"work_item_id={work_item_id} stage_counts={counts}"


def assert_no_failed_stages(engine: Engine, work_item_id: int) -> None:
    stages = get_work_item_stages(work_item_id, engine=engine)
    failed = [
        summary.execution
        for summary in stages
        if summary.execution.state is StageExecutionState.FAILED
    ]
    if failed:
        details = ", ".join(
            f"{execution.stage_key.value}@{execution.stage_index}"
            for execution in failed
        )
        raise AssertionError(
            f"platform integration run has FAILED stages: {details}; "
            f"{diagnose_work_item(engine, work_item_id)}"
        )


def run_until_quiescent(
    *,
    pg_engine: Engine,
    registry: PipelineRegistry,
    registration,
    now: datetime,
    deadline_seconds: float = 120,
    work_item_id: int | None = None,
    run_key: str | None = None,
) -> None:
    drive_until_quiescent(
        engine=pg_engine,
        registry=registry,
        registration=registration,
        now=now,
        deadline_seconds=deadline_seconds,
        run_key=run_key,
    )
    if work_item_id is not None:
        assert_no_failed_stages(pg_engine, work_item_id)


def await_run_completion(
    *,
    run_key: str,
    pg_engine: Engine,
    registration,
    registry: PipelineRegistry,
    now: datetime,
    timeout: float = 120,
) -> str:
    return _await_run_completion(
        run_key=run_key,
        engine=pg_engine,
        registration=registration,
        registry=registry,
        now=now,
        timeout=timeout,
    )


def bootstrap_platform_runtime(
    *,
    store: BlockingObjectStore,
    pg_engine: Engine,
    clean_pg: str,
    now: datetime,
    dispatch_mode: EvalDispatchMode,
    breadth: int = 2,
    depth: int = 1,
    optimizer: Literal["copro", "gepa"] = "copro",
    max_metric_calls: int = 2,
) -> PlatformIntegrationContext:
    migrate_platform_schema(pg_engine)
    suffix = uuid4().hex[:10]
    campaign_key = f"campaign-{suffix}"
    run_key = f"run-{suffix}"
    work_key = f"work-{suffix}"

    eval_engine = ReferenceEvalRuntimeConfig().build_engine(store)
    run_id = f"integration-run-{suffix}"
    if optimizer == "gepa":
        experiment = build_toy_experiment(num_seeds=1)
        gepa_control = build_toy_gepa_control(
            engine=eval_engine,
            max_metric_calls=max_metric_calls,
        )
        gepa_adapter = build_toy_gepa_adapter(
            store=store,
            engine=eval_engine,
            control=gepa_control,
            run_id=run_id,
            initial_candidate=experiment.initial_candidate,
            mutation_field=TOY_MUTATION_FIELD,
            evaluation_service=None,
        )
        copro_control = build_toy_copro_control(
            breadth=breadth,
            depth=depth,
            engine=eval_engine,
        )
        runtime = register_toy_runtime(
            store=store,
            ledger_engine=pg_engine,
            engine=eval_engine,
            copro_control=copro_control,
            extra_adapters={GEPA_ADAPTER_KEY: gepa_adapter},
            platform=True,
        )
        launch = prepare_toy_gepa_run(
            runtime,
            run_id=run_id,
            control=gepa_control,
            experiment=experiment,
        )
        if dispatch_mode is EvalDispatchMode.PLATFORM:
            gepa_adapter.bind_evaluation_service(runtime.eval_service)
    else:
        control = build_toy_copro_control(
            breadth=breadth,
            depth=depth,
            engine=eval_engine,
        )
        runtime = register_toy_runtime(
            store=store,
            ledger_engine=pg_engine,
            engine=eval_engine,
            copro_control=control,
            platform=True,
        )
        launch = prepare_toy_copro_run(
            runtime,
            run_id=run_id,
            control=control,
            terminal_top_k=1,
        )
    deployment = deploy_platform(
        runtime=runtime,
        engine=pg_engine,
        database_url=clean_pg,
        application_version=f"whetstone-test-{suffix}",
        executor_id=f"whetstone-test-exec-{suffix}",
        now=now,
        app_name=f"whetstone-platform-{suffix}",
        sweep_cron=None,
    )
    submit_optim_run(
        runtime=runtime,
        registry=deployment.registry,
        engine=pg_engine,
        campaign_key=campaign_key,
        run_key=run_key,
        members=(OptimRunMemberSpec(work_key=work_key, launch=launch),),
        controller_identity_hash=runtime.controller.runtime_hash,
        execution_config_reference=f"exec-config-{suffix}",
        dispatch_mode=dispatch_mode,
    )

    return PlatformIntegrationContext(
        suffix=suffix,
        campaign_key=campaign_key,
        run_key=run_key,
        work_key=work_key,
        runtime=runtime,
        launch=launch,
        registry=deployment.registry,
        registration=deployment.registration,
        store=store,
        deployment=deployment,
    )


def load_terminal_optim_result(
    context: PlatformIntegrationContext,
    terminal_result_ref: str,
) -> OptimResult:
    run_result = load_run_result(context.store, terminal_result_ref)
    assert run_result.platform_run_key == context.run_key
    assert len(run_result.member_results) == 1
    member = run_result.member_results[0]
    assert member.work_key == context.work_key
    parsed = parse_object_reference(member.result_reference)
    assert parsed.schema == OPTIM_RESULT_SCHEMA
    result = OptimResult.model_validate(context.store.get(parsed))
    assert result.run.record.run_id == context.launch.run.run_id
    assert result.proposals or result.seed_retained
    return result


def shutdown_platform_runtime(context: PlatformIntegrationContext) -> None:
    context.deployment.shutdown()
    context.runtime.close()
