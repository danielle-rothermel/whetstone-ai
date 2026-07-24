"""Platform-state helpers for the orchestration DB tests.

Drives real dr-platform staging state via the same seams dr-platform's own
tests use: Alembic migration, the concrete Whetstone Orchestration Pipeline,
``submit`` + ``run_admission_pass`` with a recording DBOS client (no launched
runtime), and the transition/terminal leaf ops to land a stage FAILED.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast

from dr_platform.db.migrate import upgrade_platform_schema
from dr_platform.staging import PipelineRegistry
from dr_platform.staging.admission import run_admission_pass
from dr_platform.staging.schema import StagingSchema
from dr_platform.staging.stage_attempts import record_stage_attempt_terminal
from dr_platform.staging.stage_executions import (
    get_stage_execution_for_work,
    transition_stage_execution,
)
from dr_platform.staging.states import StageExecutionState
from dr_platform.staging.submission import submit
from sqlalchemy import select

from whetstone.orchestration import (
    ROLLOUT_EXECUTION_STAGE_KEY,
    configure_provider_concurrency,
    orchestration_pipeline,
    orchestration_pipeline_identity,
    rollout_work_input,
    work_key_for_execution_key,
)

if TYPE_CHECKING:
    from dbos import DBOSClient, EnqueueOptions
    from dr_providers import ProviderQuotaIdentity
    from sqlalchemy import Connection, Engine

    from whetstone.graph.rollout import RolloutExecutionKey

NOW = datetime(2026, 7, 22, tzinfo=UTC)


def engine_dsn(engine: Engine) -> str:
    return engine.url.render_as_string(hide_password=False)


def migrate(engine: Engine) -> StagingSchema:
    upgrade_platform_schema(engine_dsn(engine))
    return StagingSchema()


class RecordingClient:
    """A DBOS client stand-in that records enqueues instead of running them."""

    def __init__(self) -> None:
        self.enqueued: list[EnqueueOptions] = []

    def enqueue_in_transaction(
        self,
        _connection: Connection,
        options: EnqueueOptions,
        *_args: object,
        **_kwargs: object,
    ) -> object:
        self.enqueued.append(cast("EnqueueOptions", dict(options)))
        return object()


def _as_client(client: object) -> DBOSClient:
    return cast("DBOSClient", client)


def register_pipeline() -> PipelineRegistry:
    """Register the concrete Whetstone Orchestration Pipeline (unwrapped body).

    The stage body is a stub here: these DB tests admit/transition rows and do
    not run the durable executor (that is the integration test). The registry
    only needs a wrapped pipeline whose stage identity matches the persisted
    work.
    """
    registry = PipelineRegistry()
    registry.register(orchestration_pipeline(lambda _ref: "unused"))
    return registry


def submit_one(
    engine: Engine,
    registry: PipelineRegistry,
    *,
    execution_key: RolloutExecutionKey,
    input_ref: str,
    quotas: tuple[ProviderQuotaIdentity, ...],
    run_key: str = "run-1",
    campaign_key: str = "camp-1",
) -> None:
    submit(
        campaign_key=campaign_key,
        run_key=run_key,
        pipeline=orchestration_pipeline_identity(),
        execution_config_reference="config:1",
        items=(
            rollout_work_input(
                execution_key=execution_key,
                input_ref=input_ref,
                quotas=quotas,
            ),
        ),
        registry=registry,
        engine=engine,
        clock=lambda: NOW,
    )


def stage_execution_id_for(
    engine: Engine,
    *,
    execution_key: RolloutExecutionKey,
    schema: StagingSchema,
) -> int:
    work_key = work_key_for_execution_key(execution_key)
    items = schema.work_items
    with engine.connect() as connection:
        work_item_id = connection.execute(
            select(items.c.work_item_id).where(
                items.c.work_key == work_key.value
            )
        ).scalar_one()
        execution = get_stage_execution_for_work(
            connection,
            work_item_id=work_item_id,
            stage_key=ROLLOUT_EXECUTION_STAGE_KEY,
            schema=schema,
        )
    assert execution is not None
    return execution.stage_execution_id


def admit(
    engine: Engine,
    registry: PipelineRegistry,
    *,
    client: RecordingClient | None = None,
) -> RecordingClient:
    recording = client or RecordingClient()
    run_admission_pass(
        engine,
        client=_as_client(recording),
        registry=registry,
        clock=lambda: NOW,
    )
    return recording


def drive_to_failed(
    engine: Engine,
    *,
    stage_execution_id: int,
    schema: StagingSchema,
) -> None:
    """Land an ADMITTED stage FAILED with a terminal current attempt.

    Mirrors dr-platform's own retry-test setup so ``retry_stage`` sees exactly
    the state it requires: FAILED with a terminal current attempt.
    """
    with engine.begin() as connection:
        transition_stage_execution(
            connection,
            stage_execution_id=stage_execution_id,
            new_state=StageExecutionState.FAILED,
            updated_at=NOW + timedelta(seconds=1),
            schema=schema,
        )
        record_stage_attempt_terminal(
            connection,
            stage_execution_id=stage_execution_id,
            attempt_number=1,
            terminal_at=NOW + timedelta(seconds=1),
            terminal_summary={"outcome": "failed"},
            schema=schema,
        )


def configure_default_and_quota_capacities(
    engine: Engine,
    *,
    default_capacity: int,
    quota_capacities: tuple,
    stage_key: str = ROLLOUT_EXECUTION_STAGE_KEY,
):
    return configure_provider_concurrency(
        engine=engine,
        default_capacity=default_capacity,
        quota_capacities=quota_capacities,
        stage_key=stage_key,
        clock=lambda: NOW,
    )
