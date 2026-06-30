from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from dr_dspy.graph import GraphSpec
from dr_dspy.platform import queue_worker, submission
from dr_dspy.records import (
    ENQUEUE_CLAIM_ID_METADATA_KEY,
    ENQUEUE_CLAIMED_AT_METADATA_KEY,
    BatchSubmitItemEnqueueStatus,
    BatchSubmitItemInsertStatus,
    BatchSubmitItemRecord,
    BatchSubmitOperationRecord,
    BatchSubmitOperationStatus,
    PredictionSpecRecord,
    stable_generation_run_id,
)
from tests.support.platform_workflow_fixtures import (
    NOW,
    direct_node,
    prediction_spec,
)
from tests.support.postgres_fixtures import (
    seed_batch_submit_item,
    seed_batch_submit_operation,
    seed_experiment,
    seed_prediction_spec,
)

pytestmark = pytest.mark.integration


def _batch_item(
    *,
    operation_key: str,
    spec: PredictionSpecRecord,
    item_index: int,
    enqueue_status: BatchSubmitItemEnqueueStatus,
    enqueue_metadata: dict[str, str] | None = None,
) -> BatchSubmitItemRecord:
    return BatchSubmitItemRecord(
        batch_submit_item_id=submission.batch_submit_item_id(
            operation_key=operation_key,
            prediction_id=spec.prediction_id,
        ),
        operation_key=operation_key,
        item_index=item_index,
        prediction_id=spec.prediction_id,
        fair_order_key=spec.fair_order_key,
        insert_status=BatchSubmitItemInsertStatus.INSERTED,
        enqueue_status=enqueue_status,
        enqueue_metadata=enqueue_metadata or {},
        created_at=NOW,
    )


def test_update_operation_summary_completes_when_all_already_scheduled(
    app_postgres_schema,
) -> None:
    graph = GraphSpec(nodes=(direct_node(),), terminal_node_id="direct")
    specs = tuple(
        prediction_spec(graph, task_id=f"HumanEval/{index}")
        for index in range(2)
    )
    operation_key = "integration-already-scheduled"
    engine = create_engine(app_postgres_schema.database_url)
    try:
        with engine.begin() as connection:
            seed_experiment(connection, experiment_name="exp")
            for spec in specs:
                seed_prediction_spec(
                    connection,
                    spec,
                    seed_experiment_row=False,
                )
            seed_batch_submit_operation(
                connection,
                BatchSubmitOperationRecord(
                    operation_key=operation_key,
                    experiment_name="exp",
                    status=BatchSubmitOperationStatus.ENQUEUING,
                    requested_count=2,
                    created_at=NOW,
                ),
            )
            for index, spec in enumerate(specs):
                seed_batch_submit_item(
                    connection,
                    _batch_item(
                        operation_key=operation_key,
                        spec=spec,
                        item_index=index,
                        enqueue_status=(
                            BatchSubmitItemEnqueueStatus.WORKFLOW_ALREADY_PRESENT
                        ),
                        enqueue_metadata={
                            "workflow_id": f"workflow:{spec.prediction_id}",
                            "generation_run_id": stable_generation_run_id(
                                prediction_id=spec.prediction_id,
                                attempt_index=0,
                            ),
                        },
                    ),
                )

        with engine.begin() as connection:
            result = submission.update_operation_summary(
                connection,
                operation_key=operation_key,
                experiment_name="exp",
                queue_name="dr-dspy-platform-generation-v1",
            )

        assert result.requested_count == 2
        assert result.enqueued_count == 0
        assert result.already_scheduled_count == 2
        assert result.failed_count == 0

        with engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT status, already_scheduled_count, completed_at "
                    "FROM dr_dspy_batch_submit_operations "
                    "WHERE operation_key = :operation_key"
                ),
                {"operation_key": operation_key},
            ).one()
        assert row[0] == BatchSubmitOperationStatus.COMPLETED.value
        assert row[1] == 2
        assert row[2] is not None
    finally:
        engine.dispose()


def test_prepare_enqueue_retries_resets_claiming_to_pending(
    app_postgres_schema,
) -> None:
    graph = GraphSpec(nodes=(direct_node(),), terminal_node_id="direct")
    spec = prediction_spec(graph)
    operation_key = "integration-claiming-reset"
    engine = create_engine(app_postgres_schema.database_url)
    try:
        with engine.begin() as connection:
            seed_prediction_spec(connection, spec)
            seed_batch_submit_operation(
                connection,
                BatchSubmitOperationRecord(
                    operation_key=operation_key,
                    experiment_name="exp",
                    status=BatchSubmitOperationStatus.ENQUEUING,
                    requested_count=1,
                    created_at=NOW,
                ),
            )
            seed_batch_submit_item(
                connection,
                _batch_item(
                    operation_key=operation_key,
                    spec=spec,
                    item_index=0,
                    enqueue_status=BatchSubmitItemEnqueueStatus.CLAIMING,
                    enqueue_metadata={
                        ENQUEUE_CLAIM_ID_METADATA_KEY: "stale-claim",
                        ENQUEUE_CLAIMED_AT_METADATA_KEY: NOW.isoformat(),
                    },
                ),
            )

        with engine.begin() as connection:
            submission.prepare_enqueue_retries(
                connection,
                operation_key=operation_key,
            )

        with engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT enqueue_status, enqueue_metadata "
                    "FROM dr_dspy_batch_submit_items "
                    "WHERE operation_key = :operation_key"
                ),
                {"operation_key": operation_key},
            ).one()
        assert row[0] == BatchSubmitItemEnqueueStatus.PENDING.value
        assert row[1] == {}
    finally:
        engine.dispose()


def test_submit_prediction_specs_round_trips_summary(
    app_postgres_schema,
) -> None:
    graph = GraphSpec(nodes=(direct_node(),), terminal_node_id="direct")
    specs = tuple(
        prediction_spec(graph, task_id=f"HumanEval/{index}")
        for index in range(2)
    )
    operation_key = "integration-submit-round-trip"
    engine = create_engine(app_postgres_schema.database_url)
    enqueue_calls: list[str] = []

    def enqueue(
        database_url: str,
        prediction_id: str,
        attempt_index: int,
        queue_name: str,
    ) -> queue_worker.EnqueuedPredictionWorkflow:
        enqueue_calls.append(prediction_id)
        return queue_worker.EnqueuedPredictionWorkflow(
            prediction_id=prediction_id,
            generation_run_id=stable_generation_run_id(
                prediction_id=prediction_id,
                attempt_index=attempt_index,
            ),
            workflow_id=f"workflow:{prediction_id}",
            enqueued=True,
        )

    try:
        result = submission.submit_prediction_specs(
            engine,
            database_url=app_postgres_schema.database_url,
            operation_key=operation_key,
            experiment_name="exp",
            specs=specs,
            chunk_size=2,
            enqueue_workflow=enqueue,
        )

        assert result.requested_count == 2
        assert result.inserted_count == 2
        assert result.enqueued_count == 2
        assert result.failed_count == 0
        assert set(enqueue_calls) == {spec.prediction_id for spec in specs}

        with engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT status, enqueued_count, completed_at "
                    "FROM dr_dspy_batch_submit_operations "
                    "WHERE operation_key = :operation_key"
                ),
                {"operation_key": operation_key},
            ).one()
        assert row[0] == BatchSubmitOperationStatus.COMPLETED.value
        assert row[1] == 2
        assert row[2] is not None
    finally:
        engine.dispose()
