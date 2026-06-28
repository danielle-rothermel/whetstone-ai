from __future__ import annotations

from dr_dspy.dbos_runtime import DbosWorkflowStatus, QueueSelection
from dr_dspy.failure_policy import FailureClass
from dr_dspy.worker_monitor import (
    WorkerMonitorConfig,
    WorkerQueueSnapshot,
    worker_halt_reason,
)


def _config() -> WorkerMonitorConfig:
    return WorkerMonitorConfig(
        database_url="postgresql://app",
        dbos_system_database_url="postgresql://sys",
        experiment_name="exp",
        prediction_table="predictions",
        queue_selection=QueueSelection.BOTH,
        queue_names=("generation", "scoring"),
        interval_seconds=1.0,
        summary_interval_seconds=5.0,
    )


def test_worker_halts_on_new_resource_exhaustion_failure() -> None:
    snapshot = WorkerQueueSnapshot(
        generation_failure_class_counts={
            FailureClass.RESOURCE_EXHAUSTION.value: 1
        }
    )

    reason = worker_halt_reason(
        snapshot,
        config=_config(),
        initial_resource_exhaustion_failures=0,
        initial_success_total=0,
        initial_failure_total=0,
    )

    assert reason == "resource exhaustion failure recorded"


def test_worker_halts_on_high_failure_fraction_after_minimum_events() -> None:
    snapshot = WorkerQueueSnapshot(
        dbos_status_counts={
            DbosWorkflowStatus.SUCCESS.value: 10,
            DbosWorkflowStatus.ERROR.value: 10,
        }
    )

    reason = worker_halt_reason(
        snapshot,
        config=_config(),
        initial_resource_exhaustion_failures=0,
        initial_success_total=0,
        initial_failure_total=0,
    )

    assert reason is not None
    assert "failure fraction" in reason
