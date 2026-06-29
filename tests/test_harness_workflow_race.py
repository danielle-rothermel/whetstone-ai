from __future__ import annotations

from unittest.mock import patch

import pytest
from dbos._error import (
    DBOSConflictingWorkflowError,
    DBOSQueueDeduplicatedError,
    DBOSWorkflowConflictIDError,
)

from dr_dspy.harness.dbos import (
    WORKFLOW_START_RACE_ERRORS,
    workflow_start_raced,
)

WORKFLOW_ID = "generate:pred-1"


@pytest.mark.parametrize(
    "error",
    [
        DBOSWorkflowConflictIDError(WORKFLOW_ID),
        DBOSQueueDeduplicatedError(WORKFLOW_ID, "generation", "dedup-1"),
        DBOSConflictingWorkflowError(WORKFLOW_ID),
    ],
)
def test_workflow_start_raced_returns_true_for_typed_dbos_errors(
    error: BaseException,
) -> None:
    assert isinstance(error, WORKFLOW_START_RACE_ERRORS)
    with patch("dr_dspy.harness.dbos.DBOS.get_workflow_status") as status:
        raced = workflow_start_raced(workflow_id=WORKFLOW_ID, error=error)
        assert raced is True
        status.assert_not_called()


def test_workflow_start_raced_handles_base_exception_conflict() -> None:
    error = DBOSWorkflowConflictIDError(WORKFLOW_ID)
    assert not isinstance(error, Exception)
    assert workflow_start_raced(workflow_id=WORKFLOW_ID, error=error) is True


def test_workflow_start_raced_returns_false_without_status() -> None:
    with patch(
        "dr_dspy.harness.dbos.DBOS.get_workflow_status",
        return_value=None,
    ):
        assert (
            workflow_start_raced(
                workflow_id=WORKFLOW_ID,
                error=ValueError("connection refused"),
            )
            is False
        )


def test_workflow_start_raced_returns_true_when_status_exists() -> None:
    with patch(
        "dr_dspy.harness.dbos.DBOS.get_workflow_status",
        return_value={"status": "ENQUEUED"},
    ):
        assert (
            workflow_start_raced(
                workflow_id=WORKFLOW_ID,
                error=ValueError("race lost"),
            )
            is True
        )
