from __future__ import annotations

from dbos import DBOS
from dbos._error import (
    DBOSConflictingWorkflowError,
    DBOSQueueDeduplicatedError,
    DBOSWorkflowConflictIDError,
)

# DBOS does not currently expose public exception classes for workflow start
# races. Keep the private import isolated here so platform and harness callers
# share one compatibility point if DBOS changes these names.
WORKFLOW_START_RACE_ERRORS: tuple[type[BaseException], ...] = (
    DBOSWorkflowConflictIDError,
    DBOSQueueDeduplicatedError,
    DBOSConflictingWorkflowError,
)


def workflow_start_raced(*, workflow_id: str, error: BaseException) -> bool:
    """Return True when a concurrent start/enqueue won (idempotent caller)."""
    if isinstance(error, WORKFLOW_START_RACE_ERRORS):
        return True
    if isinstance(error, Exception) and (
        DBOS.get_workflow_status(workflow_id) is not None
    ):
        return True
    return False
