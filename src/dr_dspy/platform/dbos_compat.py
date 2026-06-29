from __future__ import annotations

from dbos._error import (
    DBOSConflictingWorkflowError,
    DBOSQueueDeduplicatedError,
    DBOSWorkflowConflictIDError,
)

# DBOS does not currently expose public exception classes for workflow start
# races. Keep the private import isolated here so the queue worker has one
# compatibility point if DBOS changes these names.
WORKFLOW_START_RACE_ERRORS: tuple[type[BaseException], ...] = (
    DBOSWorkflowConflictIDError,
    DBOSQueueDeduplicatedError,
    DBOSConflictingWorkflowError,
)
