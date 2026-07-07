from __future__ import annotations

from enum import StrEnum
from typing import Any

from dbos import DBOS, SetWorkflowID
from dr_platform import (
    DBOS_ACTIVE_WORKFLOW_STATUSES,
    DBOS_FAILED_WORKFLOW_STATUSES,
    DbosWorkflowStatus,
)
from sqlalchemy import create_engine, select

from whetstone.db import schema


class ScoringWorkflowPresence(StrEnum):
    ABSENT = "absent"
    IN_FLIGHT = "in_flight"
    COMPLETE = "complete"
    ORPHAN = "orphan"


def dbos_workflow_status_value(workflow_status: Any) -> str | None:
    if workflow_status is None:
        return None
    if isinstance(workflow_status, dict):
        value = workflow_status.get("status")
        return str(value) if value is not None else None
    value = getattr(workflow_status, "status", None)
    return str(value) if value is not None else None


def score_attempt_exists(database_url: str, score_attempt_id: str) -> bool:
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            row = connection.execute(
                select(schema.score_attempts.c.score_attempt_id).where(
                    schema.score_attempts.c.score_attempt_id
                    == score_attempt_id
                )
            ).first()
        return row is not None
    finally:
        engine.dispose()


def classify_scoring_workflow_presence(
    *,
    database_url: str,
    score_attempt_id: str,
    workflow_id: str,
) -> ScoringWorkflowPresence:
    if score_attempt_exists(database_url, score_attempt_id):
        return ScoringWorkflowPresence.COMPLETE

    status_value = dbos_workflow_status_value(
        DBOS.get_workflow_status(workflow_id)
    )
    if status_value is None:
        return ScoringWorkflowPresence.ABSENT
    if status_value in DBOS_ACTIVE_WORKFLOW_STATUSES:
        return ScoringWorkflowPresence.IN_FLIGHT
    if status_value == DbosWorkflowStatus.SUCCESS.value:
        return ScoringWorkflowPresence.ORPHAN
    if status_value in DBOS_FAILED_WORKFLOW_STATUSES:
        return ScoringWorkflowPresence.ORPHAN
    return ScoringWorkflowPresence.ORPHAN


def recover_orphan_scoring_workflow(
    *,
    database_url: str,
    workflow_id: str,
    score_attempt_id: str,
    replay_workflow: Any,
    replay_args: tuple[Any, ...],
) -> bool:
    handle = DBOS.retrieve_workflow(workflow_id)
    try:
        handle.get_result()
    except Exception:
        pass
    if score_attempt_exists(database_url, score_attempt_id):
        return True
    replay = getattr(replay_workflow, "__wrapped__", replay_workflow)
    with SetWorkflowID(workflow_id):
        replay(*replay_args)
    return score_attempt_exists(database_url, score_attempt_id)
