"""The Operator Retry gate and the retried-executor binding recheck.

Operator Retry is the only path to a second Platform Stage Attempt after a
failed rollout-execution Stage (``design/vocab_and_defs.html`` · *Operator
Retry*). It is a two-part guard against overwriting a durable winner:

1. **Gateway precheck** (:func:`operator_retry`). A Whetstone operator command
   or gateway invokes the rebuilt dr-platform ``retry_stage`` **only** when the
   platform Stage is ``FAILED`` *and* the Rollout Execution Key is unbound in
   the authoritative Result Store. If either precondition fails it refuses
   without touching the platform.

2. **Executor recheck** (:func:`assert_unbound_before_effect`). The gateway
   precheck can race: a concurrent actor may bind the key after the check but
   before the retried executor starts. So the retried executor **rechecks the
   authoritative binding before any new semantic effect**. If the key is now
   bound, the executor stops via the idempotent-or-conflict rule (it never
   issues a new provider call and never overwrites the winner).

The Result Store's atomic compare-and-set remains authoritative: an unbound key
may acquire one reference; the same reference replays idempotently; a different
reference conflicts and never overwrites. An orphan Object Reference that was
persisted but never bound does not by itself forbid retry — only a *binding*
does.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING

from dr_platform.staging.operations import retry_stage
from dr_platform.staging.stage_executions import get_stage_execution
from dr_platform.staging.states import StageExecutionState

if TYPE_CHECKING:
    from collections.abc import Callable

    from dr_platform.staging.operations import StageRetryResult
    from dr_platform.staging.schema import StagingSchema
    from dr_store import ObjectReference
    from sqlalchemy import Engine

    from whetstone.graph.rollout import RolloutExecutionKey
    from whetstone.result import ResultStore


def _utc_now() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "OperatorRetryRefused",
    "RetriedExecutorStop",
    "RetryRefusalReason",
    "assert_unbound_before_effect",
    "operator_retry",
]


class RetryRefusalReason(Enum):
    """Why the operator gateway refused to invoke ``retry_stage``."""

    #: The platform Stage is not FAILED (only a FAILED Stage may be retried).
    STAGE_NOT_FAILED = "stage_not_failed"
    #: The Rollout Execution Key is already bound to a terminal Result.
    KEY_ALREADY_BOUND = "key_already_bound"
    #: The stage execution does not exist.
    STAGE_UNKNOWN = "stage_unknown"


class OperatorRetryRefused(Exception):
    """The operator gateway refused ``retry_stage`` before touching platform.

    Carries the specific :class:`RetryRefusalReason` and, for an already-bound
    key, the existing authoritative binding (the durable winner), so the
    operator sees exactly why retry is not permitted.
    """

    def __init__(
        self,
        *,
        reason: RetryRefusalReason,
        stage_execution_id: int,
        existing_binding: ObjectReference | None = None,
        stage_state: StageExecutionState | None = None,
    ) -> None:
        self.reason = reason
        self.stage_execution_id = stage_execution_id
        self.existing_binding = existing_binding
        self.stage_state = stage_state
        super().__init__(
            f"operator retry refused for stage {stage_execution_id}: "
            f"{reason.value}"
        )


class RetriedExecutorStop(Exception):
    """The retried executor stopped because the key won a binding meanwhile.

    Raised by :func:`assert_unbound_before_effect` when the recheck finds the
    Rollout Execution Key already bound. It is not a Stage failure of producing
    a Result: a durable winner already exists, so this attempt must not proceed
    to any new semantic effect. The existing binding (the winner) is exposed.
    """

    def __init__(
        self,
        *,
        key: RolloutExecutionKey,
        existing_binding: ObjectReference,
    ) -> None:
        self.key = key
        self.existing_binding = existing_binding
        super().__init__(
            "retried executor stopped: Rollout Execution Key already bound to "
            f"({existing_binding.schema!r}, {existing_binding.content_hash})"
        )


@dataclass(frozen=True, slots=True)
class OperatorRetryResult:
    """The platform result of a permitted operator retry."""

    stage_retry: StageRetryResult


def operator_retry(
    *,
    stage_execution_id: int,
    execution_key: RolloutExecutionKey,
    result_store: ResultStore,
    engine: Engine,
    clock: Callable[[], datetime] = _utc_now,
    schema: StagingSchema | None = None,
) -> OperatorRetryResult:
    """Verify eligibility, then invoke dr-platform ``retry_stage``.

    Refuses (raising :class:`OperatorRetryRefused`) unless the platform Stage
    is ``FAILED`` and the Rollout Execution Key is unbound in the authoritative
    Result Store. Both preconditions are checked before ``retry_stage`` is
    invoked; the precheck can still race, so the retried executor must recheck
    (:func:`assert_unbound_before_effect`).
    """
    with engine.connect() as connection:
        execution = get_stage_execution(
            connection,
            stage_execution_id=stage_execution_id,
            schema=schema,
        )
    if execution is None:
        raise OperatorRetryRefused(
            reason=RetryRefusalReason.STAGE_UNKNOWN,
            stage_execution_id=stage_execution_id,
        )
    if execution.state is not StageExecutionState.FAILED:
        raise OperatorRetryRefused(
            reason=RetryRefusalReason.STAGE_NOT_FAILED,
            stage_execution_id=stage_execution_id,
            stage_state=execution.state,
        )
    existing = result_store.resolve(execution_key)
    if existing is not None:
        raise OperatorRetryRefused(
            reason=RetryRefusalReason.KEY_ALREADY_BOUND,
            stage_execution_id=stage_execution_id,
            existing_binding=existing,
        )

    stage_retry = retry_stage(
        stage_execution_id,
        engine=engine,
        clock=clock,
        schema=schema,
    )
    return OperatorRetryResult(stage_retry=stage_retry)


def assert_unbound_before_effect(
    *,
    execution_key: RolloutExecutionKey,
    result_store: ResultStore,
) -> None:
    """Recheck the authoritative binding before any new semantic effect.

    Called by the retried executor at the top of its Stage Attempt, before it
    issues any provider call. If the Rollout Execution Key is already bound
    (a concurrent actor won after the gateway precheck), it raises
    :class:`RetriedExecutorStop` so the executor stops without a new effect and
    never overwrites the winner. If the key is unbound it returns and the
    executor proceeds; the Result Store's atomic compare-and-set is the final
    authority if two executors still race past this point.
    """
    existing = result_store.resolve(execution_key)
    if existing is not None:
        raise RetriedExecutorStop(
            key=execution_key,
            existing_binding=existing,
        )
