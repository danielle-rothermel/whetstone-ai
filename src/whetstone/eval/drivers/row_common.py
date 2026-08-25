from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ConfigDict, JsonValue

from whetstone.eval.traces import ExecutedComponentStep, ExecutedRowState


@dataclass(frozen=True, slots=True)
class RolloutRowOutput:
    """One generation row's exact trace, display output, and score."""

    candidate_id: str
    task_id: str
    task_index: int
    seed_index: int
    row_state: ExecutedRowState
    trace_steps: tuple[ExecutedComponentStep, ...]
    output_text: str | None
    score: float | None
    failure_code: str = ""
    finish_reason: str | None = None
    provider_error: dict[str, object] | None = None
    max_budget: int | None = None
    over_budget: bool | None = None
    submission_result: object | None = None
    #: Task-model usage observed for this row, carried into persisted
    #: evidence so run-level spend is re-derivable from the store.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    provider_cost: float | None = None
    #: The prompt cache served this row's provider call from a stored
    #: result. The replayed usage above is the *original* call's, so run
    #: cost must report this row separately rather than bill it again.
    cache_hit: bool = False
    #: What raised inside the graph, when a node failed. Without these a
    #: node failure persists only ``failure_code``, which cannot say why a
    #: row was lost once the run is over.
    error_type: str | None = None
    error_message: str | None = None
    failed_node_id: str | None = None
    #: How many times this row's graph was executed. Attempts beyond the
    #: first are node-failure re-executions; each one issued its own
    #: provider call and is billed.
    row_attempts: int = 1

    @property
    def failed(self) -> bool:
        return self.row_state is ExecutedRowState.FAILED

    @property
    def missing(self) -> bool:
        return self.row_state is ExecutedRowState.MISSING

    @property
    def invalid(self) -> bool:
        return self.row_state is ExecutedRowState.INVALID


class ProcessTask(BaseModel):
    """JSON-safe task payload submitted to a row process worker."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: str
    seed: int
    strata: tuple[str, ...]
    prompt_inputs: dict[str, str]
    gold: str


def _process_payload_hash(payload: JsonValue) -> str:
    """Hash the exact finite JSON payload submitted to a process worker."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def process_request_hash(model: BaseModel) -> str:
    """Hash one strict row request's submitted JSON representation."""
    return _process_payload_hash(model.model_dump(mode="json"))


MAX_REPRESENTABLE_WALL_SECONDS: Final = 9_223_372_036.854_774
"""The longest finite wall dr-exec can express as a positive limit.

``FiniteDurationLimit.from_seconds`` converts seconds to a nanosecond count
and rejects anything reaching ``sys.maxsize`` nanoseconds. This is the
largest float strictly under that ceiling — roughly 292 years.
"""


def validated_phase_wall_seconds(
    max_wall_seconds: float | None, /
) -> float | None:
    """Reject an unusable phase wall and normalise an unbounded one.

    This is the drivers' single owner of what a caller may pass as an
    operation deadline, so the in-process and subprocess drivers cannot
    drift on which walls are legal.

    A negative or NaN wall is a caller mistake: it names no interval, so it
    is raised at the call boundary rather than quietly expiring the batch and
    persisting rows as deadline misses. Positive infinity names "no
    deadline", which is exactly ``None``. Zero is a legal, already-elapsed
    wall: the operation is over before any row runs.

    A finite wall longer than :data:`MAX_REPRESENTABLE_WALL_SECONDS` is more
    generous than any deadline dr-exec can express, so it reads as "no
    deadline" too. Passing it through would let the subprocess driver's
    conversion fail and collapse a century-long wall into an immediate
    expiry — the opposite of what the caller asked for. Both drivers apply
    the one rule here so neither can disagree about where "generous" ends.
    """
    if max_wall_seconds is None:
        return None
    if type(max_wall_seconds) not in (int, float):
        raise ValueError(
            "max_wall_seconds must be a nonnegative real number of seconds, "
            f"not {max_wall_seconds!r}"
        )
    try:
        seconds = float(max_wall_seconds)
    except OverflowError:
        raise ValueError(
            "max_wall_seconds must be a nonnegative real number "
            "representable as seconds"
        ) from None
    if math.isnan(seconds) or seconds < 0:
        raise ValueError(
            "max_wall_seconds must be a nonnegative real number of seconds, "
            f"not {max_wall_seconds!r}"
        )
    if math.isinf(seconds) or seconds > MAX_REPRESENTABLE_WALL_SECONDS:
        return None
    return seconds


def start_phase_deadline(max_wall_seconds: float | None) -> float | None:
    """Validate one phase wall and convert it to an absolute deadline."""
    seconds = validated_phase_wall_seconds(max_wall_seconds)
    if seconds is None:
        return None
    return time.monotonic() + seconds


def remaining_phase_wall_seconds(deadline: float | None) -> float | None:
    """Return the nonnegative remainder of one shared phase wall."""
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


__all__ = [
    "MAX_REPRESENTABLE_WALL_SECONDS",
    "RolloutRowOutput",
    "ProcessTask",
    "_process_payload_hash",
    "process_request_hash",
    "remaining_phase_wall_seconds",
    "start_phase_deadline",
    "validated_phase_wall_seconds",
]
