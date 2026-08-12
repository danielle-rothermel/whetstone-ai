from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, JsonValue

from whetstone.evaluation.traces import ExecutedComponentStep, ExecutedRowState


@dataclass(frozen=True, slots=True)
class GenerationRowOutput:
    """One generation row's exact trace, display output, and score."""

    candidate_id: str
    task_id: str
    task_index: int
    sample_index: int
    row_state: ExecutedRowState
    executed_component_steps: tuple[ExecutedComponentStep, ...]
    output_text: str | None
    score: float | None
    failure_code: str = ""
    finish_reason: str | None = None
    provider_error: dict[str, object] | None = None
    max_budget: int | None = None
    over_budget: bool | None = None
    submission_result: object | None = None

    @property
    def failed(self) -> bool:
        return self.row_state is ExecutedRowState.FAILED

    @property
    def missing(self) -> bool:
        return self.row_state is ExecutedRowState.MISSING

    @property
    def invalid(self) -> bool:
        return False


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


def start_phase_deadline(max_wall_seconds: float | None) -> float | None:
    """Validate one phase wall and convert it to an absolute deadline."""
    if max_wall_seconds is None:
        return None
    if type(max_wall_seconds) not in (int, float):
        raise ValueError(
            "max_wall_seconds must be a finite nonnegative real number"
        )
    try:
        seconds = float(max_wall_seconds)
    except OverflowError:
        raise ValueError(
            "max_wall_seconds must be a finite nonnegative real number "
            "representable as seconds"
        ) from None
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(
            "max_wall_seconds must be a finite nonnegative real number"
        )
    return time.monotonic() + seconds


def remaining_phase_wall_seconds(deadline: float | None) -> float | None:
    """Return the nonnegative remainder of one shared phase wall."""
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


__all__ = [
    "GenerationRowOutput",
    "ProcessTask",
    "_process_payload_hash",
    "process_request_hash",
    "remaining_phase_wall_seconds",
    "start_phase_deadline",
]
