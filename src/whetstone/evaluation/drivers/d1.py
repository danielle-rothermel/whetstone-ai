"""Legacy shim; implementation in evaluation.drivers.code_comp.direct."""

from whetstone.evaluation.drivers.code_comp.direct import (
    D1EvalResult,
    D1GeneratedRowOutcome,
    D1RowJobFactory,
    D1RowOutcome,
    D1RowRequest,
    D1RowResult,
    _input_arm_text,
    drive_d1_row,
    run_d1_eval,
)
from whetstone.evaluation.drivers.internal import remaining_phase_wall_seconds
from whetstone.execution.fanout import run_call_pool
from whetstone.execution.resume import index_partial_records

__all__ = [
    "D1EvalResult",
    "D1GeneratedRowOutcome",
    "D1RowJobFactory",
    "D1RowOutcome",
    "D1RowRequest",
    "D1RowResult",
    "_input_arm_text",
    "drive_d1_row",
    "index_partial_records",
    "remaining_phase_wall_seconds",
    "run_call_pool",
    "run_d1_eval",
]
