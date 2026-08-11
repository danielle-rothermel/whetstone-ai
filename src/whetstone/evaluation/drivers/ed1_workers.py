"""Legacy shim; implementation in evaluation.drivers.code_comp.workers."""

from whetstone.evaluation.drivers.code_comp.workers import (
    DUMMY_ALTERNATE_PASSING_BODY,
    DUMMY_FAILING_BODY,
    DUMMY_PASSING_BODY,
    drive_dummy_ed1_generation,
    drive_provider_ed1_generation,
)

__all__ = [
    "DUMMY_ALTERNATE_PASSING_BODY",
    "DUMMY_FAILING_BODY",
    "DUMMY_PASSING_BODY",
    "drive_dummy_ed1_generation",
    "drive_provider_ed1_generation",
]
