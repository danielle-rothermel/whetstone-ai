"""Pure aggregation over explicit evaluation inputs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from whetstone.evaluation.config import AggregationConfig, _FrozenModel


class AggregationStatus(StrEnum):
    OK = "ok"
    MISSING_DATA = "missing_data"
    NOT_APPLICABLE = "not_applicable"
    ZERO_DENOMINATOR = "zero_denominator"


@dataclass(frozen=True, slots=True)
class AggregationInput:
    value: float | None
    applicable: bool = True


class AggregationOutput(_FrozenModel):
    status: AggregationStatus
    value: float | None
    count_total: int
    count_applicable: int
    count_present: int


def aggregate(
    config: AggregationConfig,
    inputs: tuple[AggregationInput, ...],
) -> AggregationOutput:
    assignment = config.assignment_dict()
    reduction = assignment["reduction"]
    missing_policy = assignment.get("missing_data", "propagate")
    zero_denominator = assignment.get("zero_denominator", "not_applicable")

    applicable = [item for item in inputs if item.applicable]
    present = [item.value for item in applicable if item.value is not None]
    counts = {
        "count_total": len(inputs),
        "count_applicable": len(applicable),
        "count_present": len(present),
    }
    if not applicable:
        return AggregationOutput(
            status=AggregationStatus.NOT_APPLICABLE,
            value=None,
            **counts,
        )
    if len(present) < len(applicable) and missing_policy == "propagate":
        return AggregationOutput(
            status=AggregationStatus.MISSING_DATA,
            value=None,
            **counts,
        )
    if reduction == "sum":
        return AggregationOutput(
            status=AggregationStatus.OK,
            value=float(sum(present)),
            **counts,
        )
    if not present:
        if zero_denominator == "error":
            raise ZeroDivisionError(
                "mean aggregation has zero contributing values"
            )
        return AggregationOutput(
            status=AggregationStatus.ZERO_DENOMINATOR,
            value=None,
            **counts,
        )
    return AggregationOutput(
        status=AggregationStatus.OK,
        value=sum(present) / len(present),
        **counts,
    )


__all__ = [
    "AggregationInput",
    "AggregationOutput",
    "AggregationStatus",
    "aggregate",
]
