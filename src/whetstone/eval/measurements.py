from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import model_validator

from whetstone.eval.config import _FrozenModel

type FactScalar = float | int | str | bool


class Applicability(StrEnum):
    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"


class OperatorLineage(_FrozenModel):
    evaluation_procedure_config_hash: str
    operator: str
    operator_version: str
    step: str | None = None
    step_version: str | None = None

    @model_validator(mode="after")
    def validate_step_pair(self) -> Self:
        if (self.step is None) != (self.step_version is None):
            raise ValueError("step and step_version must be set together")
        return self


class MetricFact(_FrozenModel):
    name: str
    value: FactScalar
    unit: str
    applicability: Applicability
    lineage: OperatorLineage

    @model_validator(mode="after")
    def validate_fields(self) -> Self:
        if not self.name:
            raise ValueError("a metric fact requires a name")
        if not self.unit:
            raise ValueError("a metric fact requires an explicit unit")
        return self


class Score(_FrozenModel):
    name: str
    value: FactScalar
    unit: str
    evaluation_procedure_config_hash: str
    derived_from: tuple[str, ...]

    @model_validator(mode="after")
    def validate_fields(self) -> Self:
        if not self.unit:
            raise ValueError("a score requires an explicit unit")
        if not self.derived_from:
            raise ValueError("a score requires source fact names")
        return self


__all__ = [
    "Applicability",
    "FactScalar",
    "MetricFact",
    "OperatorLineage",
    "Score",
]
