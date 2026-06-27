"""First-class sweep-dimension spec for the HumanEval eval harness.

A `Dimension` is one axis of an experiment sweep (model, temperature,
reasoning, budget ratio, ...). The spec is the single source of truth that
drives identity (prediction-id hash + UNIQUE constraint + table columns) and
reporting (status / analysis columns), so adding a sweep axis is a one-line
change instead of edits scattered across schema, SQL, and reporting.

Payload columns (generation output, scores, metadata) stay explicit
hand-written SQL in each experiment module; only the dimension columns are
generated from this spec.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr

from dr_dspy.eval_reporting import StatusDimension, validate_sql_identifier

ReportJustify = Literal["default", "left", "center", "right", "full"]

# Identity columns that bracket the dimension columns in the UNIQUE key.
IDENTITY_PREFIX_COLUMNS: tuple[str, ...] = ("experiment_name", "task_id")
IDENTITY_SUFFIX_COLUMNS: tuple[str, ...] = ("repetition_seed",)


class Dimension(BaseModel):
    """One sweep axis: a column that is part of identity and/or reporting."""

    model_config = ConfigDict(extra="forbid")

    name: StrictStr
    sql_type: StrictStr
    nullable: StrictBool = True
    default_sql: StrictStr | None = None
    in_identity: StrictBool = True
    in_reporting: StrictBool = True
    report_title: StrictStr
    report_justify: ReportJustify = "left"

    def column_ddl(self) -> str:
        validate_sql_identifier(self.name)
        parts = [self.name, self.sql_type]
        if not self.nullable:
            parts.append("NOT NULL")
        if self.default_sql is not None:
            parts.append(f"DEFAULT {self.default_sql}")
        return " ".join(parts)


def dimension_names(dimensions: Sequence[Dimension]) -> tuple[str, ...]:
    return tuple(dimension.name for dimension in dimensions)


def identity_dimension_names(
    dimensions: Sequence[Dimension],
) -> tuple[str, ...]:
    return tuple(d.name for d in dimensions if d.in_identity)


def reporting_dimension_names(
    dimensions: Sequence[Dimension],
) -> tuple[str, ...]:
    return tuple(d.name for d in dimensions if d.in_reporting)


def dimension_columns_ddl(dimensions: Sequence[Dimension]) -> str:
    """Column definitions for a CREATE TABLE statement, newline-joined."""
    return "\n".join(f"    {d.column_ddl()}," for d in dimensions)


def identity_constraint_columns(
    dimensions: Sequence[Dimension],
) -> tuple[str, ...]:
    """Full UNIQUE-key column list: prefix + identity dimensions + suffix."""
    columns = (
        *IDENTITY_PREFIX_COLUMNS,
        *identity_dimension_names(dimensions),
        *IDENTITY_SUFFIX_COLUMNS,
    )
    for column in columns:
        validate_sql_identifier(column)
    return columns


def status_dimensions(
    dimensions: Sequence[Dimension],
) -> list[StatusDimension]:
    """StatusDimension list for reporting columns (status + analysis)."""
    return [
        StatusDimension(
            key=d.name,
            title=d.report_title,
            justify=d.report_justify,
        )
        for d in dimensions
        if d.in_reporting
    ]
