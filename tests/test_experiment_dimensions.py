from __future__ import annotations

import pytest

from dr_dspy.experiment_dimensions import (
    Dimension,
    dimension_columns_ddl,
    identity_constraint_columns,
    identity_dimension_names,
    reporting_dimension_names,
)

DIMENSIONS = (
    Dimension(
        name="model",
        sql_type="TEXT",
        nullable=False,
        report_title="Model",
    ),
    Dimension(
        name="temperature",
        sql_type="DOUBLE PRECISION",
        report_title="Temp",
        report_justify="right",
    ),
    Dimension(
        name="budget_ratio",
        sql_type="DOUBLE PRECISION",
        report_title="Budget",
        report_justify="right",
    ),
    Dimension(
        name="reasoning",
        sql_type="JSONB",
        nullable=False,
        default_sql="'{}'::jsonb",
        in_reporting=False,
        report_title="Reasoning",
    ),
)


def test_column_ddl_renders_type_nullability_and_default() -> None:
    assert DIMENSIONS[0].column_ddl() == "model TEXT NOT NULL"
    assert DIMENSIONS[1].column_ddl() == "temperature DOUBLE PRECISION"
    assert (
        DIMENSIONS[3].column_ddl()
        == "reasoning JSONB NOT NULL DEFAULT '{}'::jsonb"
    )


def test_dimension_columns_ddl_joins_each_column() -> None:
    ddl = dimension_columns_ddl(DIMENSIONS)
    assert "    model TEXT NOT NULL," in ddl
    assert "    budget_ratio DOUBLE PRECISION," in ddl
    assert ddl.count("\n") == len(DIMENSIONS) - 1


def test_identity_constraint_columns_bracket_dimensions() -> None:
    columns = identity_constraint_columns(DIMENSIONS)
    assert columns[0] == "experiment_name"
    assert columns[1] == "task_id"
    assert columns[-1] == "repetition_seed"
    assert "model" in columns
    assert "reasoning" in columns


def test_identity_vs_reporting_names() -> None:
    assert identity_dimension_names(DIMENSIONS) == (
        "model",
        "temperature",
        "budget_ratio",
        "reasoning",
    )
    assert reporting_dimension_names(DIMENSIONS) == (
        "model",
        "temperature",
        "budget_ratio",
    )


def test_bad_identifier_rejected() -> None:
    bad = Dimension(name="bad name", sql_type="TEXT", report_title="x")
    with pytest.raises(ValueError):
        bad.column_ddl()
