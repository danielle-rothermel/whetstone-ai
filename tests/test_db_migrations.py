from __future__ import annotations

import importlib
from typing import Any, cast

from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from sqlalchemy import Constraint, Table, create_mock_engine

from dr_dspy.db import schema


def test_alembic_discovers_v1_schema_revision() -> None:
    config = Config("alembic.ini")
    script = ScriptDirectory.from_config(config)

    assert script.get_current_head() == "20260629_0001"


def test_alembic_v1_schema_revision_renders_upgrade_and_downgrade(
    monkeypatch: Any,
) -> None:
    migration, statements = _render_upgrade(monkeypatch)
    migration.downgrade()

    rendered = "\n".join(statements)
    assert "CREATE TABLE dr_dspy_prediction_specs" in rendered
    assert "CREATE TABLE dr_dspy_prediction_projection" in rendered
    assert "DROP TABLE dr_dspy_prediction_specs" in rendered
    assert "DROP TABLE dr_dspy_experiments" in rendered


def test_alembic_v1_schema_revision_matches_live_named_contracts(
    monkeypatch: Any,
) -> None:
    _, statements = _render_upgrade(monkeypatch)
    rendered = "\n".join(statements)

    for table in schema.v1_tables:
        assert f"CREATE TABLE {table.name}" in rendered
        for column in table.columns:
            assert column.name in rendered
        for constraint_name in _named_constraint_names(table):
            assert constraint_name in rendered
        for index in table.indexes:
            assert index.name is not None
            assert index.name in rendered


def _render_upgrade(monkeypatch: Any) -> tuple[Any, list[str]]:
    migration = importlib.import_module(
        "dr_dspy.db.migrations.versions.20260629_0001_v1_domain_schema"
    )
    statements: list[str] = []
    engine = create_mock_engine(
        "postgresql+psycopg://",
        lambda sql, *args, **kwargs: statements.append(
            str(sql.compile(dialect=engine.dialect))
        ),
    )
    context = MigrationContext.configure(cast(Any, engine.connect()))
    monkeypatch.setattr(migration, "op", Operations(context))

    migration.upgrade()
    return migration, statements


def _named_constraint_names(table: Table) -> set[str]:
    return {
        str(constraint.name)
        for constraint in table.constraints
        if _has_name(constraint)
    }


def _has_name(constraint: Constraint) -> bool:
    return constraint.name is not None
