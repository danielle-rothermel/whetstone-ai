from __future__ import annotations

import importlib
from typing import Any

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_mock_engine


def test_alembic_discovers_v1_schema_revision() -> None:
    config = Config("alembic.ini")
    script = ScriptDirectory.from_config(config)

    assert script.get_current_head() == "20260629_0001"


def test_alembic_v1_schema_revision_renders_upgrade_and_downgrade(
    monkeypatch: Any,
) -> None:
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
    monkeypatch.setattr(migration.op, "get_bind", lambda: engine)

    migration.upgrade()
    migration.downgrade()

    rendered = "\n".join(statements)
    assert "CREATE TABLE dr_dspy_prediction_specs" in rendered
    assert "CREATE TABLE dr_dspy_prediction_projection" in rendered
    assert "DROP TABLE dr_dspy_prediction_specs" in rendered
    assert "DROP TABLE dr_dspy_experiments" in rendered
