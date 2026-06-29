from __future__ import annotations

from alembic.config import Config
from alembic.script import ScriptDirectory


def test_alembic_discovers_v1_schema_revision() -> None:
    config = Config("alembic.ini")
    script = ScriptDirectory.from_config(config)

    assert script.get_current_head() == "20260629_0001"
