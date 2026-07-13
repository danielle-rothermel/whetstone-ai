"""Whetstone's fresh platform schema bootstrap."""

from __future__ import annotations

from dr_platform.db import PlatformSchema, upgrade_platform_schema
from dr_platform.dbos_config import normalize_postgresql_driver_url
from sqlalchemy import create_engine

PLATFORM_SCHEMA = PlatformSchema(prefix="whetstone")


def ensure_platform_schema(database_url: str) -> None:
    """Upgrade the independent kernel schema using its default naming."""
    resolved = normalize_postgresql_driver_url(database_url)
    engine = create_engine(resolved)
    try:
        upgrade_platform_schema(resolved, prefix="whetstone")
    finally:
        engine.dispose()
