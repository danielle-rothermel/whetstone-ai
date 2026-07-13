"""Whetstone's fresh platform schema bootstrap."""

from __future__ import annotations

import importlib
from typing import Any, cast

from alembic.migration import MigrationContext
from alembic.operations import Operations
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


def ensure_whetstone_application_schema(database_url: str) -> None:
    """Upgrade a fresh bound schema through domain and kernel baselines."""
    resolved = normalize_postgresql_driver_url(database_url)
    engine = create_engine(resolved)
    try:
        with engine.begin() as connection:
            for module_name in (
                "20260712_0001_whetstone_baseline",
                "20260713_0002_generation_manifest_shards",
            ):
                migration = cast(
                    Any,
                    importlib.import_module(
                        f"whetstone.db.migrations.versions.{module_name}"
                    ),
                )
                migration.op = Operations(
                    MigrationContext.configure(connection)
                )
                migration.upgrade()
        upgrade_platform_schema(resolved, prefix="whetstone")
    finally:
        engine.dispose()
