"""dr-platform wiring: whetstone's frozen physical naming.

The naming configuration is a frozen contract: it maps the library's
neutral schema onto the physical names whetstone's own (byte-frozen)
Alembic history created — ``dr_dspy_*`` tables and the
``prediction_id`` / ``fair_order_key`` / ``experiment_name`` column
words — and keeps the ``batch_submit_item_id`` digest bytes identical.
"""

from __future__ import annotations

from dr_platform import (
    PlatformNaming,
    PlatformSchema,
    normalize_postgresql_driver_url,
    stamp_platform_schema,
    upgrade_platform_schema,
)
from sqlalchemy import create_engine, inspect

WHETSTONE_PLATFORM_NAMING = PlatformNaming(
    prefix="dr_dspy",
    item_key_label="prediction_id",
    order_key_label="fair_order_key",
    group_key_label="experiment_name",
)

PLATFORM_SCHEMA = PlatformSchema(WHETSTONE_PLATFORM_NAMING)


def ensure_platform_schema(database_url: str) -> None:
    """Adopt/advance the dr-platform lineage on a whetstone database.

    The platform tables already exist from whetstone's own migration
    history, so the first run stamps the library baseline instead of
    creating them; every run then applies post-baseline platform
    migrations (holds/tags columns, projections registry).
    """
    resolved = normalize_postgresql_driver_url(database_url)
    engine = create_engine(resolved)
    try:
        with engine.connect() as connection:
            # Pin to the first search_path schema: a lineage living in a
            # fallback schema (e.g. public on the dev DB) must not mask
            # an unadopted scratch schema.
            current_schema = connection.exec_driver_sql(
                "SELECT current_schema()"
            ).scalar()
            has_lineage = inspect(connection).has_table(
                WHETSTONE_PLATFORM_NAMING.alembic_version_table,
                schema=current_schema,
            )
    finally:
        engine.dispose()
    if not has_lineage:
        stamp_platform_schema(resolved, naming=WHETSTONE_PLATFORM_NAMING)
    upgrade_platform_schema(resolved, naming=WHETSTONE_PLATFORM_NAMING)
