"""App-side DBOS config: whetstone's concurrency knobs.

URL resolution/normalization and runtime teardown live in dr-platform
(`resolve_database_url`, `destroy_dbos_runtime`); this module keeps
only what is whetstone-specific — the two queue concurrency knobs.
"""

from __future__ import annotations

import os

from dbos import DBOSConfig
from dr_platform import resolve_database_url
from dr_platform.dbos_config import (
    DBOS_SYSTEM_DATABASE_URL_ENV,
    normalize_postgresql_driver_url,
)
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr


class EvalDbosConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    database_url: StrictStr
    dbos_system_database_url: StrictStr
    generation_concurrency: StrictInt
    scoring_concurrency: StrictInt


def build_eval_dbos_config(
    *,
    database_url: str | None,
    dbos_system_database_url: str | None,
    generation_concurrency: int,
    scoring_concurrency: int,
    database_url_env: str = "DATABASE_URL",
    dbos_system_database_url_env: str = DBOS_SYSTEM_DATABASE_URL_ENV,
    database_url_error_suffix: str = "",
) -> EvalDbosConfig:
    resolved_database_url = resolve_database_url(
        database_url,
        database_url_env=database_url_env,
        error_suffix=database_url_error_suffix,
    )
    resolved_system_database_url = (
        dbos_system_database_url
        or os.environ.get(dbos_system_database_url_env)
        or resolved_database_url
    )
    return EvalDbosConfig(
        database_url=resolved_database_url,
        dbos_system_database_url=normalize_postgresql_driver_url(
            resolved_system_database_url
        ),
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
    )


def build_dbos_config(config: EvalDbosConfig, *, app_name: str) -> DBOSConfig:
    return {
        "name": app_name,
        "system_database_url": config.dbos_system_database_url,
    }
