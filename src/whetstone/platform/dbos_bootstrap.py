"""DBOS bootstrap for platform CLI entrypoints and integration tests."""

from __future__ import annotations

import os

from dbos import DBOS, DBOSConfig
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

from whetstone.db.migrations.url import normalize_postgresql_driver_url

DATABASE_URL_ENV = "DATABASE_URL"
DBOS_SYSTEM_DATABASE_URL_ENV = "DBOS_SYSTEM_DATABASE_URL"


class EvalDbosConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    database_url: StrictStr
    dbos_system_database_url: StrictStr
    generation_concurrency: StrictInt
    scoring_concurrency: StrictInt


def resolve_database_url(
    database_url: str | None,
    *,
    database_url_env: str = DATABASE_URL_ENV,
    error_suffix: str = "",
) -> str:
    resolved = database_url or os.environ.get(database_url_env)
    if not resolved:
        suffix = f" {error_suffix}" if error_suffix else ""
        raise ValueError(
            f"--database-url or {database_url_env} is required{suffix}"
        )
    return normalize_postgresql_driver_url(resolved)


def build_eval_dbos_config(
    *,
    database_url: str | None,
    dbos_system_database_url: str | None,
    generation_concurrency: int,
    scoring_concurrency: int,
    database_url_env: str = DATABASE_URL_ENV,
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


def destroy_dbos_runtime() -> None:
    DBOS.destroy()
