"""Application-owned runtime configuration and safe DBOS lifecycle helpers."""

from __future__ import annotations

import os

from dbos import DBOS, DBOSConfig
from dr_platform.dbos_config import normalize_postgresql_driver_url
from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, StrictStr

DATABASE_URL_ENV = "DATABASE_URL"
DBOS_SYSTEM_DATABASE_URL_ENV = "DBOS_SYSTEM_DATABASE_URL"


class WhetstoneDbosConfig(BaseModel):
    """Resolved process configuration; never pass this model to a workflow."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    database_url: StrictStr
    system_database_url: StrictStr
    generation_concurrency: StrictInt
    scoring_concurrency: StrictInt
    enable_otlp: StrictBool = False
    otlp_traces_endpoints: tuple[StrictStr, ...] = ()


def resolve_application_database_url(database_url: str | None = None) -> str:
    """Resolve the app database only at an application/step boundary."""
    resolved = database_url or os.environ.get(DATABASE_URL_ENV)
    if not resolved:
        raise ValueError(
            f"{DATABASE_URL_ENV} is required for Whetstone execution"
        )
    return normalize_postgresql_driver_url(resolved)


def build_whetstone_dbos_config(
    *,
    database_url: str | None,
    system_database_url: str | None,
    generation_concurrency: int,
    scoring_concurrency: int,
    enable_otlp: bool = False,
    otlp_traces_endpoints: tuple[str, ...] = (),
) -> WhetstoneDbosConfig:
    app_url = resolve_application_database_url(database_url)
    return WhetstoneDbosConfig(
        database_url=app_url,
        system_database_url=normalize_postgresql_driver_url(
            system_database_url
            or os.environ.get(DBOS_SYSTEM_DATABASE_URL_ENV)
            or app_url
        ),
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
        enable_otlp=enable_otlp,
        otlp_traces_endpoints=otlp_traces_endpoints,
    )


def dbos_config(config: WhetstoneDbosConfig, *, app_name: str) -> DBOSConfig:
    result: DBOSConfig = {
        "name": app_name,
        "system_database_url": config.system_database_url,
        "enable_otlp": config.enable_otlp,
        "otel_attribute_format": "semconv",
    }
    if config.otlp_traces_endpoints:
        result["otlp_traces_endpoints"] = list(config.otlp_traces_endpoints)
    return result


def shutdown_dbos_runtime() -> None:
    """Tear down only Whetstone's process-local DBOS runtime."""
    DBOS.destroy()
