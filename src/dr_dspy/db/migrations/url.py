from __future__ import annotations

DATABASE_URL_ENV = "DATABASE_URL"
POSTGRESQL_URL_PREFIX = "postgresql://"
POSTGRESQL_PSYCOPG_URL_PREFIX = "postgresql+psycopg://"


def normalize_postgresql_driver_url(database_url: str) -> str:
    if database_url.startswith(POSTGRESQL_URL_PREFIX):
        return database_url.replace(
            POSTGRESQL_URL_PREFIX,
            POSTGRESQL_PSYCOPG_URL_PREFIX,
            1,
        )
    return database_url
