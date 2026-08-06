from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import pytest
from psycopg import connect
from psycopg.sql import SQL, Identifier


@dataclass(frozen=True, slots=True)
class PostgresTestSchema:
    dsn: str
    name: str


class _GatedPostgresCursor:
    def __init__(
        self,
        cursor: Any,
        *,
        before_query: str | None,
        before_query_reached: Any,
        after_query: str | None,
        after_query_reached: Any,
        release: Any,
    ) -> None:
        self._cursor = cursor
        self._before_query = before_query
        self._before_query_reached = before_query_reached
        self._after_query = after_query
        self._after_query_reached = after_query_reached
        self._release = release

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    def execute(
        self,
        query: str,
        params: tuple[Any, ...] | None = None,
    ) -> Any:
        normalized = " ".join(query.split())
        if self._before_query is not None and self._before_query in normalized:
            self._before_query_reached.set()
        result = self._cursor.execute(query, params)
        if self._after_query is not None and self._after_query in normalized:
            self._after_query_reached.set()
            if not self._release.wait(timeout=60):
                raise TimeoutError(
                    "PostgreSQL holder transaction was not released"
                )
        return result

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._cursor.fetchone()

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._cursor.fetchall()

    def __enter__(self) -> _GatedPostgresCursor:
        self._cursor.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self._cursor.__exit__(*args)


class _GatedPostgresConnection:
    def __init__(
        self,
        connection: Any,
        *,
        before_query: str | None,
        before_query_reached: Any,
        after_query: str | None,
        after_query_reached: Any,
        release: Any,
    ) -> None:
        self._connection = connection
        self._before_query = before_query
        self._before_query_reached = before_query_reached
        self._after_query = after_query
        self._after_query_reached = after_query_reached
        self._release = release

    def cursor(self) -> _GatedPostgresCursor:
        return _GatedPostgresCursor(
            self._connection.cursor(),
            before_query=self._before_query,
            before_query_reached=self._before_query_reached,
            after_query=self._after_query,
            after_query_reached=self._after_query_reached,
            release=self._release,
        )

    def __enter__(self) -> _GatedPostgresConnection:
        self._connection.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self._connection.__exit__(*args)


class PostgresOperationGate:
    def __init__(
        self,
        *,
        schema: str,
        backend_pid: Any,
        before_query: str | None = None,
        before_query_reached: Any = None,
        after_query: str | None = None,
        after_query_reached: Any = None,
        release: Any = None,
    ) -> None:
        self._schema = schema
        self._backend_pid = backend_pid
        self._before_query = before_query
        self._before_query_reached = before_query_reached
        self._after_query = after_query
        self._after_query_reached = after_query_reached
        self._release = release
        self._connection_count = 0

    def __call__(self, dsn: str) -> Any:
        self._connection_count += 1
        connection = connect_in_postgres_schema(dsn, schema=self._schema)
        if self._connection_count != 2:
            return connection
        self._backend_pid.value = connection.info.backend_pid
        return _GatedPostgresConnection(
            connection,
            before_query=self._before_query,
            before_query_reached=self._before_query_reached,
            after_query=self._after_query,
            after_query_reached=self._after_query_reached,
            release=self._release,
        )


def connect_in_postgres_schema(
    dsn: str,
    *,
    schema: str,
) -> Any:
    connection = connect(dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                SQL("SET search_path TO {}, pg_catalog").format(
                    Identifier(schema)
                )
            )
        connection.commit()
    except BaseException:
        connection.close()
        raise
    return connection


def require_postgres_lock_wait(
    schema: PostgresTestSchema,
    backend_pid: int,
) -> None:
    with connect_in_postgres_schema(
        schema.dsn,
        schema=schema.name,
    ) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            for _ in range(1_000):
                cursor.execute(
                    """
                    SELECT state, wait_event_type
                    FROM pg_catalog.pg_stat_activity
                    WHERE pid = %s
                    """,
                    (backend_pid,),
                )
                if cursor.fetchone() == ("active", "Lock"):
                    return
    raise AssertionError(
        f"PostgreSQL backend {backend_pid} did not enter a lock wait"
    )


@contextmanager
def isolated_postgres_schema(prefix: str) -> Iterator[PostgresTestSchema]:
    dsn = os.environ.get("WHETSTONE_TEST_POSTGRES_DSN")
    if dsn is None:
        pytest.skip(
            "WHETSTONE_TEST_POSTGRES_DSN is not configured; PostgreSQL "
            "integration did not run"
        )
    schema = f"{prefix}_{uuid4().hex}"
    with connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(SQL("CREATE SCHEMA {}").format(Identifier(schema)))
            cursor.execute(
                SQL("SET LOCAL search_path TO {}, pg_catalog").format(
                    Identifier(schema)
                )
            )
            cursor.execute("SELECT current_schema()")
            assert cursor.fetchone() == (schema,)

    test_schema = PostgresTestSchema(dsn=dsn, name=schema)
    try:
        yield test_schema
    finally:
        with connect_in_postgres_schema(dsn, schema=schema) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    SQL("DROP SCHEMA {} CASCADE").format(Identifier(schema))
                )
