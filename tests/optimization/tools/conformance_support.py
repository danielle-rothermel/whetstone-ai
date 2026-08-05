"""Backend construction for public Tool admission conformance tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Literal

from dr_store import MemoryBackend, ObjectStore

from tests.postgres import (
    connect_in_postgres_schema,
    isolated_postgres_schema,
)
from whetstone.core.effects.authority import EffectAuthority
from whetstone.optimization.tools.facade import (
    ToolAdmissionAuthority,
    ToolCallStore,
)

ToolAdmissionBackend = Literal["memory", "sqlite", "postgresql"]


@contextmanager
def conformance_store(
    backend: ToolAdmissionBackend,
    tmp_path: Path,
) -> Iterator[ToolCallStore]:
    """Yield one isolated store using the selected admission backend."""
    if backend == "memory":
        store = ToolCallStore(
            ObjectStore(MemoryBackend()),
            ToolAdmissionAuthority.memory(),
            EffectAuthority.memory(),
        )
        try:
            yield store
        finally:
            store.close()
        return

    if backend == "sqlite":
        database = tmp_path / "admission-conformance.sqlite"
        store = ToolCallStore(
            ObjectStore(MemoryBackend()),
            ToolAdmissionAuthority.sqlite(database),
            EffectAuthority.memory(),
        )
        try:
            yield store
        finally:
            store.close()
        return

    with isolated_postgres_schema("tool_conformance") as schema:
        connect_in_schema = partial(
            connect_in_postgres_schema,
            schema=schema.name,
        )
        store = ToolCallStore(
            ObjectStore(MemoryBackend()),
            ToolAdmissionAuthority.postgresql(
                schema.dsn,
                _connect=connect_in_schema,
            ),
            EffectAuthority.memory(),
        )
        try:
            yield store
        finally:
            store.close()
