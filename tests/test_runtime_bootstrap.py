from __future__ import annotations

from whetstone.coordination.runtime_bootstrap import register_runtime
from dr_store.sync import BlockingObjectStore


def test_register_runtime_is_idempotent_for_same_store(sqlite_store) -> None:
    first = register_runtime(store=sqlite_store)
    second = register_runtime(store=sqlite_store)
    assert first.harness is not second.harness
    assert first.controller.runtime_hash != second.controller.runtime_hash


def test_register_runtime_opens_store_when_omitted(tmp_path) -> None:
    runtime = register_runtime(sqlite_path=str(tmp_path / "runtime.sqlite"))
    assert isinstance(runtime.store, BlockingObjectStore)
    assert runtime.store.resolve("absent") is None
