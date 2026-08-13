from __future__ import annotations

from whetstone.coordination.runtime_bootstrap import register_runtime


def test_register_runtime_is_idempotent_for_same_store(sqlite_store) -> None:
    first = register_runtime(store=sqlite_store)
    second = register_runtime(store=sqlite_store)
    assert first.harness is not second.harness
    assert first.controller.runtime_hash != second.controller.runtime_hash
