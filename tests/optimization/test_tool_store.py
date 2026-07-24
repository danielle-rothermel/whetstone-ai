"""Authoritative Tool Call Store: atomic state machine + concurrency."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from dr_store import ObjectStore, SqliteBackend

from whetstone.optimization import (
    RefusalClass,
    ToolCall,
    ToolCallState,
    ToolCallStore,
    ToolCallStoreConflictError,
    ToolResult,
)

from .support import make_store, make_tool_definition_config


def _call(call_id: str, cfg, *, args=None) -> ToolCall:
    return ToolCall(
        call_id=call_id,
        tool_config_hash=cfg.identity_hash(),
        store_namespace=cfg.store_namespace,
        args=args or {"model_route": "r0", "template": call_id},
    )


def _completed_result(call: ToolCall, cfg) -> ToolResult:
    return ToolResult(
        call_id=call.call_id,
        tool_config_ref="toolcfg://x",
        tool_config_hash=cfg.identity_hash(),
        store_namespace=cfg.store_namespace,
        output={"rollout_refs": []},
    )


def test_absent_to_accepted_debits_capacity_once(tmp_path) -> None:
    cfg = make_tool_definition_config(capacity=2)
    store = ToolCallStore(make_store(tmp_path))
    tch = cfg.identity_hash()
    entry = store.accept_or_refuse(_call("c1", cfg), cfg)
    assert entry.state is ToolCallState.ACCEPTED
    assert entry.capacity_debit_ordinal == 1
    assert store.accepted_count(tch) == 1
    # Replay of the same key + same args is idempotent; no second debit.
    replay = store.accept_or_refuse(_call("c1", cfg), cfg)
    assert replay == entry
    assert store.accepted_count(tch) == 1


def test_capacity_exhaustion_refuses_without_debit(tmp_path) -> None:
    cfg = make_tool_definition_config(capacity=1)
    store = ToolCallStore(make_store(tmp_path))
    store.accept_or_refuse(_call("c1", cfg), cfg)
    refused = store.accept_or_refuse(_call("c2", cfg), cfg)
    assert refused.state is ToolCallState.REFUSED
    assert refused.refusal is not None
    assert refused.refusal.refusal_class is RefusalClass.CAPACITY
    assert refused.capacity_debit_ordinal is None
    # Capacity consumption is still exactly 1 (the refusal debited nothing).
    assert store.accepted_count(cfg.identity_hash()) == 1


def test_accepted_to_completed_transition(tmp_path) -> None:
    cfg = make_tool_definition_config(capacity=1)
    store = ToolCallStore(make_store(tmp_path))
    call = _call("c1", cfg)
    store.accept_or_refuse(call, cfg)
    result = _completed_result(call, cfg)
    completed = store.complete(cfg.identity_hash(), result)
    assert completed.state is ToolCallState.COMPLETED
    assert completed.tool_result_ref is not None
    # Idempotent replay of the same completion.
    again = store.complete(cfg.identity_hash(), result)
    assert again == completed


def test_divergent_args_for_existing_key_conflicts(tmp_path) -> None:
    cfg = make_tool_definition_config(capacity=2)
    store = ToolCallStore(make_store(tmp_path))
    store.accept_or_refuse(_call("c1", cfg, args={"a": 1}), cfg)
    with pytest.raises(ToolCallStoreConflictError):
        store.accept_or_refuse(_call("c1", cfg, args={"a": 2}), cfg)


def test_divergent_completion_conflicts_and_preserves_winner(tmp_path) -> None:
    cfg = make_tool_definition_config(capacity=1)
    store = ToolCallStore(make_store(tmp_path))
    call = _call("c1", cfg)
    store.accept_or_refuse(call, cfg)
    store.complete(cfg.identity_hash(), _completed_result(call, cfg))
    # A different Tool Result for the same key conflicts; the winner stays.
    divergent = ToolResult(
        call_id="c1",
        tool_config_ref="toolcfg://x",
        tool_config_hash=cfg.identity_hash(),
        store_namespace=cfg.store_namespace,
        output={"rollout_refs": ["different"]},
    )
    with pytest.raises(ToolCallStoreConflictError) as exc:
        store.complete(cfg.identity_hash(), divergent)
    assert exc.value.existing.state is ToolCallState.COMPLETED


def test_completing_absent_key_conflicts(tmp_path) -> None:
    cfg = make_tool_definition_config(capacity=1)
    store = ToolCallStore(make_store(tmp_path))
    call = _call("c1", cfg)
    with pytest.raises(ToolCallStoreConflictError):
        store.complete(cfg.identity_hash(), _completed_result(call, cfg))


def test_concurrent_capacity_race_debits_each_slot_once(tmp_path) -> None:
    # 32 distinct calls race against capacity 8: exactly 8 accepted, 24
    # refused, and the accepted ordinals are exactly 1..8 (each slot once).
    cfg = make_tool_definition_config(capacity=8)
    store = ToolCallStore(make_store(tmp_path))
    call_ids = [f"c{i}" for i in range(32)]

    def worker(cid: str):
        return store.accept_or_refuse(_call(cid, cfg), cfg)

    with ThreadPoolExecutor(max_workers=16) as pool:
        entries = list(pool.map(worker, call_ids))

    accepted = [e for e in entries if e.state is ToolCallState.ACCEPTED]
    refused = [e for e in entries if e.state is ToolCallState.REFUSED]
    assert len(accepted) == 8
    assert len(refused) == 24
    ordinals = sorted(e.capacity_debit_ordinal for e in accepted)
    assert ordinals == list(range(1, 9))
    assert store.accepted_count(cfg.identity_hash()) == 8


def test_concurrent_same_key_replay_yields_one_winner(tmp_path) -> None:
    # Many concurrent replays of the SAME key + args resolve to one entry and
    # debit exactly one capacity slot.
    cfg = make_tool_definition_config(capacity=4)
    store = ToolCallStore(make_store(tmp_path))

    def worker(_i: int):
        return store.accept_or_refuse(_call("c1", cfg), cfg)

    with ThreadPoolExecutor(max_workers=16) as pool:
        entries = list(pool.map(worker, range(16)))

    ordinals = {e.capacity_debit_ordinal for e in entries}
    assert ordinals == {1}
    assert store.accepted_count(cfg.identity_hash()) == 1


def test_fresh_store_over_same_backend_restores_state_and_capacity(
    tmp_path,
) -> None:
    # Restart durability: a FRESH ToolCallStore over the SAME dr-store backend
    # observes the accepted+completed decision and the debited capacity, and
    # re-accepting the same call re-debits nothing.
    cfg = make_tool_definition_config(capacity=2)
    tch = cfg.identity_hash()
    backend = make_store(tmp_path)

    s1 = ToolCallStore(backend)
    call = _call("c1", cfg)
    s1.accept_or_refuse(call, cfg)
    s1.complete(tch, _completed_result(call, cfg))
    assert s1.accepted_count(tch) == 1

    # New store instance, same backend: state and capacity are reconstructed.
    s2 = ToolCallStore(backend)
    entry = s2.get(tch, "c1")
    assert entry is not None
    assert entry.state is ToolCallState.COMPLETED
    assert s2.accepted_count(tch) == 1  # restored, not zero
    # Re-accepting the same call is idempotent and debits nothing new.
    replay = s2.accept_or_refuse(call, cfg)
    assert replay.state is ToolCallState.COMPLETED
    assert s2.accepted_count(tch) == 1


def test_capacity_is_exactly_once_across_sqlite_restart(tmp_path) -> None:
    # SQLite-backed cross-process durability: capacity slots and the accept
    # decision survive dropping the ObjectStore and reopening the same file.
    cfg = make_tool_definition_config(capacity=1)
    tch = cfg.identity_hash()
    db = tmp_path / "toolstore.sqlite"

    s1 = ToolCallStore(ObjectStore(SqliteBackend(db)))
    s1.accept_or_refuse(_call("c1", cfg), cfg)
    assert s1.accepted_count(tch) == 1

    # Reopen the same durable file in a brand-new store: capacity is consumed,
    # so a different call is refused (never a second debit).
    s2 = ToolCallStore(ObjectStore(SqliteBackend(db)))
    assert s2.accepted_count(tch) == 1
    refused = s2.accept_or_refuse(_call("c2", cfg), cfg)
    assert refused.state is ToolCallState.REFUSED
    assert refused.refusal is not None
    assert refused.refusal.refusal_class is RefusalClass.CAPACITY
    assert s2.accepted_count(tch) == 1
