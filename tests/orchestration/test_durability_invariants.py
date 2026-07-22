"""Structural durability invariants for the Whetstone executor.

AST-level proofs (DB-free, deterministic) that the durable executor uses
exactly ``@DBOS.step(retries_allowed=False)`` for the per-attempt provider
call and enables no automatic DBOS step retry anywhere — the Whetstone-owned
half of "dr-providers native retries and DBOS automatic step retries are
demonstrably zero." Whetstone alone bounds and backs off.
"""

from __future__ import annotations

import ast
from pathlib import Path

import whetstone.orchestration.executor as executor_module

_SOURCE = Path(executor_module.__file__).read_text(encoding="utf-8")
_TREE = ast.parse(_SOURCE)


def _dbos_step_calls() -> list[ast.Call]:
    calls: list[ast.Call] = []
    for node in ast.walk(_TREE):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "step"
            and isinstance(func.value, ast.Name)
            and func.value.id == "DBOS"
        ):
            calls.append(node)
    return calls


def test_provider_call_uses_exactly_one_dbos_step() -> None:
    calls = _dbos_step_calls()
    assert len(calls) == 1


def test_the_dbos_step_disables_retries() -> None:
    (call,) = _dbos_step_calls()
    retries = {
        keyword.arg: keyword.value
        for keyword in call.keywords
        if keyword.arg == "retries_allowed"
    }
    assert "retries_allowed" in retries
    value = retries["retries_allowed"]
    assert isinstance(value, ast.Constant)
    assert value.value is False


def test_no_automatic_retry_keyword_is_enabled_on_any_dbos_call() -> None:
    """No DBOS call enables automatic step retries.

    ``max_attempts`` / ``max_retries`` / a truthy ``retries_allowed`` would
    each turn on automatic step retry; none may appear on any ``DBOS.*`` call
    in the executor source.
    """
    banned = {"max_attempts", "max_retries", "backoff_rate"}
    for node in ast.walk(_TREE):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "DBOS"
        ):
            continue
        for keyword in node.keywords:
            assert keyword.arg not in banned, (
                f"DBOS.{func.attr} enables automatic retry via {keyword.arg}"
            )
            if keyword.arg == "retries_allowed":
                assert isinstance(keyword.value, ast.Constant)
                assert keyword.value.value is False


def test_backoff_uses_durable_dbos_sleep() -> None:
    """Backoff sleeps durably via ``DBOS.sleep`` (not wall-clock sleep)."""
    dbos_sleep_calls = [
        node
        for node in ast.walk(_TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "sleep"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "DBOS"
    ]
    assert len(dbos_sleep_calls) == 1
    # No time.sleep in the durable executor.
    assert "time.sleep" not in _SOURCE
