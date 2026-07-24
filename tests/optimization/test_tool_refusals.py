"""Tool Refusals never masquerade as measurements.

A capacity, budget, validation, or authorization refusal is a typed
non-execution outcome: it carries no evaluation evidence, no Rollout refs, and
no Reward, at every layer -- the Tool Call Store Entry, the Tool Result, the
executor, and the Reward contract.
"""

from __future__ import annotations

import pytest

from tests.optimization.tool_support import (
    evaluate_candidate_config,
    evaluating_executor,
    make_store,
)
from whetstone.optimization import (
    RefusalClass,
    ToolCall,
    ToolCallState,
    ToolCallStore,
    ToolRefusal,
    ToolResult,
)
from whetstone.optimization.tool_eval import ToolValidationError


def _call(config, call_id, *, template="t", route="route-0"):
    return ToolCall(
        call_id=call_id,
        tool_config_hash=config.identity_hash(),
        store_namespace=config.store_namespace,
        args={"model_route": route, "template": template},
    )


def test_capacity_refusal_result_carries_no_evidence_or_reward():
    store = make_store()
    tool_store = ToolCallStore(store)
    config = evaluate_candidate_config(capacity=1)
    handle = evaluating_executor().runtime_handle(config, tool_store)

    accepted = handle(_call(config, "c0"))
    assert accepted.output is not None
    assert accepted.reward is not None

    refused = handle(_call(config, "c1"))
    assert refused.refusal is not None
    assert refused.refusal.refusal_class is RefusalClass.CAPACITY
    # No measurement smuggled onto a refusal.
    assert refused.output is None
    assert refused.reward is None
    assert refused.evaluation_evidence_refs == ()


def test_validation_refusal_debits_no_capacity_and_has_no_measurement():
    store = make_store()
    tool_store = ToolCallStore(store)
    config = evaluate_candidate_config(capacity=5)
    handle = evaluating_executor().runtime_handle(config, tool_store)

    # Empty template -> VALIDATION refusal BEFORE the store accepts.
    refused = handle(_call(config, "bad", template=""))
    assert refused.refusal is not None
    assert refused.refusal.refusal_class is RefusalClass.VALIDATION
    assert refused.output is None
    assert refused.reward is None
    # Capacity untouched: a validation reject is not an accepted call.
    assert tool_store.accepted_count(config.identity_hash()) == 0
    entry = tool_store.get(config.identity_hash(), "bad")
    assert entry is not None
    assert entry.state is ToolCallState.REFUSED
    # The refused Store Entry itself records no capacity debit ordinal.
    assert entry.capacity_debit_ordinal is None


def test_tool_result_schema_forbids_refusal_with_evidence():
    # The ToolResult model itself refuses a refusal carrying evidence/reward.
    with pytest.raises(ValueError, match="refusal never masquerades"):
        ToolResult(
            call_id="x",
            tool_config_ref="tooldef://evaluate_candidate",
            tool_config_hash="a" * 64,
            store_namespace="ns",
            refusal=ToolRefusal(
                refusal_class=RefusalClass.BUDGET, reason="over budget"
            ),
            reward={"reward_name": "reward", "value": 1.0},
        )


def test_store_refuse_rejects_capacity_class():
    store = make_store()
    tool_store = ToolCallStore(store)
    config = evaluate_candidate_config(capacity=5)
    # refuse() records validation/budget/auth refusals, NOT capacity (which is
    # the store's own accounting via accept_or_refuse).
    with pytest.raises(ValueError, match="capacity refusal is decided only"):
        tool_store.refuse(
            _call(config, "c0"),
            config,
            refusal=ToolRefusal(
                refusal_class=RefusalClass.CAPACITY, reason="x"
            ),
        )


def test_budget_refusal_is_a_typed_non_execution_outcome():
    store = make_store()
    tool_store = ToolCallStore(store)
    config = evaluate_candidate_config(capacity=5)
    entry = tool_store.refuse(
        _call(config, "c0"),
        config,
        refusal=ToolRefusal(
            refusal_class=RefusalClass.BUDGET,
            reason="optimization rollout budget exhausted",
        ),
    )
    assert entry.state is ToolCallState.REFUSED
    assert entry.refusal is not None
    assert entry.refusal.refusal_class is RefusalClass.BUDGET
    # A budget refusal debits no capacity and completes no measurement.
    assert entry.capacity_debit_ordinal is None
    assert tool_store.accepted_count(config.identity_hash()) == 0


def test_refused_entry_cannot_be_completed_as_a_measurement():
    store = make_store()
    tool_store = ToolCallStore(store)
    config = evaluate_candidate_config(capacity=5)
    tool_store.refuse(
        _call(config, "c0"),
        config,
        refusal=ToolRefusal(
            refusal_class=RefusalClass.VALIDATION, reason="bad args"
        ),
    )
    # Completing a refused key with a fabricated measurement conflicts: a
    # refusal can never be turned into a measured Tool Result.
    from whetstone.optimization.tool_store import ToolCallStoreConflictError

    fabricated = ToolResult(
        call_id="c0",
        tool_config_ref="tooldef://evaluate_candidate",
        tool_config_hash=config.identity_hash(),
        store_namespace=config.store_namespace,
        output={"rollout_ref_count": 20},
    )
    with pytest.raises(ToolCallStoreConflictError):
        tool_store.complete(config.identity_hash(), fabricated)


def test_validate_args_rejects_missing_route():
    from whetstone.optimization.tool_eval import EvaluatingToolExecutor

    config = evaluate_candidate_config(capacity=5)
    call = ToolCall(
        call_id="c0",
        tool_config_hash=config.identity_hash(),
        store_namespace=config.store_namespace,
        args={"template": "t", "model_route": ""},
    )
    with pytest.raises(ToolValidationError):
        EvaluatingToolExecutor._validate_args(call)
