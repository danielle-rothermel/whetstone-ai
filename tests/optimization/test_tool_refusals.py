"""Tool refusal and completed-result replay contracts."""

from dataclasses import dataclass

import pytest

from whetstone.optimization import (
    EvaluatingToolExecutor,
    RefusalClass,
    RewardPolicy,
    RewardTerm,
    ToolCall,
    ToolCallState,
    ToolCallStore,
    ToolEvaluation,
    ToolRefusal,
    ToolResult,
    typed_ref_for_record,
)

from .support import (
    FULL_B,
    make_store,
    make_tool_definition_config,
)


@dataclass
class CountingEvaluator:
    calls: int = 0

    def evaluate(self, call, config):
        del call, config
        self.calls += 1
        return ToolEvaluation(
            rollout_refs=(typed_ref_for_record("test.rollout", {"value": 1}),),
            aggregates={"score": 1.0},
            eval_config_hash=FULL_B,
        )


def _policy() -> RewardPolicy:
    return RewardPolicy(
        policy_name="tool",
        terms=(RewardTerm(name="score", weight=1.0),),
    )


def _config(*, capacity: int = 2):
    policy = _policy()
    config = make_tool_definition_config(capacity=capacity).model_copy(
        update={"reward_policy_ref": policy.identity_hash()}
    )
    return policy, config


def _call(config, call_id: str, *, template: str = "t") -> ToolCall:
    return ToolCall(
        call_id=call_id,
        tool_config_hash=config.identity_hash(),
        store_namespace=config.store_namespace,
        args={"model_route": "route", "template": template},
    )


def test_completed_call_loads_persisted_result_after_restart(tmp_path) -> None:
    policy, config = _config()
    evaluator = CountingEvaluator()
    first_store = ToolCallStore(make_store(tmp_path))
    handle = EvaluatingToolExecutor(evaluator, policy).runtime_handle(
        config, first_store
    )
    first = handle(_call(config, "c1"))
    assert evaluator.calls == 1
    entry = first_store.get(config.identity_hash(), "c1")
    assert entry is not None and entry.state is ToolCallState.COMPLETED

    fresh_store = ToolCallStore(make_store(tmp_path))
    replay = EvaluatingToolExecutor(evaluator, policy).runtime_handle(
        config, fresh_store
    )(_call(config, "c1"))
    assert replay == first
    assert evaluator.calls == 1


def test_capacity_refusal_has_no_measurement(tmp_path) -> None:
    policy, config = _config(capacity=1)
    evaluator = CountingEvaluator()
    tool_store = ToolCallStore(make_store(tmp_path))
    handle = EvaluatingToolExecutor(evaluator, policy).runtime_handle(
        config, tool_store
    )
    assert handle(_call(config, "c1")).output is not None
    refused = handle(_call(config, "c2"))
    assert refused.refusal is not None
    assert refused.refusal.refusal_class is RefusalClass.CAPACITY
    assert refused.output is None
    assert refused.reward is None
    assert refused.evaluation_evidence_refs == ()
    assert evaluator.calls == 1


def test_validation_refusal_debits_no_capacity(tmp_path) -> None:
    policy, config = _config()
    evaluator = CountingEvaluator()
    tool_store = ToolCallStore(make_store(tmp_path))
    handle = EvaluatingToolExecutor(evaluator, policy).runtime_handle(
        config, tool_store
    )
    refused = handle(_call(config, "bad", template=""))
    assert refused.refusal is not None
    assert refused.refusal.refusal_class is RefusalClass.VALIDATION
    assert tool_store.accepted_count(config.identity_hash()) == 0
    assert evaluator.calls == 0


def test_refused_result_schema_forbids_reward() -> None:
    with pytest.raises(ValueError, match="never masquerades"):
        ToolResult(
            call_id="c",
            tool_config_ref="tool://definition",
            tool_config_hash="a" * 64,
            store_namespace="ns",
            refusal=ToolRefusal(
                refusal_class=RefusalClass.BUDGET,
                reason="budget exhausted",
            ),
            reward={"value": 1.0},
        )
