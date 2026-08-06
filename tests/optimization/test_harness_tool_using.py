from typing import Any, cast

import pytest

from whetstone.core.effects.authority import EffectAuthority
from whetstone.core.effects.models import ReplayPolicy
from whetstone.optimization.contracts import BudgetState
from whetstone.optimization.tools.admission import ToolCallState
from whetstone.optimization.tools.contracts import (
    ToolCapacityScope,
    tool_capacity_binding,
)
from whetstone.optimization.tools.execution import EvaluatingToolExecutor
from whetstone.optimization.tools.facade import (
    ToolAdmissionAuthority,
    ToolCallStore,
)

from .support import (
    RecordingToolExecutor,
    ToolUsingAdapter,
    internal_reward_policy,
    make_harness,
    make_store,
    make_tool_definition_config,
    registry,
    tool_request,
)


def test_tool_results_and_store_entries_are_step_evidence(tmp_path) -> None:
    store = make_store(tmp_path)
    adapter = ToolUsingAdapter(call_ids=("c1",))
    authority = EffectAuthority.memory()
    executor = RecordingToolExecutor(authority)
    request = tool_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=request.run,
        effect_authority=authority,
        tool_executor=executor,
    )
    result, _ = harness.run_step(request)
    evidence = result.tool_evidence[0]
    assert evidence.store_entry.state is ToolCallState.COMPLETED
    assert evidence.store_entry.tool_result_ref == evidence.result.record_ref
    assert adapter.invocations == 1
    assert executor.handles_built == 1


def test_zero_tool_budget_refuses_before_executor_dispatch(
    tmp_path,
) -> None:
    adapter = ToolUsingAdapter(call_ids=("c1",))
    request = tool_request().model_copy(
        update={"budget": BudgetState(remaining={"tool_calls": 0})}
    )
    first_store = make_store(tmp_path)
    admission_path = tmp_path / "optimization.sqlite"
    effect_path = tmp_path / "effects.sqlite"
    authority = EffectAuthority.sqlite(effect_path)
    executor = RecordingToolExecutor(authority)
    tool_store = ToolCallStore(
        first_store,
        ToolAdmissionAuthority.sqlite(admission_path),
        authority,
    )
    first = make_harness(
        store=first_store,
        adapter_registry=registry(adapter),
        run=request.run,
        tool_store=tool_store,
        effect_authority=authority,
        tool_executor=executor,
    )
    with pytest.raises(ValueError, match="only 0"):
        first.run_step(request)
    assert adapter.invocations == 1
    assert executor.calls == []
    config = request.tool_configs[0].record
    binding = tool_capacity_binding(
        ToolCapacityScope.RUN, request.run.record_ref
    )
    assert tool_store.accepted_count(config, binding) == 0
    assert first.resolve_step_result(request.run_id, 0) is None


def test_tool_step_requires_executor_before_invocation(tmp_path) -> None:
    store = make_store(tmp_path)
    adapter = ToolUsingAdapter()
    request = tool_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=request.run,
    )
    for _attempt in range(2):
        with pytest.raises(ValueError, match="ToolExecutor"):
            harness.run_step(request)
    assert adapter.invocations == 0


def test_invalid_tool_replay_policy_repeats_without_adapter_lease(
    tmp_path,
) -> None:
    store = make_store(tmp_path)
    adapter = ToolUsingAdapter()
    authority = EffectAuthority.memory()
    reward_policy = internal_reward_policy()
    config = make_tool_definition_config().model_copy(
        update={"reward_policy_hash": reward_policy.identity_hash()}
    )
    request = tool_request(config=config)
    executor = EvaluatingToolExecutor(
        cast(Any, object()),
        reward_policy,
        authority,
        owner_id="wrong-policy-test",
        replay_policy=ReplayPolicy.NO_REDRIVE,
    )
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=request.run,
        effect_authority=authority,
        tool_executor=executor,
    )

    for _attempt in range(2):
        with pytest.raises(ValueError, match="ReplayPolicy disagrees"):
            harness.run_step(request)

    assert adapter.invocations == 0
