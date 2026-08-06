from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import Event

import pytest

from tests.optimization.support import (
    RecordingToolExecutor,
    ToolUsingAdapter,
    make_harness,
    make_store,
    make_tool_definition_config,
    output_contract,
    proposal_request,
    proposed_candidate,
    registry,
    tool_request,
    tool_run,
)
from whetstone.core.effects.authority import EffectAuthority
from whetstone.core.effects.models import ReplayPolicy
from whetstone.core.identity import (
    TerminalFailure,
    TypedRef,
    typed_ref_for_record,
)
from whetstone.optimization.adapters import AdapterOutput
from whetstone.optimization.contracts import (
    BudgetState,
    OptimizationStepRequest,
    StepKind,
    StepMode,
    StepStatus,
    step_request_reference,
)
from whetstone.optimization.run_store import (
    HISTORY_SNAPSHOT_SCHEMA,
    STATE_SNAPSHOT_SCHEMA,
)
from whetstone.optimization.tools.contracts import (
    RefusalClass,
    RuntimeToolHandle,
    ToolCall,
    ToolCapacityBinding,
    ToolCapacityScope,
    ToolConfig,
    ToolRefusal,
    tool_call_reference,
    tool_capacity_binding,
    tool_config_reference,
)
from whetstone.optimization.tools.facade import ToolCallStore
from whetstone.optimization.tools.issued import (
    ISSUED_TOOL_CALL_CLAIM_SCHEMA,
    ISSUED_TOOL_CALL_KEY_PREFIX,
    ISSUED_TOOL_CALL_SLOT_KEY_PREFIX,
    ISSUED_TOOL_CALL_SLOT_SCHEMA,
    ISSUED_TOOL_CALL_TERMINAL_KEY_PREFIX,
    ISSUED_TOOL_CALL_TERMINAL_SCHEMA,
    IssuedToolCallConflictError,
    _issued_tool_call_binding_key,
    _issued_tool_call_slot_binding_key,
    _issued_tool_call_terminal_binding_key,
    _IssuedToolCallClaim,
    _IssuedToolCallClaimRef,
    _IssuedToolCallSlot,
)


def _call(
    handle: RuntimeToolHandle,
    *,
    call_id: str,
    template: str = "candidate",
) -> ToolCall:
    return ToolCall(
        call_id=call_id,
        tool_config=handle.tool_config_ref,
        capacity_binding=handle.binding,
        args={"model_route": "r0", "template": template},
    )


def _successful_output(
    request: OptimizationStepRequest,
    *,
    status: StepStatus = StepStatus.COMPLETE,
) -> AdapterOutput:
    proposed = proposed_candidate(
        request.candidates[0],
        f"proposal-{request.step_index}",
        text=f"tool-{request.step_index}",
    )
    return AdapterOutput(
        proposed_candidates=(proposed,),
        accepted_candidates=() if status is StepStatus.FAILED else (proposed,),
        proposed_status=status,
        terminal_failure=(
            TerminalFailure(
                code="adapter_failed",
                message="adapter stopped after a terminal tool call",
            )
            if status is StepStatus.FAILED
            else None
        ),
    )


@dataclass
class MutableClock:
    current: datetime

    def __call__(self) -> datetime:
        return self.current


class CrashAfterTerminalAdapter:
    def __init__(self, *, divergent_on_recovery: bool = False) -> None:
        self.invocations = 0
        self.divergent_on_recovery = divergent_on_recovery

    @property
    def key(self) -> str:
        return "tool-test"

    @property
    def mode(self) -> StepMode:
        return StepMode.TOOL_USING

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.IDEMPOTENT

    def invoke(
        self,
        request: OptimizationStepRequest,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        self.invocations += 1
        index = 1 if self.invocations > 1 and self.divergent_on_recovery else 0
        handle = handles[index]
        handle(_call(handle, call_id="recoverable"))
        if self.invocations == 1:
            raise RuntimeError("adapter crashed after terminal tool result")
        return _successful_output(request)


class CrashThenReplayAdapter:
    def __init__(
        self,
        *,
        recovery_call_ids: tuple[str, ...],
    ) -> None:
        self.invocations = 0
        self._recovery_call_ids = recovery_call_ids

    @property
    def key(self) -> str:
        return "tool-test"

    @property
    def mode(self) -> StepMode:
        return StepMode.TOOL_USING

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.IDEMPOTENT

    def invoke(
        self,
        request: OptimizationStepRequest,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        self.invocations += 1
        call_ids = (
            ("first", "second")
            if self.invocations == 1
            else self._recovery_call_ids
        )
        for call_id in call_ids:
            handles[0](_call(handles[0], call_id=call_id))
        if self.invocations == 1:
            raise RuntimeError("adapter crashed after two terminal calls")
        return _successful_output(request)


class DivergentBeforeUnresolvedAdapter:
    def __init__(self) -> None:
        self.invocations = 0

    @property
    def key(self) -> str:
        return "tool-test"

    @property
    def mode(self) -> StepMode:
        return StepMode.TOOL_USING

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.IDEMPOTENT

    def invoke(
        self,
        request: OptimizationStepRequest,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        self.invocations += 1
        call_id = "unresolved" if self.invocations == 1 else "divergent"
        handles[0](_call(handles[0], call_id=call_id))
        return _successful_output(request)


class CrashFirstToolExecutor(RecordingToolExecutor):
    def runtime_handle(
        self,
        config: ToolConfig,
        store: ToolCallStore,
        binding: ToolCapacityBinding,
    ) -> RuntimeToolHandle:
        inner = super().runtime_handle(config, store, binding)

        def execute(call: ToolCall):
            if not self.calls:
                self.calls.append(call)
                raise RuntimeError("tool crashed before terminal persistence")
            return inner(call)

        return RuntimeToolHandle(config, binding, execute)


class RefusingToolExecutor:
    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    def runtime_handle(
        self,
        config: ToolConfig,
        store: ToolCallStore,
        binding: ToolCapacityBinding,
    ) -> RuntimeToolHandle:
        def execute(call: ToolCall):
            self.calls.append(call)
            entry = store.refuse(
                call,
                config,
                refusal=ToolRefusal(
                    refusal_class=RefusalClass.VALIDATION,
                    reason="candidate is invalid",
                ),
            )
            return store.load_terminal_result(entry)

        return RuntimeToolHandle(config, binding, execute)


class FailedAfterTerminalAdapter:
    @property
    def key(self) -> str:
        return "tool-test"

    @property
    def mode(self) -> StepMode:
        return StepMode.TOOL_USING

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.IDEMPOTENT

    def invoke(
        self,
        request: OptimizationStepRequest,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        handles[0](_call(handles[0], call_id="before-failure"))
        return _successful_output(request, status=StepStatus.FAILED)


class SameCallEveryStepAdapter:
    @property
    def key(self) -> str:
        return "tool-test"

    @property
    def mode(self) -> StepMode:
        return StepMode.TOOL_USING

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.IDEMPOTENT

    def invoke(
        self,
        request: OptimizationStepRequest,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        handles[0](_call(handles[0], call_id="same-step-local-id"))
        status = (
            StepStatus.CONTINUE
            if request.step_index == 0
            else StepStatus.COMPLETE
        )
        return _successful_output(request, status=status)


class SameIdAcrossConfigsAdapter:
    @property
    def key(self) -> str:
        return "tool-test"

    @property
    def mode(self) -> StepMode:
        return StepMode.TOOL_USING

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.IDEMPOTENT

    def invoke(
        self,
        request: OptimizationStepRequest,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        handles[0](_call(handles[0], call_id="shared"))
        handles[1](_call(handles[1], call_id="shared"))
        raise AssertionError("second config must not dispatch the shared ID")


class BlockingToolExecutor(RecordingToolExecutor):
    def __init__(
        self,
        authority: EffectAuthority,
        *,
        entered: Event,
        release: Event,
    ) -> None:
        super().__init__(authority)
        self._entered = entered
        self._release = release

    def runtime_handle(
        self,
        config: ToolConfig,
        store: ToolCallStore,
        binding: ToolCapacityBinding,
    ) -> RuntimeToolHandle:
        inner = super().runtime_handle(config, store, binding)

        def execute(call: ToolCall):
            self._entered.set()
            if not self._release.wait(timeout=10):
                raise AssertionError(
                    "concurrent test did not release executor"
                )
            return inner(call)

        return RuntimeToolHandle(config, binding, execute)


class ConcurrentDuplicateAdapter:
    def __init__(self, *, entered: Event, release: Event) -> None:
        self._entered = entered
        self._release = release

    @property
    def key(self) -> str:
        return "tool-test"

    @property
    def mode(self) -> StepMode:
        return StepMode.TOOL_USING

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.IDEMPOTENT

    def invoke(
        self,
        request: OptimizationStepRequest,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        call = _call(handles[0], call_id="concurrent")
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(handles[0], call)
            if not self._entered.wait(timeout=10):
                raise AssertionError("first call never reached executor")
            try:
                handles[0](call)
            finally:
                self._release.set()
            first.result(timeout=10)
        raise AssertionError("concurrent duplicate must fail")


class SnapshotAdapter:
    @property
    def key(self) -> str:
        return "proposal-test"

    @property
    def mode(self) -> StepMode:
        return StepMode.PROPOSAL_ONLY

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.IDEMPOTENT

    def invoke(
        self,
        request: OptimizationStepRequest,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        assert handles == ()
        proposed = proposed_candidate(
            request.candidates[0], "snapshot", text="snapshot"
        )
        return AdapterOutput(
            proposed_candidates=(proposed,),
            accepted_candidates=(proposed,),
            proposed_status=StepStatus.COMPLETE,
            state_delta={
                "optimizer": {
                    "round": 2,
                    "frontier": [
                        {"candidate_id": "candidate-a", "score": 0.75},
                        {"candidate_id": "candidate-b", "score": None},
                    ],
                }
            },
            history_delta={
                "events": [
                    {
                        "kind": "proposal",
                        "metadata": {"accepted": True, "tags": ["best"]},
                    }
                ]
            },
        )


def test_nested_immutable_snapshots_persist_as_exact_strict_json(
    tmp_path,
) -> None:
    store = make_store(tmp_path)
    request = proposal_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(SnapshotAdapter()),
        run=request.run,
    )

    result, _ = harness.run_step(request)

    assert result.state_ref is not None
    assert store.get(result.state_ref.reference) == {
        "optimizer": {
            "round": 2,
            "frontier": [
                {"candidate_id": "candidate-a", "score": 0.75},
                {"candidate_id": "candidate-b", "score": None},
            ],
        }
    }
    assert result.history_ref is not None
    assert store.get(result.history_ref.reference) == {
        "events": [
            {
                "kind": "proposal",
                "metadata": {"accepted": True, "tags": ["best"]},
            }
        ]
    }


def test_terminal_recovery_uses_ledger_without_second_executor_call(
    tmp_path,
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    authority = EffectAuthority.memory(clock=clock)
    store = make_store(tmp_path)
    executor = RecordingToolExecutor(authority)
    adapter = CrashAfterTerminalAdapter()
    request = tool_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=request.run,
        effect_authority=authority,
        tool_executor=executor,
        lease_duration=timedelta(seconds=1),
    )

    with pytest.raises(RuntimeError, match="crashed"):
        harness.run_step(request)
    clock.current += timedelta(seconds=2)

    result, _ = harness.run_step(request)

    assert adapter.invocations == 2
    assert [call.call_id for call in executor.calls] == ["recoverable"]
    assert len(result.tool_evidence) == 1
    assert result.budget_delta.consumed == {"tool_calls": 1}


def test_recovery_must_replay_terminal_prefix_in_exact_order(
    tmp_path,
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    authority = EffectAuthority.memory(clock=clock)
    executor = RecordingToolExecutor(authority)
    adapter = CrashThenReplayAdapter(recovery_call_ids=("second", "first"))
    request = tool_request()
    harness = make_harness(
        store=make_store(tmp_path),
        adapter_registry=registry(adapter),
        run=request.run,
        effect_authority=authority,
        tool_executor=executor,
        lease_duration=timedelta(seconds=1),
    )

    with pytest.raises(RuntimeError, match="two terminal calls"):
        harness.run_step(request)
    clock.current += timedelta(seconds=2)

    with pytest.raises(
        ValueError, match=r"prefix in order.*expected ordinal 0"
    ):
        harness.run_step(request)

    assert [call.call_id for call in executor.calls] == ["first", "second"]


def test_recovery_cannot_skip_a_terminal_prefix_call_before_checkpoint(
    tmp_path,
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    authority = EffectAuthority.memory(clock=clock)
    executor = RecordingToolExecutor(authority)
    adapter = CrashThenReplayAdapter(recovery_call_ids=("first",))
    request = tool_request()
    harness = make_harness(
        store=make_store(tmp_path),
        adapter_registry=registry(adapter),
        run=request.run,
        effect_authority=authority,
        tool_executor=executor,
        lease_duration=timedelta(seconds=1),
    )

    with pytest.raises(RuntimeError, match="two terminal calls"):
        harness.run_step(request)
    clock.current += timedelta(seconds=2)

    with pytest.raises(ValueError, match=r"skipped.*expected ordinal 1"):
        harness.run_step(request)

    assert [call.call_id for call in executor.calls] == ["first", "second"]
    assert harness.resolve_step_result(request.run_id, 0) is None


def test_recovery_rejects_divergence_before_an_unresolved_prefix_call(
    tmp_path,
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    authority = EffectAuthority.memory(clock=clock)
    executor = CrashFirstToolExecutor(authority)
    adapter = DivergentBeforeUnresolvedAdapter()
    request = tool_request()
    harness = make_harness(
        store=make_store(tmp_path),
        adapter_registry=registry(adapter),
        run=request.run,
        effect_authority=authority,
        tool_executor=executor,
        lease_duration=timedelta(seconds=1),
    )

    with pytest.raises(RuntimeError, match="tool crashed"):
        harness.run_step(request)
    clock.current += timedelta(seconds=2)

    with pytest.raises(ValueError, match=r"expected ordinal 0.*unresolved"):
        harness.run_step(request)

    assert [call.call_id for call in executor.calls] == ["unresolved"]


def test_terminal_store_entry_repairs_ledger_bind_crash_without_execution(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    authority = EffectAuthority.memory(clock=clock)
    store = make_store(tmp_path)
    executor = RecordingToolExecutor(authority)
    request = tool_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(ToolUsingAdapter()),
        run=request.run,
        effect_authority=authority,
        tool_executor=executor,
        lease_duration=timedelta(seconds=1),
    )
    real_bind = store.bind
    crash_enabled = True

    def crash_before_terminal_binding(key, reference):
        nonlocal crash_enabled
        if crash_enabled and key.startswith(
            ISSUED_TOOL_CALL_TERMINAL_KEY_PREFIX
        ):
            crash_enabled = False
            raise RuntimeError("crash before ledger terminal binding")
        return real_bind(key, reference)

    monkeypatch.setattr(store, "bind", crash_before_terminal_binding)
    with pytest.raises(RuntimeError, match="ledger terminal binding"):
        harness.run_step(request)
    assert [call.call_id for call in executor.calls] == ["c1"]

    monkeypatch.setattr(store, "bind", real_bind)
    clock.current += timedelta(seconds=2)
    result, _ = harness.run_step(request)

    assert [call.call_id for call in executor.calls] == ["c1"]
    assert len(result.tool_evidence) == 1


def test_slot_recovers_crash_before_call_id_claim_binding(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    authority = EffectAuthority.memory(clock=clock)
    store = make_store(tmp_path)
    real_bind = store.bind
    crash_enabled = True

    def crash_before_claim(key, reference):
        nonlocal crash_enabled
        if crash_enabled and key.startswith(ISSUED_TOOL_CALL_KEY_PREFIX):
            crash_enabled = False
            raise RuntimeError("crash before claim binding")
        return real_bind(key, reference)

    monkeypatch.setattr(store, "bind", crash_before_claim)
    executor = RecordingToolExecutor(authority)
    adapter = ToolUsingAdapter(call_ids=("recoverable-window",))
    request = tool_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=request.run,
        effect_authority=authority,
        tool_executor=executor,
        lease_duration=timedelta(seconds=1),
    )

    with pytest.raises(RuntimeError, match="crash before claim binding"):
        harness.run_step(request)

    exact_request = step_request_reference(request)
    assert (
        store.resolve(_issued_tool_call_slot_binding_key(exact_request, 0))
        is not None
    )
    assert (
        store.resolve(
            _issued_tool_call_binding_key(exact_request, "recoverable-window")
        )
        is None
    )
    assert executor.calls == []

    monkeypatch.setattr(store, "bind", real_bind)
    clock.current += timedelta(seconds=2)
    result, _ = harness.run_step(request)

    assert [call.call_id for call in executor.calls] == ["recoverable-window"]
    assert len(result.tool_evidence) == 1
    assert (
        store.resolve(
            _issued_tool_call_binding_key(exact_request, "recoverable-window")
        )
        is not None
    )


def test_noncontiguous_hidden_slot_fails_before_tool_effect(tmp_path) -> None:
    store = make_store(tmp_path)
    authority = EffectAuthority.memory()
    executor = RecordingToolExecutor(authority)
    request = tool_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(ToolUsingAdapter()),
        run=request.run,
        effect_authority=authority,
        tool_executor=executor,
    )
    exact_request = step_request_reference(request)
    config = request.tool_configs[0]
    call = ToolCall(
        call_id="hidden-after-gap",
        tool_config=config,
        capacity_binding=tool_capacity_binding(
            ToolCapacityScope.RUN,
            request.run.record_ref,
        ),
        args={"model_route": "r0", "template": "hidden"},
    )
    claim_record = _IssuedToolCallClaim(
        request=exact_request,
        call=tool_call_reference(call),
    )
    claim_ref = _IssuedToolCallClaimRef(
        record=claim_record,
        record_ref=typed_ref_for_record(
            ISSUED_TOOL_CALL_CLAIM_SCHEMA,
            claim_record.model_dump(mode="json"),
        ),
    )
    store.put(
        ISSUED_TOOL_CALL_CLAIM_SCHEMA,
        claim_record.model_dump(mode="json"),
    )
    slot = _IssuedToolCallSlot(
        request=exact_request,
        ordinal=1,
        claim=claim_ref,
    )
    slot_ref, _ = store.put(
        ISSUED_TOOL_CALL_SLOT_SCHEMA,
        slot.model_dump(mode="json"),
    )
    store.bind(
        _issued_tool_call_slot_binding_key(exact_request, 1),
        slot_ref,
    )

    with pytest.raises(ValueError, match="contiguous from ordinal zero"):
        harness.run_step(request)

    assert executor.calls == []


def test_budget_limit_slot_is_detected_by_the_overflow_sentinel(
    tmp_path,
) -> None:
    store = make_store(tmp_path)
    authority = EffectAuthority.memory()
    executor = RecordingToolExecutor(authority)
    request = tool_request(budget=BudgetState(remaining={"tool_calls": 1}))
    harness = make_harness(
        store=store,
        adapter_registry=registry(ToolUsingAdapter()),
        run=request.run,
        effect_authority=authority,
        tool_executor=executor,
    )
    exact_request = step_request_reference(request)
    config = request.tool_configs[0]
    call = ToolCall(
        call_id="overflow",
        tool_config=config,
        capacity_binding=tool_capacity_binding(
            ToolCapacityScope.RUN,
            request.run.record_ref,
        ),
        args={"model_route": "r0", "template": "overflow"},
    )
    claim_record = _IssuedToolCallClaim(
        request=exact_request,
        call=tool_call_reference(call),
    )
    claim_ref = _IssuedToolCallClaimRef(
        record=claim_record,
        record_ref=typed_ref_for_record(
            ISSUED_TOOL_CALL_CLAIM_SCHEMA,
            claim_record.model_dump(mode="json"),
        ),
    )
    store.put(
        ISSUED_TOOL_CALL_CLAIM_SCHEMA,
        claim_record.model_dump(mode="json"),
    )
    slot = _IssuedToolCallSlot(
        request=exact_request,
        ordinal=1,
        claim=claim_ref,
    )
    slot_ref, _ = store.put(
        ISSUED_TOOL_CALL_SLOT_SCHEMA,
        slot.model_dump(mode="json"),
    )
    store.bind(
        _issued_tool_call_slot_binding_key(exact_request, 1),
        slot_ref,
    )

    with pytest.raises(ValueError, match="outside the bounded budget"):
        harness.run_step(request)

    assert executor.calls == []


def test_recovered_divergent_same_id_conflicts_before_effect(tmp_path) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    authority = EffectAuthority.memory(clock=clock)
    store = make_store(tmp_path)
    first_config = make_tool_definition_config(namespace="first")
    second_config = make_tool_definition_config(namespace="second")
    run = tool_run(config=first_config).record.model_copy(
        update={
            "tool_configs": (
                tool_config_reference(first_config),
                tool_config_reference(second_config),
            )
        }
    )
    request = tool_request(run=run)
    executor = RecordingToolExecutor(authority)
    adapter = CrashAfterTerminalAdapter(divergent_on_recovery=True)
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=request.run,
        effect_authority=authority,
        tool_executor=executor,
        lease_duration=timedelta(seconds=1),
    )
    with pytest.raises(RuntimeError, match="crashed"):
        harness.run_step(request)
    clock.current += timedelta(seconds=2)

    with pytest.raises(IssuedToolCallConflictError):
        harness.run_step(request)

    assert [call.tool_config for call in executor.calls] == (
        [tool_config_reference(first_config)]
    )


def test_refusal_is_counted_once_and_published_as_evidence(tmp_path) -> None:
    store = make_store(tmp_path)
    executor = RefusingToolExecutor()
    request = tool_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(ToolUsingAdapter()),
        run=request.run,
        tool_executor=executor,
    )

    result, _ = harness.run_step(request)

    assert len(executor.calls) == 1
    assert result.tool_evidence[0].result.record.refusal is not None
    assert result.budget_delta.consumed == {"tool_calls": 1}


def test_failed_step_keeps_terminal_evidence_and_spend(tmp_path) -> None:
    store = make_store(tmp_path)
    authority = EffectAuthority.memory()
    executor = RecordingToolExecutor(authority)
    request = tool_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(FailedAfterTerminalAdapter()),
        run=request.run,
        effect_authority=authority,
        tool_executor=executor,
    )

    result, _ = harness.run_step(request)

    assert result.status is StepStatus.FAILED
    assert len(result.tool_evidence) == 1
    assert result.budget_delta.consumed == {"tool_calls": 1}


def test_same_step_local_id_may_be_reused_by_the_next_exact_request(
    tmp_path,
) -> None:
    store = make_store(tmp_path)
    authority = EffectAuthority.memory()
    executor = RecordingToolExecutor(authority)
    adapter = SameCallEveryStepAdapter()
    first_request = tool_request(
        contract=output_contract(),
    )
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=first_request.run,
        effect_authority=authority,
        tool_executor=executor,
    )
    first, first_ref = harness.run_step(first_request)
    second_request = OptimizationStepRequest(
        run=first_request.run,
        step_id="run-tool-s1",
        kind=StepKind.TOOL,
        step_index=1,
        prior_step_result_ref=first_ref,
        prior_state_ref=first.state_ref,
        prior_history_ref=first.history_ref,
        candidates=(first.accepted_candidates[0].record,),
        budget=first.budget,
        step_output_contract=first_request.step_output_contract,
    )

    second, _ = harness.run_step(second_request)

    assert second.status is StepStatus.COMPLETE
    assert len(first.tool_evidence) == len(second.tool_evidence) == 1
    assert first.request.record_ref != second.request.record_ref
    assert second.budget_delta.consumed == {"tool_calls": 1}
    assert [call.call_id for call in executor.calls] == ["same-step-local-id"]


def test_same_id_across_configured_handles_dispatches_only_once(
    tmp_path,
) -> None:
    first_config = make_tool_definition_config(namespace="first")
    second_config = make_tool_definition_config(namespace="second")
    run = tool_run(config=first_config).record.model_copy(
        update={
            "tool_configs": (
                tool_config_reference(first_config),
                tool_config_reference(second_config),
            )
        }
    )
    request = tool_request(run=run)
    authority = EffectAuthority.memory()
    executor = RecordingToolExecutor(authority)
    harness = make_harness(
        store=make_store(tmp_path),
        adapter_registry=registry(SameIdAcrossConfigsAdapter()),
        run=request.run,
        effect_authority=authority,
        tool_executor=executor,
    )

    with pytest.raises(ValueError, match="unique within a Step attempt"):
        harness.run_step(request)

    assert len(executor.calls) == 1


def test_one_call_budget_never_dispatches_the_second_call(tmp_path) -> None:
    request = tool_request(budget=BudgetState(remaining={"tool_calls": 1}))
    authority = EffectAuthority.memory()
    executor = RecordingToolExecutor(authority)
    harness = make_harness(
        store=make_store(tmp_path),
        adapter_registry=registry(ToolUsingAdapter(call_ids=("one", "two"))),
        run=request.run,
        effect_authority=authority,
        tool_executor=executor,
    )

    with pytest.raises(ValueError, match="only 1"):
        harness.run_step(request)

    assert [call.call_id for call in executor.calls] == ["one"]
    assert harness.resolve_step_result(request.run_id, 0) is None


def test_concurrent_duplicate_fails_before_second_executor_call(
    tmp_path,
) -> None:
    entered = Event()
    release = Event()
    authority = EffectAuthority.memory()
    executor = BlockingToolExecutor(
        authority,
        entered=entered,
        release=release,
    )
    request = tool_request()
    harness = make_harness(
        store=make_store(tmp_path),
        adapter_registry=registry(
            ConcurrentDuplicateAdapter(entered=entered, release=release)
        ),
        run=request.run,
        effect_authority=authority,
        tool_executor=executor,
    )

    with pytest.raises(ValueError, match="unique within a Step attempt"):
        harness.run_step(request)

    assert len(executor.calls) == 1


def test_issued_ledger_keys_and_exact_record_schemas_are_golden(
    tmp_path,
) -> None:
    store = make_store(tmp_path)
    authority = EffectAuthority.memory()
    request = tool_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(ToolUsingAdapter(call_ids=("golden",))),
        run=request.run,
        effect_authority=authority,
        tool_executor=RecordingToolExecutor(authority),
    )
    result, _ = harness.run_step(request)
    exact_request = step_request_reference(request)

    claim_key = _issued_tool_call_binding_key(exact_request, "golden")
    assert claim_key == (
        f"{ISSUED_TOOL_CALL_KEY_PREFIX}"
        "5bfb828ec4458df52305421053f03b835b0f858c404f03f39519a82f2c85c1a2"
    )
    claim_object_ref = store.resolve(claim_key)
    assert claim_object_ref is not None
    claim_ref = TypedRef(
        schema_name=claim_object_ref.schema,
        content_hash=claim_object_ref.content_hash,
    )
    assert claim_ref.schema_name == ISSUED_TOOL_CALL_CLAIM_SCHEMA
    claim = _IssuedToolCallClaimRef(
        record=_IssuedToolCallClaim.model_validate(
            store.get(claim_ref.reference)
        ),
        record_ref=claim_ref,
    )
    assert claim.record.model_dump(mode="json") == {
        "request": exact_request.model_dump(mode="json"),
        "call": result.tool_evidence[0].result.record.call.model_dump(
            mode="json"
        ),
    }

    slot_key = _issued_tool_call_slot_binding_key(exact_request, 0)
    assert slot_key == (
        f"{ISSUED_TOOL_CALL_SLOT_KEY_PREFIX}"
        "e1e1f20518f375cd8171b1956d6fb30f7b98df9165d7f484a250361a85901f73"
    )
    slot_ref = store.resolve(slot_key)
    assert slot_ref is not None
    assert slot_ref.schema == ISSUED_TOOL_CALL_SLOT_SCHEMA
    assert store.get(slot_ref) == {
        "request": exact_request.model_dump(mode="json"),
        "ordinal": 0,
        "claim": claim.model_dump(mode="json"),
    }

    terminal_key = _issued_tool_call_terminal_binding_key(claim)
    assert terminal_key == (
        f"{ISSUED_TOOL_CALL_TERMINAL_KEY_PREFIX}"
        "06345594fdd1ea598752c3b70d5487299468f06f6d37871fa66ccde056be8702"
    )
    terminal_ref = store.resolve(terminal_key)
    assert terminal_ref is not None
    assert terminal_ref.schema == ISSUED_TOOL_CALL_TERMINAL_SCHEMA
    assert store.get(terminal_ref) == {
        "claim": claim.model_dump(mode="json"),
        "result": result.tool_evidence[0].result.model_dump(mode="json"),
    }
    assert result.tool_evidence[0].result.record_ref == (
        result.tool_evidence[0].store_entry.tool_result_ref
    )


@pytest.mark.parametrize(
    "snapshot_schema",
    [STATE_SNAPSHOT_SCHEMA, HISTORY_SNAPSHOT_SCHEMA],
)
def test_snapshot_ref_is_exact_verified_before_result_binding(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    snapshot_schema: str,
) -> None:
    store = make_store(tmp_path)
    request = proposal_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(SnapshotAdapter()),
        run=request.run,
    )
    real_put = harness._put

    def wrong_snapshot_ref(
        schema: str,
        content: dict[str, object],
    ) -> TypedRef:
        persisted = real_put(schema, content)
        if schema == snapshot_schema:
            return TypedRef(
                schema_name=schema,
                content_hash="f" * 64,
            )
        return persisted

    monkeypatch.setattr(harness, "_put", wrong_snapshot_ref)

    with pytest.raises(ValueError, match=r"snapshot.*failed validation"):
        harness.run_step(request)
    assert harness.resolve_step_result(request.run_id, 0) is None
