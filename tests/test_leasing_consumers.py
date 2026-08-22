"""Exact-failure equality through ToolCallStore.complete and harness replay."""

from __future__ import annotations

from datetime import timedelta

import pytest

from whetstone.core.identity import (
    IdentityRef,
    ImmutableJsonObject,
    TerminalFailure,
    typed_ref_for_record,
)
from whetstone.core.leasing import EffectLeaseAuthority, ReplayPolicy
from whetstone.optim.adapters import AdapterOutput, MappingAdapterRegistry
from whetstone.optim.contracts import (
    OptimRun,
    OptimStepRequest,
    OutputContract,
    StepKind,
    StepMode,
    StepStatus,
)
from whetstone.optim.harness import OptimHarness
from whetstone.optim.tools.admission import tool_effect_request
from whetstone.optim.tools.contracts import (
    ToolCall,
    ToolCapacity,
    ToolCapacityScope,
    ToolConfig,
    ToolDefinition,
    ToolResult,
    tool_call_reference,
    tool_capacity_binding,
    tool_config_reference,
    tool_definition_reference,
    tool_result_reference,
)
from whetstone.optim.tools.facade import ToolAdmissionAuthority, ToolCallStore
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    build_toy_experiment,
    toy_template_render_contract,
)

_LEASE = timedelta(seconds=30)
_ADAPTER_KEY = "lease-failing"


def _dirty_failure(message: str) -> TerminalFailure:
    return TerminalFailure(
        code="evaluation_RuntimeError",
        message=message,
        details=ImmutableJsonObject({"attempt": 1}),
    )


def _tool_call() -> tuple[ToolCall, ToolConfig]:
    experiment = build_toy_experiment(num_seeds=1)
    definition = ToolDefinition(
        tool_name="lease_probe",
        input_fields=("probe",),
        output_fields=("result",),
    )
    config = ToolConfig(
        definition=tool_definition_reference(definition),
        endpoint_key="lease-probe",
        eval_config=experiment.eval_configs.internal.eval_config,
        reward_policy_hash=experiment.reward_policy.identity_hash(),
        capacity=ToolCapacity(
            max_accepted_calls=1, scope=ToolCapacityScope.GLOBAL
        ),
        store_namespace_key="lease-probe-ns",
        candidate_template_field=TOY_MUTATION_FIELD,
    )
    call = ToolCall(
        call_id="lease-probe-call",
        tool_config=tool_config_reference(config),
        capacity_binding=tool_capacity_binding(scope=ToolCapacityScope.GLOBAL),
        args=ImmutableJsonObject({"probe": "x"}),
    )
    return call, config


@pytest.mark.parametrize(
    "message",
    [
        "provider error: " + "x" * 1200,
        "hello\x00world",
        "bad\ud800surrogate",
    ],
    ids=["oversized", "nul", "unpaired-surrogate"],
)
def test_tool_call_store_complete_accepts_a_dirty_failed_terminal(
    sqlite_store, message: str
) -> None:
    """fail() then complete() keeps terminal.failure equal to the Tool Result."""
    call, config = _tool_call()
    failure = _dirty_failure(message)
    result = ToolResult(
        call=tool_call_reference(call),
        terminal_failure=failure,
        provenance_ordinal=1,
    )
    effect_authority = EffectLeaseAuthority.memory()
    store = ToolCallStore(
        sqlite_store, ToolAdmissionAuthority.memory(), effect_authority
    )
    try:
        store.admit(call, config)
        result_ref = store.persist_result(result)
        assert result_ref == tool_result_reference(result).record_ref
        acquisition = effect_authority.acquire(
            tool_effect_request(call),
            owner_id="owner-a",
            attempt_id="attempt-a",
            lease_duration=_LEASE,
        )
        assert acquisition.lease is not None
        terminal = effect_authority.fail(
            acquisition.lease, result_ref=result_ref, failure=failure
        )
        assert terminal.failure == failure
        completed = store.complete(result, terminal=terminal)
        loaded = store.load_terminal_result(completed)
    finally:
        effect_authority.close()

    assert loaded.terminal_failure == failure
    assert completed.effect_terminal is not None
    assert completed.effect_terminal.failure == failure


class _FailingAdapter:
    def __init__(self, failure: TerminalFailure) -> None:
        self._failure = failure

    @property
    def key(self) -> str:
        return _ADAPTER_KEY

    @property
    def mode(self) -> StepMode:
        return StepMode.PROPOSAL_ONLY

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.DURABLE_WORKFLOW

    def invoke(self, request, handles) -> AdapterOutput:
        return AdapterOutput(
            proposed_status=StepStatus.FAILED,
            terminal_failure=self._failure,
        )


def test_harness_failed_replay_accepts_an_oversized_checkpoint_failure(
    sqlite_store,
) -> None:
    """A second _effectful_output takes FAILED replay without rejecting."""
    experiment = build_toy_experiment(num_seeds=1)
    failure = _dirty_failure("provider error: " + "x" * 1200)
    effect_authority = EffectLeaseAuthority.memory()
    harness = OptimHarness(
        store=sqlite_store,
        adapter_registry=MappingAdapterRegistry(
            {_ADAPTER_KEY: _FailingAdapter(failure)}
        ),
        tool_store=ToolCallStore(
            sqlite_store, ToolAdmissionAuthority.memory(), effect_authority
        ),
        effect_authority=effect_authority,
        owner_id="lease-fail-owner",
        adapter_replay_policy=ReplayPolicy.DURABLE_WORKFLOW,
        lease_duration=_LEASE,
    )
    control_ref = typed_ref_for_record(
        "whetstone.leasing_test_control", {"adapter": _ADAPTER_KEY}
    )
    run = OptimRun(
        run_id="lease-fail-run",
        optimizer_config=IdentityRef(
            record_ref=control_ref, record_hash=control_ref.content_hash
        ),
        adapter_key=_ADAPTER_KEY,
        mode=StepMode.PROPOSAL_ONLY,
        terminal_output_contract=OutputContract(returned_proposal_count=0),
        template_render_contract=toy_template_render_contract(),
        mutation_field=TOY_MUTATION_FIELD,
        reward_policy=experiment.reward_policy,
    )
    bound = harness.bind_run(run)
    request = OptimStepRequest(
        run=bound,
        step_id="lease-fail-step",
        kind=StepKind.PROPOSAL,
        step_index=0,
        candidates=(experiment.initial_candidate,),
        step_output_contract=OutputContract(returned_proposal_count=0),
    )
    request_ref = harness._put_request(request)
    harness._persist_candidate(experiment.initial_candidate)
    resolved = harness._resolve_compatible_adapter(
        request.adapter_key, expected_mode=request.mode
    )
    try:
        first = harness._effectful_output(
            request,
            request_ref,
            resolved,
            ledger=None,
            guarded_handles=(),
        )
        replayed = harness._effectful_output(
            request,
            request_ref,
            resolved,
            ledger=None,
            guarded_handles=(),
        )
    finally:
        effect_authority.close()

    assert first.terminal_failure == failure
    assert replayed == first
