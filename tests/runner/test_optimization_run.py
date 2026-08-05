"""Run control and controller tests.

These pin the settled decisions the controller carries: one harness per
optimizer, the ordered step-result accumulation ``terminalize`` requires, and
a control identity that covers every run input.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from tests.optimization.support import (
    candidate,
    make_store,
    memory_tool_call_store,
    pure_run,
    registry,
)
from whetstone.core.effects.authority import (
    EffectAuthority,
    ReplayPolicy,
)
from whetstone.optimization.adapters import (
    AdapterOutput,
    IdentityOptimizerAdapter,
    MappingAdapterRegistry,
)
from whetstone.optimization.contracts import (
    BudgetState,
    OptimizationStepRequest,
    OutputContract,
    StepKind,
    StepMode,
    StepStatus,
)
from whetstone.optimization.harness import OptimizationHarness
from whetstone.runner.optimization_run import (
    RUN_CONTROL_SCHEMA,
    HarnessRunController,
    OptimizationRunControl,
    RunControlError,
    StepRequestBuilder,
)


def _control(
    *,
    run=None,
    candidates=None,
    step_ceiling: int = 512,
    adapter_replay_policy: ReplayPolicy = ReplayPolicy.IDEMPOTENT,
    **overrides,
) -> OptimizationRunControl:
    records = candidates if candidates is not None else (candidate(),)
    contract = OutputContract(returned_proposal_count=len(records))
    exact_run = run if run is not None else pure_run(contract=contract)
    fields = {
        "run": exact_run,
        "initial_candidates": records,
        "initial_budget": BudgetState(remaining={"rollouts": 10}),
        "step_kind": StepKind.IDENTITY,
        "adapter_replay_policy": adapter_replay_policy,
        "owner_id": "runner-test-owner",
        "step_output_contract": contract,
        "step_ceiling": step_ceiling,
    }
    fields.update(overrides)
    return OptimizationRunControl(**fields)


def _harness_and_store(
    tmp_path: Path,
    *,
    adapter_registry=None,
    adapter_replay_policy: ReplayPolicy = ReplayPolicy.IDEMPOTENT,
):
    store = make_store(tmp_path)
    authority = EffectAuthority.memory()
    return store, OptimizationHarness(
        store=store,
        adapter_registry=adapter_registry or registry(),
        tool_store=memory_tool_call_store(store, authority),
        effect_authority=authority,
        owner_id="runner-test-owner",
        adapter_replay_policy=adapter_replay_policy,
        lease_duration=timedelta(seconds=1),
    )


def _harness(tmp_path: Path, **kwargs) -> OptimizationHarness:
    return _harness_and_store(tmp_path, **kwargs)[1]


class _ContinuingAdapter:
    """A pure adapter that continues for a fixed number of steps."""

    def __init__(self, *, continue_for: int) -> None:
        self._continue_for = continue_for
        self.invocations = 0

    @property
    def key(self) -> str:
        return "identity"

    @property
    def mode(self) -> StepMode:
        return StepMode.PURE

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.IDEMPOTENT

    def invoke(self, request, handles) -> AdapterOutput:
        self.invocations += 1
        status = (
            StepStatus.CONTINUE
            if request.step_index < self._continue_for
            else StepStatus.COMPLETE
        )
        return AdapterOutput(
            proposed_candidates=request.candidates,
            accepted_candidates=request.candidates,
            proposed_status=status,
        )


class _NeverTerminatingAdapter(_ContinuingAdapter):
    def invoke(self, request, handles) -> AdapterOutput:
        self.invocations += 1
        return AdapterOutput(
            proposed_candidates=request.candidates,
            accepted_candidates=request.candidates,
            proposed_status=StepStatus.CONTINUE,
        )


class _WrongPolicyAdapter(_ContinuingAdapter):
    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.NO_REDRIVE


# --------------------------------------------------------------------------
# Control identity
# --------------------------------------------------------------------------


def test_the_control_schema_literal_is_pinned() -> None:
    assert RUN_CONTROL_SCHEMA == "whetstone.runner.run_control"


def test_the_control_identity_is_stable() -> None:
    assert _control().identity_hash() == _control().identity_hash()


def test_changed_initial_candidates_change_the_control_identity() -> None:
    # The initial candidate is the restart anchor: the algorithm layer treats
    # it as controller input and will not detect a substitution, so it must
    # move the control identity.
    original = _control(candidates=(candidate("A", text="one"),))
    substituted = _control(candidates=(candidate("A", text="two"),))

    assert substituted.identity_hash() != original.identity_hash()


def test_a_changed_budget_changes_the_control_identity() -> None:
    original = _control()
    richer = _control(initial_budget=BudgetState(remaining={"rollouts": 99}))

    assert richer.identity_hash() != original.identity_hash()


def test_duplicate_initial_candidates_are_refused() -> None:
    with pytest.raises(ValueError, match="identity-unique"):
        _control(candidates=(candidate("A"), candidate("A")))


def test_an_empty_owner_id_is_refused() -> None:
    with pytest.raises(ValueError, match="owner_id must be non-empty"):
        _control(owner_id="")


def test_a_non_positive_step_ceiling_is_refused() -> None:
    with pytest.raises(ValueError, match="step_ceiling must be positive"):
        _control(step_ceiling=0)


def test_the_run_request_carries_the_control_identity() -> None:
    control = _control()

    request = control.run_request(controller_identity_hash="a" * 64)

    assert request.control_identity_hash == control.identity_hash()
    assert request.run_id == control.run_id


# --------------------------------------------------------------------------
# One harness per optimizer
# --------------------------------------------------------------------------


def test_an_adapter_with_a_different_replay_policy_is_refused(
    tmp_path: Path,
) -> None:
    # The harness requires exact policy equality, so a single harness can
    # never host Codex (no_redrive) alongside identity (idempotent). A later
    # refactor that collapses the harnesses must fail here.
    adapter = _WrongPolicyAdapter(continue_for=0)
    adapter_registry = MappingAdapterRegistry({"identity": adapter})
    control = _control(adapter_replay_policy=ReplayPolicy.IDEMPOTENT)
    store, harness = _harness_and_store(
        tmp_path, adapter_registry=adapter_registry
    )

    with pytest.raises(RunControlError, match="needs its own controller"):
        HarnessRunController(
            control=control,
            harness=harness,
            adapter_registry=adapter_registry,
            store=store,
        )


def test_the_harness_itself_refuses_a_mismatched_policy(
    tmp_path: Path,
) -> None:
    # The same constraint, proven at the layer that owns it: even if a
    # controller were bypassed, the harness refuses the step.
    from tests.optimization.support import pure_request
    from whetstone.optimization.adapters import (
        AdapterReplayPolicyMismatchError,
    )

    adapter = _WrongPolicyAdapter(continue_for=0)
    adapter_registry = MappingAdapterRegistry({"identity": adapter})
    harness = _harness(
        tmp_path,
        adapter_registry=adapter_registry,
        adapter_replay_policy=ReplayPolicy.IDEMPOTENT,
    )
    request = pure_request()
    harness.bind_run(request.run)

    with pytest.raises(AdapterReplayPolicyMismatchError):
        harness.run_step(request)


def test_a_mismatched_adapter_mode_is_refused(tmp_path: Path) -> None:
    control = _control()

    class _ToolModeAdapter(_ContinuingAdapter):
        @property
        def mode(self) -> StepMode:
            return StepMode.TOOL_USING

    adapter_registry = MappingAdapterRegistry(
        {"identity": _ToolModeAdapter(continue_for=0)}
    )
    store, harness = _harness_and_store(
        tmp_path, adapter_registry=adapter_registry
    )

    with pytest.raises(RunControlError, match="adapter mode does not match"):
        HarnessRunController(
            control=control,
            harness=harness,
            adapter_registry=adapter_registry,
            store=store,
        )


# --------------------------------------------------------------------------
# Driving a run
# --------------------------------------------------------------------------


def _controller(
    tmp_path: Path, adapter, *, step_ceiling: int = 512
) -> HarnessRunController:
    adapter_registry = MappingAdapterRegistry({adapter.key: adapter})
    control = _control(step_ceiling=step_ceiling)
    store, harness = _harness_and_store(
        tmp_path, adapter_registry=adapter_registry
    )
    return HarnessRunController(
        control=control,
        harness=harness,
        adapter_registry=adapter_registry,
        store=store,
    )


def test_a_single_step_run_terminalizes(tmp_path: Path) -> None:
    controller = _controller(tmp_path, IdentityOptimizerAdapter())

    reference = controller.drive(
        controller.control.run_request(
            controller_identity_hash=controller.runtime_identity_hash
        )
    )

    result = controller.resolve_result(reference)
    assert len(result.step_results) == 1
    assert len(result.proposals) == 1


def test_a_multi_step_run_accumulates_every_step_result(
    tmp_path: Path,
) -> None:
    adapter = _ContinuingAdapter(continue_for=3)
    controller = _controller(tmp_path, adapter)

    reference = controller.drive(
        controller.control.run_request(
            controller_identity_hash=controller.runtime_identity_hash
        )
    )

    # terminalize requires the complete ordered contiguous sequence, so a
    # controller that dropped one would fail rather than silently truncate.
    result = controller.resolve_result(reference)
    assert len(result.step_results) == 4
    assert adapter.invocations == 4


def test_driving_the_same_run_twice_replays_rather_than_repaying(
    tmp_path: Path,
) -> None:
    # This is what makes the controller safe to re-enter after a recovery.
    adapter = _ContinuingAdapter(continue_for=2)
    controller = _controller(tmp_path, adapter)
    request = controller.control.run_request(
        controller_identity_hash=controller.runtime_identity_hash
    )

    first = controller.drive(request)
    invocations_after_first = adapter.invocations
    replay = controller.drive(request)

    assert replay == first
    # The adapter is never invoked a second time for an already-bound step.
    assert adapter.invocations == invocations_after_first


def test_a_run_that_never_terminates_fails_loudly(tmp_path: Path) -> None:
    adapter = _NeverTerminatingAdapter(continue_for=0)
    controller = _controller(tmp_path, adapter, step_ceiling=3)

    with pytest.raises(RunControlError, match="did not reach a terminal"):
        controller.drive(
            controller.control.run_request(
                controller_identity_hash=controller.runtime_identity_hash
            )
        )

    assert adapter.invocations == 3


def test_driving_a_request_for_another_run_is_refused(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path, IdentityOptimizerAdapter())
    other = _control(run=pure_run(run_id="run-other"))

    with pytest.raises(RunControlError, match="different run"):
        controller.drive(
            other.run_request(
                controller_identity_hash=controller.runtime_identity_hash
            )
        )


def test_driving_a_drifted_control_identity_is_refused(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path, IdentityOptimizerAdapter())
    request = controller.control.run_request(
        controller_identity_hash=controller.runtime_identity_hash
    )
    drifted = request.model_copy(update={"control_identity_hash": "c" * 64})

    with pytest.raises(RunControlError, match="control identity does not"):
        controller.drive(drifted)


def test_the_controller_identity_is_the_control_identity(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path, IdentityOptimizerAdapter())

    assert controller.runtime_identity_hash == (
        controller.control.identity_hash()
    )


# --------------------------------------------------------------------------
# StepRequestBuilder carry-forward
# --------------------------------------------------------------------------


def test_the_initial_request_carries_no_prior_references() -> None:
    control = _control()

    request = StepRequestBuilder(control).build(
        step_index=0, prior_result=None, prior_result_ref=None
    )

    assert request.step_index == 0
    assert request.prior_step_result_ref is None
    assert request.prior_state_ref is None
    assert request.prior_history_ref is None
    assert request.budget == control.initial_budget


def test_a_continuation_requires_both_the_result_and_its_ref() -> None:
    control = _control()

    with pytest.raises(RunControlError, match="both the prior result"):
        StepRequestBuilder(control).build(
            step_index=1, prior_result=None, prior_result_ref=object()
        )


def test_the_initial_step_refuses_a_prior_result(tmp_path: Path) -> None:
    control = _control()
    adapter_registry = MappingAdapterRegistry(
        {"identity": IdentityOptimizerAdapter()}
    )
    harness = _harness(tmp_path, adapter_registry=adapter_registry)
    harness.bind_run(control.run)
    builder = StepRequestBuilder(control)
    first = builder.build(
        step_index=0, prior_result=None, prior_result_ref=None
    )
    result, reference = harness.run_step(first)

    with pytest.raises(RunControlError, match="carries no prior result"):
        builder.build(
            step_index=0, prior_result=result, prior_result_ref=reference
        )


def test_a_continuation_threads_the_debited_budget_and_refs(
    tmp_path: Path,
) -> None:
    control = _control()
    adapter = _ContinuingAdapter(continue_for=5)
    adapter_registry = MappingAdapterRegistry({"identity": adapter})
    harness = _harness(tmp_path, adapter_registry=adapter_registry)
    harness.bind_run(control.run)
    builder = StepRequestBuilder(control)
    first = builder.build(
        step_index=0, prior_result=None, prior_result_ref=None
    )
    result, reference = harness.run_step(first)

    second = builder.build(
        step_index=1, prior_result=result, prior_result_ref=reference
    )

    # The harness enforces exact carry-forward, including the absence of a
    # state or history reference, so this request must satisfy it verbatim.
    assert second.prior_step_result_ref == reference
    assert second.prior_state_ref == result.state_ref
    assert second.prior_history_ref == result.history_ref
    assert second.budget == result.budget
    harness.run_step(second)


def test_a_built_continuation_satisfies_the_harness_carry_forward(
    tmp_path: Path,
) -> None:
    # Proven by driving a real multi-step run: any carry-forward mistake
    # raises inside run_step rather than passing silently.
    adapter = _ContinuingAdapter(continue_for=2)
    controller = _controller(tmp_path, adapter)

    reference = controller.drive(
        controller.control.run_request(
            controller_identity_hash=controller.runtime_identity_hash
        )
    )

    assert len(controller.resolve_result(reference).step_results) == 3


def test_a_step_request_is_rejected_when_it_skips_an_index(
    tmp_path: Path,
) -> None:
    control = _control()
    adapter_registry = MappingAdapterRegistry(
        {"identity": IdentityOptimizerAdapter()}
    )
    harness = _harness(tmp_path, adapter_registry=adapter_registry)
    harness.bind_run(control.run)
    builder = StepRequestBuilder(control)
    skipped = OptimizationStepRequest.model_validate(
        builder.build(
            step_index=0, prior_result=None, prior_result_ref=None
        ).model_dump(mode="json")
    ).model_copy(update={"step_index": 2})

    with pytest.raises(ValueError):
        harness.run_step(skipped)
