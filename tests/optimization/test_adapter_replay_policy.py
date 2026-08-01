"""Adapter replay requirements are mandatory and fail closed."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from pydantic import BaseModel, ConfigDict

import whetstone.optimization as optimization
from whetstone.optimization import (
    AdapterOutput,
    AdapterReplayPolicyMismatchError,
    EffectRecoveryRequiredError,
    EffectRequestConflictError,
    IdentityOptimizerAdapter,
    ReplayPolicy,
    RuntimeToolHandle,
    StepMode,
    StepStatus,
    TerminalFailure,
    step_request_reference,
)
from whetstone.optimization.effect_authority import EffectAuthority

from .support import (
    CountingProposalAdapter,
    RecordingEvaluationService,
    RecordingToolExecutor,
    ToolUsingAdapter,
    make_harness,
    make_store,
    proposal_request,
    registry,
    tool_request,
)


class NoRedriveToolAdapter(ToolUsingAdapter):
    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.NO_REDRIVE


class MutablePolicyAdapter:
    def __init__(self, policies: tuple[ReplayPolicy, ...]) -> None:
        self._policies = policies
        self.policy_reads = 0
        self.invocations = 0

    @property
    def key(self) -> str:
        return "proposal-test"

    @property
    def mode(self) -> StepMode:
        return StepMode.PROPOSAL_ONLY

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        index = min(self.policy_reads, len(self._policies) - 1)
        policy = self._policies[index]
        self.policy_reads += 1
        return policy

    def invoke(
        self,
        request: Any,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        del request, handles
        self.invocations += 1
        return AdapterOutput(
            proposed_status=StepStatus.FAILED,
            terminal_failure=TerminalFailure(
                code="test-adapter-failed",
                message="test adapter stopped",
            ),
        )


class ForgedPolicyModelAdapter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    required_replay_policy: ReplayPolicy

    @property
    def key(self) -> str:
        return "proposal-test"

    @property
    def mode(self) -> StepMode:
        return StepMode.PROPOSAL_ONLY

    def invoke(
        self,
        request: Any,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        del request, handles
        raise AssertionError("invalid policy must block invocation")


class MissingPolicyAdapter:
    @property
    def key(self) -> str:
        return "proposal-test"

    @property
    def mode(self) -> StepMode:
        return StepMode.PROPOSAL_ONLY

    def invoke(
        self,
        request: Any,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        del request, handles
        raise AssertionError("missing policy must block invocation")


class ForgedKey:
    def __eq__(self, other: object) -> bool:
        del other
        return True

    def __ne__(self, other: object) -> bool:
        del other
        return False


class ForgedKeyAdapter(CountingProposalAdapter):
    @property
    def key(self) -> Any:
        return ForgedKey()


def test_identity_adapter_requires_idempotent_replay() -> None:
    assert (
        IdentityOptimizerAdapter().required_replay_policy
        is ReplayPolicy.IDEMPOTENT
    )


def test_missing_policy_property_has_no_fallback(tmp_path) -> None:
    adapter = MissingPolicyAdapter()
    request = proposal_request()
    harness = make_harness(
        store=make_store(tmp_path),
        adapter_registry=registry(cast(Any, adapter)),
        run=request.run,
    )

    with pytest.raises(AttributeError, match="required_replay_policy"):
        harness.resolve_adapter(request.adapter_key)


def test_mismatch_fails_before_writes_handles_effects_or_invocation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = make_store(tmp_path)
    authority = EffectAuthority.memory()
    adapter = NoRedriveToolAdapter()
    executor = RecordingToolExecutor(authority)
    request = tool_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=request.run,
        effect_authority=authority,
        tool_executor=executor,
    )
    puts = 0
    acquisitions = 0
    real_put = harness._put
    real_acquire = authority.acquire

    def record_put(*args, **kwargs):
        nonlocal puts
        puts += 1
        return real_put(*args, **kwargs)

    def record_acquire(*args, **kwargs):
        nonlocal acquisitions
        acquisitions += 1
        return real_acquire(*args, **kwargs)

    monkeypatch.setattr(harness, "_put", record_put)
    monkeypatch.setattr(authority, "acquire", record_acquire)

    with pytest.raises(AdapterReplayPolicyMismatchError) as caught:
        harness.run_step(request)

    error = caught.value
    assert error.adapter_key == "tool-test"
    assert error.configured_policy is ReplayPolicy.IDEMPOTENT
    assert error.required_policy is ReplayPolicy.NO_REDRIVE
    assert str(error) == (
        "adapter 'tool-test' requires replay policy 'no_redrive'; "
        "configured policy is 'idempotent'"
    )
    assert puts == 0
    assert executor.handles_built == 0
    assert acquisitions == 0
    assert adapter.invocations == 0


def test_matching_policy_reaches_adapter_invocation(tmp_path) -> None:
    store = make_store(tmp_path)
    authority = EffectAuthority.memory()
    adapter = NoRedriveToolAdapter()
    executor = RecordingToolExecutor(authority)
    request = tool_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=request.run,
        effect_authority=authority,
        tool_executor=executor,
        adapter_replay_policy=ReplayPolicy.NO_REDRIVE,
    )

    harness.run_step(request)

    assert adapter.invocations == 1
    assert executor.handles_built == 1


@pytest.mark.parametrize("construction", ["model_construct", "model_copy"])
def test_forged_string_policy_is_rejected(
    tmp_path,
    construction: str,
) -> None:
    valid = ForgedPolicyModelAdapter(
        required_replay_policy=ReplayPolicy.IDEMPOTENT
    )
    if construction == "model_construct":
        adapter = ForgedPolicyModelAdapter.model_construct(
            required_replay_policy=ReplayPolicy.IDEMPOTENT.value
        )
    else:
        adapter = valid.model_copy(
            update={"required_replay_policy": ReplayPolicy.IDEMPOTENT.value}
        )
    request = proposal_request()
    harness = make_harness(
        store=make_store(tmp_path),
        adapter_registry=registry(cast(Any, adapter)),
        run=request.run,
    )

    with pytest.raises(
        TypeError,
        match="required_replay_policy must be an actual ReplayPolicy enum",
    ):
        harness.resolve_adapter(request.adapter_key)


def test_request_model_copy_cannot_override_host_policy(tmp_path) -> None:
    adapter = NoRedriveToolAdapter()
    request = tool_request().model_copy(
        update={"step_id": "run-tool-policy-copy"}
    )
    assert "adapter_replay_policy" not in type(request).model_fields
    harness = make_harness(
        store=make_store(tmp_path),
        adapter_registry=registry(adapter),
        run=request.run,
    )

    with pytest.raises(AdapterReplayPolicyMismatchError):
        harness.run_step(request)

    assert adapter.invocations == 0


def test_required_policy_is_read_once_and_snapshotted(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = MutablePolicyAdapter(
        (ReplayPolicy.NO_REDRIVE, ReplayPolicy.IDEMPOTENT)
    )
    authority = EffectAuthority.memory()
    acquired_policies: list[ReplayPolicy] = []
    real_acquire = authority.acquire

    def record_acquire(effect_request, **kwargs):
        acquired_policies.append(effect_request.replay_policy)
        return real_acquire(effect_request, **kwargs)

    monkeypatch.setattr(authority, "acquire", record_acquire)
    request = proposal_request()
    harness = make_harness(
        store=make_store(tmp_path),
        adapter_registry=registry(cast(Any, adapter)),
        run=request.run,
        effect_authority=authority,
        adapter_replay_policy=ReplayPolicy.NO_REDRIVE,
    )

    harness.run_step(request)

    assert adapter.policy_reads == 1
    assert acquired_policies == [ReplayPolicy.NO_REDRIVE]
    assert adapter.invocations == 1


def test_public_resolution_rejects_alias_key(tmp_path) -> None:
    adapter = CountingProposalAdapter()

    class AliasRegistry:
        def resolve(self, adapter_key: str):
            assert adapter_key == "alias"
            return adapter

    request = proposal_request()
    harness = make_harness(
        store=make_store(tmp_path),
        adapter_registry=AliasRegistry(),
        run=request.run,
    )

    with pytest.raises(ValueError, match="wrong key"):
        harness.resolve_adapter("alias")


def test_forged_key_type_fails_public_and_run_before_writes(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ForgedKeyAdapter()

    class ForgedRegistry:
        def resolve(self, adapter_key: str):
            assert adapter_key == "proposal-test"
            return adapter

    request = proposal_request()
    harness = make_harness(
        store=make_store(tmp_path),
        adapter_registry=ForgedRegistry(),
        run=request.run,
    )
    puts = 0
    real_put = harness._put

    def record_put(*args, **kwargs):
        nonlocal puts
        puts += 1
        return real_put(*args, **kwargs)

    monkeypatch.setattr(harness, "_put", record_put)

    with pytest.raises(TypeError, match="key must be an actual string"):
        harness.resolve_adapter(request.adapter_key)
    with pytest.raises(TypeError, match="key must be an actual string"):
        harness.run_step(request)

    assert puts == 0
    assert adapter.invocations == 0


def test_completed_result_replay_skips_policy_resolution(tmp_path) -> None:
    store = make_store(tmp_path)
    request = proposal_request()
    first = make_harness(
        store=store,
        adapter_registry=registry(CountingProposalAdapter()),
        run=request.run,
        evaluation_service=RecordingEvaluationService(store),
    )
    result, result_ref = first.run_step(request)

    class ExplodingRegistry:
        def resolve(self, adapter_key: str):
            del adapter_key
            raise AssertionError(
                "completed replay must not resolve an adapter"
            )

    fresh = make_harness(
        store=make_store(tmp_path),
        adapter_registry=ExplodingRegistry(),
        run=request.run,
    )

    assert fresh.run_step(request) == (result, result_ref)


class MutableClock:
    def __init__(self) -> None:
        self.current = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.current


@pytest.mark.parametrize("legacy_policy", list(ReplayPolicy))
def test_legacy_adapter_effect_policy_fails_closed_without_invocation(
    tmp_path,
    legacy_policy: ReplayPolicy,
) -> None:
    clock = MutableClock()
    authority = EffectAuthority.memory(clock=clock)
    adapter = MutablePolicyAdapter((ReplayPolicy.NO_REDRIVE,))
    request = proposal_request()
    harness = make_harness(
        store=make_store(tmp_path),
        adapter_registry=registry(cast(Any, adapter)),
        run=request.run,
        effect_authority=authority,
        adapter_replay_policy=ReplayPolicy.NO_REDRIVE,
        lease_duration=timedelta(seconds=1),
    )
    exact_request = step_request_reference(request)
    legacy_effect = harness._adapter_effect_request(
        request,
        exact_request.record_ref,
        legacy_policy,
    )
    acquired = authority.acquire(
        legacy_effect,
        owner_id="legacy-owner",
        attempt_id="legacy-attempt",
        lease_duration=timedelta(seconds=1),
    )
    assert acquired.lease is not None
    clock.current += timedelta(seconds=2)

    expected_error = (
        EffectRecoveryRequiredError
        if legacy_policy is ReplayPolicy.NO_REDRIVE
        else EffectRequestConflictError
    )
    with pytest.raises(expected_error):
        harness.run_step(request)

    assert adapter.invocations == 0


def test_error_is_exported_from_public_facade() -> None:
    assert (
        optimization.AdapterReplayPolicyMismatchError
        is AdapterReplayPolicyMismatchError
    )
