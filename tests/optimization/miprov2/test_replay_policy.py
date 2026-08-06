from __future__ import annotations

from typing import cast

import pytest

from tests.optimization.miprov2.support import make_minimal_miprov2_runtime
from tests.optimization.support import make_harness, make_store, registry
from whetstone.core.effects.authority import EffectAuthority
from whetstone.core.effects.models import ReplayPolicy
from whetstone.optimization.adapters import AdapterReplayPolicyMismatchError
from whetstone.optimization.contracts import (
    BudgetState,
    OptimizationRun,
    OutputContract,
    StepMode,
    optimization_run_reference,
)
from whetstone.optimization.miprov2.adapter import Miprov2Adapter
from whetstone.optimization.miprov2.eval_config import (
    Miprov2EvalConfigBinding,
    Miprov2EvalConfigBindingRequest,
    Miprov2EvalConfigResolver,
)
from whetstone.optimization.proposal.proposer import (
    DurableProposalExecutor,
    FakeProposerTransport,
    ProposalExecutorDurabilityContract,
    _durable_proposal_executor,
)


class _UnusedEvalConfigResolver:
    def resolve(
        self,
        _request: Miprov2EvalConfigBindingRequest,
    ) -> Miprov2EvalConfigBinding:
        raise AssertionError("proposal policy test must not reach evaluation")


class _ExecutionRecorder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, config, request, transport, count):
        self.calls += 1
        return transport.draft(config, request, count)


def _recording_executor(
    recorder: _ExecutionRecorder,
) -> DurableProposalExecutor:
    return _durable_proposal_executor(
        durability_contract=ProposalExecutorDurabilityContract(
            recovery_policy=ReplayPolicy.DURABLE_WORKFLOW,
            policy_identity_hash="c" * 64,
        ),
        execute=recorder,
    )


def _case(tmp_path, *, replay_policy: ReplayPolicy):
    driver, state = make_minimal_miprov2_runtime()
    store = make_store(tmp_path)
    transport = FakeProposerTransport(
        {},
        default=("Instruction: durable {query}.",),
        execution_policy_hash=(state.control.provider_execution_policy_hash),
        prompt_adapter_identity_hash=(
            state.control.prompt_adapter_identity_hash
        ),
    )
    recorder = _ExecutionRecorder()
    adapter = Miprov2Adapter(
        store=store,
        proposer_config=state.control.prompt_model,
        transport=transport,
        eval_config_resolver=cast(
            Miprov2EvalConfigResolver,
            _UnusedEvalConfigResolver(),
        ),
        proposal_executor=_recording_executor(recorder),
        driver=driver,
    )
    run = optimization_run_reference(
        OptimizationRun(
            run_id=state.run_id,
            optimizer_config=state.control.reference(),
            adapter_key=adapter.key,
            mode=StepMode.PROPOSAL_ONLY,
            terminal_output_contract=OutputContract(returned_proposal_count=1),
            template_render_contract=(state.control.template_render_contract),
            reward_policy=state.control.reward_policy,
        )
    )
    budget = BudgetState(
        remaining={
            "bootstrap_rollouts": 0,
            "proposal_calls": 2,
            "evaluations": 2,
            "task_rows": 6,
        }
    )
    request = adapter.build_step_request(
        run=run,
        step_index=0,
        initial_state=state,
        initial_budget=budget,
    )
    authority = EffectAuthority.memory()
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=run,
        effect_authority=authority,
        adapter_replay_policy=replay_policy,
    )
    return harness, request, authority, adapter, recorder, transport


@pytest.mark.parametrize(
    "configured_policy",
    (ReplayPolicy.IDEMPOTENT, ReplayPolicy.NO_REDRIVE),
)
def test_miprov2_policy_mismatch_fails_before_any_effect(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    configured_policy: ReplayPolicy,
) -> None:
    (
        harness,
        request,
        authority,
        adapter,
        recorder,
        transport,
    ) = _case(tmp_path, replay_policy=configured_policy)
    writes = 0
    acquisitions = 0
    real_put = harness._put
    real_acquire = authority.acquire

    def record_put(*args, **kwargs):
        nonlocal writes
        writes += 1
        return real_put(*args, **kwargs)

    def record_acquire(*args, **kwargs):
        nonlocal acquisitions
        acquisitions += 1
        return real_acquire(*args, **kwargs)

    monkeypatch.setattr(harness, "_put", record_put)
    monkeypatch.setattr(authority, "acquire", record_acquire)

    with pytest.raises(AdapterReplayPolicyMismatchError) as caught:
        harness.run_step(request)

    assert caught.value.adapter_key == adapter.key
    assert caught.value.configured_policy is configured_policy
    assert caught.value.required_policy is ReplayPolicy.DURABLE_WORKFLOW
    assert writes == 0
    assert acquisitions == 0
    assert recorder.calls == 0
    assert transport.calls == []


def test_durable_workflow_reaches_executor_and_completed_replay_is_effect_free(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness, request, authority, _, recorder, transport = _case(
        tmp_path,
        replay_policy=ReplayPolicy.DURABLE_WORKFLOW,
    )
    acquisitions = 0
    real_acquire = authority.acquire

    def record_acquire(*args, **kwargs):
        nonlocal acquisitions
        acquisitions += 1
        return real_acquire(*args, **kwargs)

    monkeypatch.setattr(authority, "acquire", record_acquire)

    first = harness.run_step(request)
    assert recorder.calls == 1
    assert len(transport.calls) == 1
    assert acquisitions == 1

    replayed = harness.run_step(request)

    assert replayed == first
    assert recorder.calls == 1
    assert len(transport.calls) == 1
    assert acquisitions == 1
