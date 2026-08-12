from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from dr_store import MemoryBackend, ObjectStore

from tests.optimization.copro.support import (
    copro_step_request,
    make_test_copro_adapter,
)
from tests.optimization.miprov2.support import make_minimal_miprov2_runtime
from tests.optimization.support import (
    RecordingEvaluationService,
    make_harness,
    make_intent,
    make_store,
    output_contract,
    proposal_request,
    pure_request,
    registry,
)
from whetstone.core.effects.models import ReplayPolicy
from whetstone.experiment.candidate import candidate_reference
from whetstone.optimization.adapters import (
    AdapterOutput,
    IdentityOptimizerAdapter,
    OptimizerAdapter,
)
from whetstone.optimization.contracts import (
    BudgetState,
    OptimizationRun,
    OptimizationStepRequest,
    OutputContract,
    StepMode,
    optimization_run_reference,
)
from whetstone.optimization.copro.adapter import SEED_PROPOSAL
from whetstone.optimization.miprov2.adapter import Miprov2Adapter
from whetstone.optimization.miprov2.eval_config import (
    Miprov2EvalConfigResolver,
)
from whetstone.optimization.proposal.proposer import (
    FakeProposerTransport,
    ProposalExecutorDurabilityContract,
    _durable_proposal_executor,
)

type AdapterCase = tuple[OptimizerAdapter, OptimizationStepRequest]


class _UnusedEvalConfigResolver:
    def resolve(self, request):
        raise AssertionError(
            f"first MIPROv2 proposal must not resolve {request!r}"
        )


class _RequestCandidateEvaluationAdapter:
    @property
    def key(self) -> str:
        return "proposal-test"

    @property
    def mode(self) -> StepMode:
        return StepMode.PROPOSAL_ONLY

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.IDEMPOTENT

    def invoke(self, request, handles) -> AdapterOutput:
        assert handles == ()
        return AdapterOutput(
            evaluation_intents=(
                make_intent(
                    request.candidates[0],
                    run_id=request.run_id,
                    step_index=request.step_index,
                    reward_policy=request.run.record.reward_policy,
                ),
            )
        )


def _identity_case(_tmp_path: Path) -> AdapterCase:
    adapter = IdentityOptimizerAdapter()
    return adapter, pure_request()


def _copro_case(_tmp_path: Path) -> AdapterCase:
    adapter, _, control = make_test_copro_adapter(
        {(SEED_PROPOSAL, 0): ("new {input}", "other {input}")}
    )
    return adapter, copro_step_request(control)


def _miprov2_case(_tmp_path: Path) -> AdapterCase:
    driver, state = make_minimal_miprov2_runtime()
    store = ObjectStore(MemoryBackend())
    transport = FakeProposerTransport(
        {},
        default=("Instruction: improved {query}.",),
        execution_policy_hash=state.control.provider_execution_policy_hash,
        prompt_adapter_identity_hash=state.control.prompt_adapter_identity_hash,
    )

    def execute(*, config, request, transport, count):
        return transport.draft(config, request, count)

    adapter = Miprov2Adapter(
        store=store,
        proposer_config=state.control.prompt_model,
        transport=transport,
        eval_config_resolver=cast(
            Miprov2EvalConfigResolver,
            _UnusedEvalConfigResolver(),
        ),
        proposal_executor=_durable_proposal_executor(
            durability_contract=ProposalExecutorDurabilityContract(
                recovery_policy=ReplayPolicy.DURABLE_WORKFLOW,
                policy_identity_hash="c" * 64,
            ),
            execute=execute,
        ),
        driver=driver,
    )
    run = optimization_run_reference(
        OptimizationRun(
            run_id=state.run_id,
            optimizer_config=state.control.reference(),
            adapter_key=adapter.key,
            mode=StepMode.PROPOSAL_ONLY,
            terminal_output_contract=OutputContract(returned_proposal_count=1),
            template_render_contract=state.control.template_render_contract,
            reward_policy=state.control.reward_policy,
        )
    )
    request = adapter.build_step_request(
        run=run,
        step_index=0,
        initial_state=state,
        initial_budget=BudgetState(
            remaining={
                "bootstrap_generations": 0,
                "proposal_calls": 2,
                "evaluations": 2,
                "task_rows": 6,
            }
        ),
    )
    return adapter, request


@pytest.mark.parametrize(
    ("factory", "key", "mode", "replay_policy"),
    (
        (
            _identity_case,
            "identity",
            StepMode.PURE,
            ReplayPolicy.IDEMPOTENT,
        ),
        (
            _copro_case,
            "copro",
            StepMode.PROPOSAL_ONLY,
            ReplayPolicy.DURABLE_WORKFLOW,
        ),
        (
            _miprov2_case,
            "miprov2",
            StepMode.PROPOSAL_ONLY,
            ReplayPolicy.DURABLE_WORKFLOW,
        ),
    ),
    ids=("identity", "copro", "miprov2"),
)
def test_native_adapter_host_conformance(
    tmp_path: Path,
    factory: Callable[[Path], AdapterCase],
    key: str,
    mode: StepMode,
    replay_policy: ReplayPolicy,
) -> None:
    adapter, request = factory(tmp_path)

    assert isinstance(adapter, OptimizerAdapter)
    assert adapter.key == key
    assert adapter.key
    assert adapter.mode is mode
    assert adapter.required_replay_policy is replay_policy
    assert isinstance(adapter.invoke(request, ()), AdapterOutput)


def test_proposal_step_may_measure_exact_request_candidate_without_proposal(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path, "request-candidate-evaluation.sqlite")
    adapter = _RequestCandidateEvaluationAdapter()
    request = proposal_request(contract=output_contract(0))
    service = RecordingEvaluationService(store)
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=request.run,
        evaluation_service=service,
    )

    result, _ = harness.run_step(request)

    assert result.proposed_candidates == ()
    assert len(result.resolved_intents) == 1
    assert result.resolved_intents[0].intent.candidate == candidate_reference(
        request.candidates[0]
    )
