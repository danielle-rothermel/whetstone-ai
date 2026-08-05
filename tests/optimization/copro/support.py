"""Explicit adapter and request construction shared by COPRO tests."""

from __future__ import annotations

from typing import Any

from tests.optimization.support import (
    FULL_A,
    FULL_C,
    evaluation_binding,
    internal_reward_policy,
)
from whetstone.core.effects.models import ReplayPolicy
from whetstone.core.identity import IdentityRef, typed_ref_for_record
from whetstone.experiment.candidate import (
    Candidate,
    TemplateRenderContract,
    TemplateRenderKind,
)
from whetstone.optimization.contracts import (
    STEP_RESULT_SCHEMA,
    BudgetState,
    OptimizationRun,
    OptimizationRunRef,
    OptimizationStepRequest,
    OutputContract,
    StepKind,
    StepMode,
    optimization_run_reference,
)
from whetstone.optimization.copro.adapter import CoproAdapter
from whetstone.optimization.copro.control import (
    CoproInjectedDefaults,
    configure_copro,
)
from whetstone.optimization.proposal.proposer import (
    DurableProposalExecutor,
    FakeProposerTransport,
    ProposalExecutorDurabilityContract,
    ProposerConfig,
    _durable_proposal_executor,
)
from whetstone.provider.language_model import PlainPromptAdapter


def durable_copro_proposal_executor(
    *,
    policy_identity_hash: str = FULL_C,
) -> DurableProposalExecutor:
    """Mint the canonical capability over an in-process pass-through."""

    def execute(*, config, request, transport, count):
        return transport.draft(config, request, count)

    return _durable_proposal_executor(
        durability_contract=ProposalExecutorDurabilityContract(
            recovery_policy=ReplayPolicy.DURABLE_WORKFLOW,
            policy_identity_hash=policy_identity_hash,
        ),
        execute=execute,
    )


def copro_prompt_model(*, temperature: float = 1.4) -> ProposerConfig:
    return ProposerConfig(
        provider_call_config=IdentityRef(
            record_ref=typed_ref_for_record(
                "dr_providers.provider_call_config",
                {"route": "copro-proposer"},
            ),
            identity_hash=FULL_A,
        ),
        temperature=temperature,
    )


def configure_test_copro(
    *,
    breadth: int = 3,
    depth: int = 1,
    track_stats: bool = False,
):
    policy = internal_reward_policy()
    return configure_copro(
        breadth=breadth,
        depth=depth,
        track_stats=track_stats,
        defaults=CoproInjectedDefaults(
            prompt_model=copro_prompt_model(),
            evaluation_binding=evaluation_binding(),
            expected_reward_policy_hash=policy.identity_hash(),
            provider_execution_policy_hash=FULL_A,
            prompt_adapter=PlainPromptAdapter(),
        ),
    )


def make_test_copro_adapter(
    script: dict[tuple[str, int], tuple[str, ...]],
    *,
    control: Any | None = None,
):
    exact_control = control or configure_test_copro()
    transport = FakeProposerTransport(
        script,
        execution_policy_hash=FULL_A,
        prompt_adapter_identity_hash=exact_control.prompt_adapter_identity_hash,
    )
    return (
        CoproAdapter(
            control=exact_control,
            transport=transport,
            proposal_executor=durable_copro_proposal_executor(),
        ),
        transport,
        exact_control,
    )


def copro_candidate(
    candidate_id: str,
    template: str,
    *,
    parent: str = "root",
) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        base_ref=typed_ref_for_record(
            "test.copro_candidate_parent",
            {"id": parent},
        ),
        payload={
            "user_prompt_template": template,
            "fixed": "unchanged",
        },
    )


def copro_run(control: Any) -> OptimizationRunRef:
    return optimization_run_reference(
        OptimizationRun(
            run_id="copro-run",
            optimizer_config=control.reference(),
            adapter_key="copro",
            mode=StepMode.PROPOSAL_ONLY,
            terminal_output_contract=OutputContract(returned_proposal_count=1),
            template_render_contract=TemplateRenderContract(
                kind=TemplateRenderKind.PYTHON_FORMAT_V1,
                available_fields=("input",),
                required_fields=("input",),
            ),
            reward_policy=internal_reward_policy(),
        )
    )


def copro_step_request(
    control: Any,
    *,
    step_index: int = 0,
    candidates: tuple[Candidate, ...] | None = None,
    history: list[dict[str, object]] | None = None,
    proposal_budget: int | None = None,
) -> OptimizationStepRequest:
    accepted_count = (
        control.breadth - 1 if step_index == 0 else control.breadth
    )
    return OptimizationStepRequest(
        run=copro_run(control),
        step_id=f"copro-{step_index}",
        kind=StepKind.PROPOSAL,
        step_index=step_index,
        prior_step_result_ref=(
            None
            if step_index == 0
            else typed_ref_for_record(
                STEP_RESULT_SCHEMA,
                {"step": step_index - 1},
            )
        ),
        candidates=candidates
        or (copro_candidate("baseline", "base {input}"),),
        pools={"attempt_history": history or []},
        hyperparameters=control.step_hyperparameters(iteration=step_index),
        budget=BudgetState(
            remaining={
                "proposal_calls": (
                    proposal_budget
                    if proposal_budget is not None
                    else control.breadth
                )
            }
        ),
        step_output_contract=OutputContract(
            returned_proposal_count=accepted_count
        ),
    )
