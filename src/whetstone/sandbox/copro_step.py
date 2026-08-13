from __future__ import annotations

from whetstone.core.identity import (
    IdentityRef,
    ImmutableJsonObject,
    compute_identity_hash,
    typed_ref_for_record,
)
from whetstone.experiment.candidate import Candidate, candidate_reference
from whetstone.optimization.copro.adapter import CoproConfig, CoproDriver
from whetstone.optimization.copro.prompts import (
    COPRO_INSTRUCTION_CONTRACT_KEY,
    COPRO_INSTRUCTION_HISTORY_KEY,
)
from whetstone.optimization.copro.proposal_contract import (
    CoproProposalContractRecord,
)
from whetstone.optimization.proposal.proposer import (
    ProposerConfig,
    ProposalRequest,
    prompt_adapter_identity_hash,
)
from whetstone.provider.language_model import PlainPromptAdapter
from whetstone.testing.fakes.proposer import DummyProposerTransport
from whetstone.testing.toy.experiment import TOY_MUTATION_FIELD, TOY_ROOT_BASE_SCHEMA
from whetstone.sandbox.transcript import (
    SandboxCandidateMutation,
    SandboxCoproSeedTranscript,
    SandboxProposalCall,
    SandboxProposalDraft,
    SandboxRoundPlan,
)

__all__ = ["run_copro_seed_preview", "toy_copro_proposal_contract"]


def toy_copro_proposal_contract(
    *,
    task_context: str,
) -> CoproProposalContractRecord:
    return CoproProposalContractRecord(
        target_name="toy_instruction",
        task_context=task_context,
        output_rule=(
            "Return one non-empty instruction body with no markdown fences."
        ),
    )


def _baseline_candidate(*, task_prompt: str) -> Candidate:
    return Candidate(
        candidate_id="sandbox-copro-baseline",
        base_ref=typed_ref_for_record(TOY_ROOT_BASE_SCHEMA, {"kind": "root"}),
        payload={TOY_MUTATION_FIELD: task_prompt},
    )


def _toy_proposer_config() -> ProposerConfig:
    record_ref = typed_ref_for_record(
        "dr_providers.provider_call_config",
        {"placeholder": True},
    )
    return ProposerConfig(
        provider_call_config=IdentityRef(
            record_ref=record_ref,
            record_hash="b" * 64,
        )
    )


def run_copro_seed_preview(
    *,
    breadth: int = 3,
    depth: int = 1,
    task_prompt: str = "Say hello to the user.",
    scripted_bodies: tuple[str, ...] | None = None,
    task_context: str | None = None,
) -> SandboxCoproSeedTranscript:
    if breadth <= 1:
        raise ValueError("breadth must be greater than 1")
    if depth < 1:
        raise ValueError("depth must be at least 1")

    baseline = _baseline_candidate(task_prompt=task_prompt)
    config = CoproConfig(breadth=breadth, depth=depth)
    driver = CoproDriver(config)
    plan = driver.plan_round(
        iteration=0,
        initial_candidates=(baseline,),
        attempt_history=(),
    )
    contract = toy_copro_proposal_contract(
        task_context=task_context or task_prompt,
    )
    bodies = scripted_bodies or (
        "Greet the user warmly in one short sentence.",
        "Respond with a concise friendly hello.",
    )
    adapter = PlainPromptAdapter()
    transport = DummyProposerTransport(
        scripted_bodies=bodies,
        execution_policy_hash="a" * 64,
        prompt_adapter_identity_hash=prompt_adapter_identity_hash(adapter),
        proposal_mode=plan.proposal_mode,
        request_ordinal=plan.iteration,
    )
    prompt = (
        "Propose diverse instruction variants for the toy task.\n"
        f"Task context: {contract.task_context}\n"
        f"Output rule: {contract.output_rule}"
    )
    proposal_request = ProposalRequest(
        proposal_mode=plan.proposal_mode,
        request_ordinal=plan.iteration,
        proposal_authority_identity_hash=compute_identity_hash(
            schema="whetstone.sandbox.copro",
            schema_version=1,
            payload={"sandbox": "copro"},
        ),
        mutation_field=TOY_MUTATION_FIELD,
        base_candidate=candidate_reference(baseline),
        context=ImmutableJsonObject(
            {
                COPRO_INSTRUCTION_CONTRACT_KEY: contract.model_dump(mode="json"),
                COPRO_INSTRUCTION_HISTORY_KEY: [],
                "proposal_prompt": prompt,
            }
        ),
    )
    proposer_config = _toy_proposer_config()
    drafts = transport.draft(
        proposer_config,
        proposal_request,
        plan.proposal_count,
    )
    base_template = baseline.payload[TOY_MUTATION_FIELD]
    mutations: list[SandboxCandidateMutation] = []
    draft_rows: list[SandboxProposalDraft] = []
    for index, draft in enumerate(drafts):
        draft_rows.append(
            SandboxProposalDraft(
                ordinal=index,
                template=draft.template,
                failed=draft.failed,
                failure_message=(
                    None
                    if draft.terminal_failure is None
                    else draft.terminal_failure.message
                ),
            )
        )
        disposition = "accepted"
        reason: str | None = None
        candidate_id = f"sandbox:copro:{index}"
        template = draft.template.strip('"').strip()
        try:
            if draft.failed:
                raise ValueError("provider returned a failed draft")
            contract.validate_instruction(template)
            if template == base_template:
                raise ValueError("mutation must differ from the base template")
        except ValueError as error:
            disposition = "provider_failed" if draft.failed else "rejected"
            reason = str(error)
        mutations.append(
            SandboxCandidateMutation(
                ordinal=index,
                candidate_id=candidate_id,
                template=template,
                disposition=disposition,
                reason=reason,
            )
        )

    return SandboxCoproSeedTranscript(
        task_prompt=task_prompt,
        breadth=breadth,
        depth=depth,
        round_plan=SandboxRoundPlan(
            iteration=plan.iteration,
            proposal_mode=plan.proposal_mode,
            proposal_count=plan.proposal_count,
            include_initial_candidate=plan.include_initial_candidate,
        ),
        proposal_call=SandboxProposalCall(
            proposal_mode=proposal_request.proposal_mode,
            request_ordinal=proposal_request.request_ordinal,
            mutation_field=proposal_request.mutation_field,
            base_template=proposal_request.base_template,
            prompt=prompt,
            context_keys=tuple(proposal_request.context.keys()),
        ),
        drafts=tuple(draft_rows),
        mutations=tuple(mutations),
        contract=ImmutableJsonObject(contract.model_dump(mode="json")),
    )
