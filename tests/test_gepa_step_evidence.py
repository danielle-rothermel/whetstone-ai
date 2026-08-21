"""A GEPA harness step exposes the eval evidence its search paid for."""

from __future__ import annotations

from datetime import timedelta

import pytest

from whetstone.coordination.eval_service import EvalEngineService
from whetstone.core.effects.authority import EffectAuthority, ReplayPolicy
from whetstone.core.identity import compute_identity_hash
from whetstone.coordination.step_request_builder import StepRequestBuilder
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.experiment.candidate import candidate_reference
from whetstone.optim.adapters import MappingAdapterRegistry
from whetstone.optim.contracts import (
    IntentOutcome,
    OptimRun,
    OutputContract,
    SearchEvidence,
    StepMode,
    StepStatus,
)
from whetstone.optim.gepa.authorities import (
    CanonicalGepaCandidateAssembler,
    CanonicalGepaEvalAuthority,
    CanonicalGepaProposalAuthority,
    GepaCandidateFieldBinding,
    GepaDataRegistry,
)
from whetstone.optim.gepa.control import configure_gepa
from whetstone.optim.gepa.factory import CanonicalGepaAdapterFactory
from whetstone.optim.gepa.harness_adapter import (
    GEPA_ADAPTER_KEY,
    GepaHarnessAdapter,
    GepaHarnessAdapterFactory,
)
from whetstone.optim.gepa.prompts import (
    GepaComponentFormat,
    GepaPromptFormatDescriptor,
    GepaPromptServices,
    NativeGepaReflectionPromptBuilder,
    NativeGepaReflectionResponseParser,
)
from whetstone.optim.harness import OptimHarness
from whetstone.optim.proposal.proposer import (
    FakeProposerTransport,
    build_inline_proposal_executor,
    ProposerConfig,
    prompt_adapter_identity_hash,
)
from whetstone.optim.tools.facade import ToolAdmissionAuthority, ToolCallStore
from whetstone.provider.language_model import PlainPromptAdapter
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    build_toy_experiment,
    toy_template_render_contract,
)

COMPONENT = "generate"
INLINE_POLICY_HASH = compute_identity_hash(
    schema="whetstone.testing.inline_proposal_executor",
    schema_version=1,
    payload={"mode": "inline"},
)
REFLECTION_BODIES = ("Answer {prompt} in one short friendly sentence.",)


def _prompt_services() -> GepaPromptServices:
    component_schema_hash = compute_identity_hash(
        schema="whetstone.testing.gepa_component",
        schema_version=1,
        payload={"field": TOY_MUTATION_FIELD},
    )
    return GepaPromptServices(
        descriptor=GepaPromptFormatDescriptor(
            format_name="toy_prompt_template",
            components=(
                GepaComponentFormat(
                    component_name=COMPONENT,
                    component_schema_identity_hash=component_schema_hash,
                    allowed_placeholders=("prompt",),
                    required_placeholders=("prompt",),
                ),
            ),
        ),
        reflection_builder=NativeGepaReflectionPromptBuilder(),
        reflection_parser=NativeGepaReflectionResponseParser(),
    )


def _build_gepa_adapter(store, *, run_id: str, max_metric_calls: int):
    experiment = build_toy_experiment(num_seeds=1)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        store, experiment=experiment
    )
    prompt_adapter = PlainPromptAdapter()
    services = _prompt_services()
    control = configure_gepa(
        reflection_model=ProposerConfig(
            provider_call_config=engine.provider_execution_policy_ref,
        ),
        metric=engine.eval_config_ref,
        reward_policy_hash=experiment.reward_policy.identity_hash(),
        evaluation_execution_policy_hash=(
            engine.execution_policy_identity_hash()
        ),
        proposal_execution_policy_hash=(
            engine.execution_policy_identity_hash()
        ),
        proposal_prompt_adapter_identity_hash=prompt_adapter_identity_hash(
            prompt_adapter
        ),
        proposal_durability_policy_identity_hash=INLINE_POLICY_HASH,
        task_model_identity_hash=engine.task_model_identity_hash(),
        prompt_format_identity_hash=services.descriptor.identity_hash(),
        prompt_binding_identity_hash=services.binding.identity_hash(),
        trainset_task_hashes=engine.sampling.task_hashes,
        valset_task_hashes=None,
        component_names=(COMPONENT,),
        num_predictors=1,
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=1,
    )
    registry = GepaDataRegistry.from_engine(store=store, engine=engine)
    assembler = CanonicalGepaCandidateAssembler(
        base_candidate=candidate_reference(experiment.initial_candidate),
        fields=(
            GepaCandidateFieldBinding(
                component_name=COMPONENT,
                candidate_field=TOY_MUTATION_FIELD,
            ),
        ),
    )
    eval_authority = CanonicalGepaEvalAuthority(
        store=store,
        engine=engine,
        control=control,
        candidate_assembler=assembler,
        data_registry=registry,
    )
    proposal_authority = CanonicalGepaProposalAuthority(
        store=store,
        control=control,
        prompt_services=services,
        transport=FakeProposerTransport(
            {("gepa_reflection", 0): REFLECTION_BODIES},
            default=REFLECTION_BODIES,
            execution_policy_hash=engine.execution_policy_identity_hash(),
            prompt_adapter_identity_hash=prompt_adapter_identity_hash(
                prompt_adapter
            ),
        ),
        proposal_executor=build_inline_proposal_executor(
            policy_identity_hash=INLINE_POLICY_HASH,
        ),
    )
    factory = CanonicalGepaAdapterFactory(
        store=store,
        run_id=run_id,
        control=control,
        evaluation_authority=eval_authority,
        proposal_authority=proposal_authority,
        prompt_services=services,
    )
    seed_text = experiment.initial_candidate.payload[TOY_MUTATION_FIELD]
    adapter = GepaHarnessAdapter(
        control=control,
        seed_candidate={COMPONENT: seed_text},
        trainset=registry.entries,
        valset=None,
        adapter_factory=GepaHarnessAdapterFactory(factory=factory),
    )
    return experiment, engine, control, adapter, eval_authority


def _harness(store, engine, adapter):
    effect_authority = EffectAuthority.memory()
    return OptimHarness(
        store=store,
        adapter_registry=MappingAdapterRegistry({GEPA_ADAPTER_KEY: adapter}),
        tool_store=ToolCallStore(
            store,
            ToolAdmissionAuthority.memory(),
            effect_authority,
        ),
        effect_authority=effect_authority,
        owner_id="gepa-evidence-owner",
        adapter_replay_policy=ReplayPolicy.DURABLE_WORKFLOW,
        lease_duration=timedelta(minutes=5),
        evaluation_service=EvalEngineService(store=store, engine=engine),
    )


def test_a_gepa_step_carries_resolvable_eval_evidence(sqlite_store) -> None:
    """The step exposes eval/reward refs, and every ref resolves in store."""
    run_id = "gepa-evidence-run"
    experiment, engine, control, adapter, _authority = _build_gepa_adapter(
        sqlite_store, run_id=run_id, max_metric_calls=4
    )
    run = OptimRun(
        run_id=run_id,
        optimizer_config=control.reference(),
        adapter_key=GEPA_ADAPTER_KEY,
        mode=StepMode.PROPOSAL_ONLY,
        terminal_output_contract=OutputContract(returned_proposal_count=1),
        template_render_contract=toy_template_render_contract(),
        initial_candidate_ref=candidate_reference(
            experiment.initial_candidate
        ),
        mutation_field=TOY_MUTATION_FIELD,
        reward_policy=experiment.reward_policy,
    )
    harness = _harness(sqlite_store, engine, adapter)
    bound = harness.bind_run(run)
    step_request = StepRequestBuilder(store=sqlite_store).build_first(
        run=bound,
        adapter_key=GEPA_ADAPTER_KEY,
        initial_candidate=experiment.initial_candidate,
        control=control,
    )

    result, _ref = harness.run_step(step_request)

    # The search paid for evaluations, and the step says so.
    assert result.search_evidence, "GEPA step reported no eval evidence"
    for evidence in result.search_evidence:
        assert evidence.outcome is IntentOutcome.COMPLETED
        assert evidence.eval_result_ref is not None
        assert evidence.reward_ref is not None
        assert evidence.evidence_refs
        # Every cited ref resolves to a record in the store.
        for ref in evidence.evidence_refs:
            assert sqlite_store.get(ref.reference) is not None


def test_search_evidence_reward_refs_must_match_the_reward(
    sqlite_store,
) -> None:
    """The last SearchEvidence invariant, on a reward the search produced.

    reward_evidence_refs must mirror the Reward's own ordered evidence_refs;
    a mismatch means the record cites evidence the Reward does not.
    """
    run_id = "gepa-evidence-reward"
    experiment, engine, control, adapter, _authority = _build_gepa_adapter(
        sqlite_store, run_id=run_id, max_metric_calls=2
    )
    run = OptimRun(
        run_id=run_id,
        optimizer_config=control.reference(),
        adapter_key=GEPA_ADAPTER_KEY,
        mode=StepMode.PROPOSAL_ONLY,
        terminal_output_contract=OutputContract(returned_proposal_count=1),
        template_render_contract=toy_template_render_contract(),
        initial_candidate_ref=candidate_reference(
            experiment.initial_candidate
        ),
        mutation_field=TOY_MUTATION_FIELD,
        reward_policy=experiment.reward_policy,
    )
    harness = _harness(sqlite_store, engine, adapter)
    bound = harness.bind_run(run)
    request = StepRequestBuilder(store=sqlite_store).build_first(
        run=bound,
        adapter_key=GEPA_ADAPTER_KEY,
        initial_candidate=experiment.initial_candidate,
        control=control,
    )
    result, _ref = harness.run_step(request)
    evidence = next(
        item for item in result.search_evidence if item.reward_ref is not None
    )

    # Rebuilding it unchanged is valid.
    SearchEvidence(
        eval_request_id=evidence.eval_request_id,
        optim_run_id=evidence.optim_run_id,
        optim_step_index=evidence.optim_step_index,
        candidate=evidence.candidate,
        outcome=evidence.outcome,
        eval_result_ref=evidence.eval_result_ref,
        reward_ref=evidence.reward_ref,
        reward_evidence_refs=evidence.reward_evidence_refs,
    )

    # Dropping one cited ref no longer mirrors the Reward.
    assert evidence.reward_evidence_refs
    with pytest.raises(ValueError, match="must equal the ordered"):
        SearchEvidence(
            eval_request_id=evidence.eval_request_id,
            optim_run_id=evidence.optim_run_id,
            optim_step_index=evidence.optim_step_index,
            candidate=evidence.candidate,
            outcome=evidence.outcome,
            eval_result_ref=evidence.eval_result_ref,
            reward_ref=evidence.reward_ref,
            reward_evidence_refs=evidence.reward_evidence_refs[:-1],
        )


def test_a_terminal_gepa_step_carries_its_search_evidence(
    sqlite_store,
) -> None:
    """The COMPLETE step reports evidence too, not just continuing steps.

    Both terminal branches of the GEPA harness adapter -- seed-retained and
    accepted-candidate -- attach search_evidence, and nothing else asserts it
    on a terminal step.
    """
    run_id = "gepa-evidence-terminal"
    experiment, engine, control, adapter, _authority = _build_gepa_adapter(
        sqlite_store, run_id=run_id, max_metric_calls=2
    )
    run = OptimRun(
        run_id=run_id,
        optimizer_config=control.reference(),
        adapter_key=GEPA_ADAPTER_KEY,
        mode=StepMode.PROPOSAL_ONLY,
        terminal_output_contract=OutputContract(returned_proposal_count=1),
        template_render_contract=toy_template_render_contract(),
        initial_candidate_ref=candidate_reference(
            experiment.initial_candidate
        ),
        mutation_field=TOY_MUTATION_FIELD,
        reward_policy=experiment.reward_policy,
    )
    harness = _harness(sqlite_store, engine, adapter)
    bound = harness.bind_run(run)
    builder = StepRequestBuilder(store=sqlite_store)
    request = builder.build_first(
        run=bound,
        adapter_key=GEPA_ADAPTER_KEY,
        initial_candidate=experiment.initial_candidate,
        control=control,
    )

    result, result_ref = harness.run_step(request)
    # Walk to termination; every step must carry its own evidence.
    while result.status is StepStatus.CONTINUE:
        assert result.search_evidence, "a continuing step reported no evidence"
        request = builder.build_next(
            prior=result,
            prior_ref=result_ref,
            prior_results=(result,),
            control=control,
            mutation_field=TOY_MUTATION_FIELD,
        )
        result, result_ref = harness.run_step(request)

    assert result.status is StepStatus.COMPLETE
    assert result.search_evidence, "the terminal step reported no evidence"
    for evidence in result.search_evidence:
        assert evidence.eval_result_ref is not None
        for ref in evidence.evidence_refs:
            assert sqlite_store.get(ref.reference) is not None


def test_gepa_step_evidence_is_per_step_not_cumulative(sqlite_store) -> None:
    """A second step reports only the evaluations that step drove."""
    run_id = "gepa-evidence-two-step"
    experiment, engine, control, adapter, _authority = _build_gepa_adapter(
        sqlite_store, run_id=run_id, max_metric_calls=8
    )
    run = OptimRun(
        run_id=run_id,
        optimizer_config=control.reference(),
        adapter_key=GEPA_ADAPTER_KEY,
        mode=StepMode.PROPOSAL_ONLY,
        terminal_output_contract=OutputContract(returned_proposal_count=1),
        template_render_contract=toy_template_render_contract(),
        initial_candidate_ref=candidate_reference(
            experiment.initial_candidate
        ),
        mutation_field=TOY_MUTATION_FIELD,
        reward_policy=experiment.reward_policy,
    )
    harness = _harness(sqlite_store, engine, adapter)
    bound = harness.bind_run(run)
    builder = StepRequestBuilder(store=sqlite_store)
    first_request = builder.build_first(
        run=bound,
        adapter_key=GEPA_ADAPTER_KEY,
        initial_candidate=experiment.initial_candidate,
        control=control,
    )

    first, first_ref = harness.run_step(first_request)
    # A budget change that terminalizes step 1 must fail here rather than
    # silently skip the per-step assertion this test exists for.
    assert first.status is StepStatus.CONTINUE
    second_request = builder.build_next(
        prior=first,
        prior_ref=first_ref,
        prior_results=(first,),
        control=control,
        mutation_field=TOY_MUTATION_FIELD,
    )
    second, _second_ref = harness.run_step(second_request)

    first_ids = {e.eval_request_id for e in first.search_evidence}
    second_ids = {e.eval_request_id for e in second.search_evidence}
    assert first_ids
    assert second_ids
    # Step 2 reports its own evaluations, not step 1's replayed again.
    # Disjoint, not merely unequal: re-reporting step 1's evidence alongside
    # a new entry is exactly the cumulative failure this guards against.
    assert not (first_ids & second_ids)


def _gepa_run(experiment, control, *, run_id: str) -> OptimRun:
    return OptimRun(
        run_id=run_id,
        optimizer_config=control.reference(),
        adapter_key=GEPA_ADAPTER_KEY,
        mode=StepMode.PROPOSAL_ONLY,
        terminal_output_contract=OutputContract(returned_proposal_count=1),
        template_render_contract=toy_template_render_contract(),
        initial_candidate_ref=candidate_reference(
            experiment.initial_candidate
        ),
        mutation_field=TOY_MUTATION_FIELD,
        reward_policy=experiment.reward_policy,
    )


def test_gepa_eval_requests_carry_the_harness_step_index(
    sqlite_store,
) -> None:
    """Two GEPA steps must not mint identical eval intents.

    ``run_one_gepa_iteration`` resets the effect ordinal every step, so an
    ``optim_step_index`` taken from the effect ordinal would let two steps
    that replay the same candidate on the same batch produce byte-identical
    ``OptimEvalRequest`` values -- and therefore identical intent and claim
    keys inside ``EvalEngineService``.
    """
    from whetstone.coordination.eval_service import EvalEngineService as _Svc

    run_id = "gepa-step-index"
    experiment, engine, control, adapter, authority = _build_gepa_adapter(
        sqlite_store, run_id=run_id, max_metric_calls=8
    )
    run = _gepa_run(experiment, control, run_id=run_id)
    harness = _harness(sqlite_store, engine, adapter)
    bound = harness.bind_run(run)
    builder = StepRequestBuilder(store=sqlite_store)

    first_request = builder.build_first(
        run=bound,
        adapter_key=GEPA_ADAPTER_KEY,
        initial_candidate=experiment.initial_candidate,
        control=control,
    )
    first, first_ref = harness.run_step(first_request)
    assert first.status is StepStatus.CONTINUE
    first_requests = tuple(
        resolution.optim_eval_request
        for resolution in authority.resolved_intents
    )

    second_request = builder.build_next(
        prior=first,
        prior_ref=first_ref,
        prior_results=(first,),
        control=control,
        mutation_field=TOY_MUTATION_FIELD,
    )
    second, _second_ref = harness.run_step(second_request)
    second_requests = tuple(
        resolution.optim_eval_request
        for resolution in authority.resolved_intents
    )

    assert first_requests and second_requests
    # The index is the harness step index, not a per-step effect ordinal.
    assert {r.optim_step_index for r in first_requests} == {0}
    assert {r.optim_step_index for r in second_requests} == {1}
    # And that difference reaches the keys EvalEngineService derives.
    first_keys = {_Svc._intent_ref(r).content_hash for r in first_requests}
    second_keys = {_Svc._intent_ref(r).content_hash for r in second_requests}
    assert not (first_keys & second_keys)
    # The step evidence agrees with the requests it projects.
    assert {e.optim_step_index for e in first.search_evidence} == {0}
    assert {e.optim_step_index for e in second.search_evidence} == {1}
    assert {e.optim_run_id for e in second.search_evidence} == {run_id}


def test_search_evidence_bound_to_another_step_is_rejected(
    sqlite_store,
) -> None:
    """The harness verifies the binding rather than trusting the adapter."""
    from whetstone.optim.adapters import AdapterOutput

    run_id = "gepa-evidence-misbound"
    experiment, engine, control, adapter, _authority = _build_gepa_adapter(
        sqlite_store, run_id=run_id, max_metric_calls=8
    )
    run = _gepa_run(experiment, control, run_id=run_id)
    harness = _harness(sqlite_store, engine, adapter)
    bound = harness.bind_run(run)
    builder = StepRequestBuilder(store=sqlite_store)
    request = builder.build_first(
        run=bound,
        adapter_key=GEPA_ADAPTER_KEY,
        initial_candidate=experiment.initial_candidate,
        control=control,
    )
    result, _ref = harness.run_step(request)
    truthful = result.search_evidence[0]

    wrong_step = truthful.model_copy(
        update={"optim_step_index": truthful.optim_step_index + 1}
    )
    with pytest.raises(ValueError, match="another optimization step"):
        harness._validate_output_intents(
            request,
            AdapterOutput(search_evidence=(wrong_step,)),
        )

    wrong_run = truthful.model_copy(update={"optim_run_id": "other-run"})
    with pytest.raises(ValueError, match="another optimization run"):
        harness._validate_output_intents(
            request,
            AdapterOutput(search_evidence=(wrong_run,)),
        )
