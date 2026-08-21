"""Per-optimizer step contracts and no-improvement termination."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.test_gepa_harness_adapter import _toy_gepa_control
from whetstone.coordination.step_contracts import (
    resolve_step_contract_provider,
    step_contract_provider_keys,
)
from whetstone.coordination.step_request_builder import StepRequestBuilder
from whetstone.core.identity import (
    IdentityRef,
    ImmutableJsonObject,
    TypedRef,
    typed_ref_for_record,
)
from whetstone.experiment.candidate import Candidate, candidate_reference
from whetstone.optim.adapters import AdapterOutput
from whetstone.optim.contracts import (
    OptimRun,
    OptimStepRequest,
    OptimStepResult,
    OutputContract,
    StepKind,
    StepMode,
    StepStatus,
    optimization_run_reference,
    step_request_reference,
)
from whetstone.optim.harness import OptimHarness
from whetstone.optim.copro.adapter import COPRO_ADAPTER_KEY
from whetstone.optim.gepa.adapter import GepaTerminalResult
from whetstone.optim.gepa.engine import GepaDetailedResult
from whetstone.optim.gepa.harness_adapter import (
    GEPA_ADAPTER_KEY,
    GepaHarnessAdapter,
    GepaHarnessAdapterFactory,
)
from whetstone.optim.gepa.step_contract import gepa_step_output_contract
from whetstone.optim.gepa.step_engine import GepaStepCheckpoint
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    build_toy_experiment,
    toy_template_render_contract,
)

SEED_COMPONENTS = {"generate": "seed"}


def _gepa_run(control, *, run_id: str = "gepa-seed-retained") -> OptimRun:
    experiment = build_toy_experiment(num_seeds=1)
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


# --- registry -------------------------------------------------------------


def test_every_registered_optimizer_declares_its_own_key() -> None:
    assert set(step_contract_provider_keys()) == {
        COPRO_ADAPTER_KEY,
        GEPA_ADAPTER_KEY,
    }
    for key in step_contract_provider_keys():
        assert resolve_step_contract_provider(key).adapter_key == key


def test_an_unregistered_adapter_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="no optimizer step contract"):
        resolve_step_contract_provider("miprov2")


# --- contract shape -------------------------------------------------------


def test_gepa_first_step_is_honest_about_terminalizing(tmp_path) -> None:
    """A GEPA first step may complete, so its contract permits that."""
    control = _toy_gepa_control(
        max_metric_calls=4,
        sqlite_path=str(tmp_path / "gepa-first.sqlite"),
    )
    run = _gepa_run(control)
    run_ref = optimization_run_reference(run)
    experiment = build_toy_experiment(num_seeds=1)

    request = StepRequestBuilder(store=MagicMock()).build_first(
        run=run_ref,
        adapter_key=GEPA_ADAPTER_KEY,
        initial_candidate=experiment.initial_candidate,
        control=control,
    )
    contract = request.step_output_contract

    # The budget is nowhere near exhausted, yet the step may still complete.
    assert control.resolved_max_metric_calls == 4
    assert contract.accepted_count_for(StepStatus.CONTINUE) == 0
    assert contract.accepted_count_for(StepStatus.COMPLETE) == 1
    assert contract.honors_terminal(run.terminal_output_contract)
    assert contract == gepa_step_output_contract(run_ref)


def test_gepa_contract_derives_the_terminal_cardinality(tmp_path) -> None:
    """The GEPA step contract honors a split run terminal contract.

    A run may bind ``returned_proposal_count`` and
    ``terminal_proposal_count`` to different values. ``honors_terminal``
    compares COMPLETE cardinalities, so deriving the GEPA step contract from
    the run's *continuing* count would reject every honest completing step.
    """
    control = _toy_gepa_control(
        max_metric_calls=4,
        sqlite_path=str(tmp_path / "gepa-split-terminal.sqlite"),
    )
    experiment = build_toy_experiment(num_seeds=1)
    terminal = OutputContract(
        returned_proposal_count=3,
        terminal_proposal_count=1,
    )
    assert terminal.returned_proposal_count != terminal.terminal_proposal_count
    run = OptimRun(
        run_id="gepa-split-terminal",
        optimizer_config=control.reference(),
        adapter_key=GEPA_ADAPTER_KEY,
        mode=StepMode.PROPOSAL_ONLY,
        terminal_output_contract=terminal,
        template_render_contract=toy_template_render_contract(),
        initial_candidate_ref=candidate_reference(
            experiment.initial_candidate
        ),
        mutation_field=TOY_MUTATION_FIELD,
        reward_policy=experiment.reward_policy,
    )
    run_ref = optimization_run_reference(run)

    contract = gepa_step_output_contract(run_ref)

    # The run's COMPLETE cardinality, not its continuing count of 3.
    assert contract.accepted_count_for(StepStatus.COMPLETE) == 1
    assert contract.accepted_count_for(StepStatus.CONTINUE) == 0
    assert contract.honors_terminal(terminal)


def test_copro_contracts_are_unchanged(copro_launch) -> None:
    """COPRO's seed round still asks for breadth - 1 proposals."""
    runtime, launch = copro_launch
    bound = runtime.harness.bind_run(launch.run)
    control = launch.control

    request = StepRequestBuilder(store=runtime.store).build_first(
        run=bound,
        adapter_key=COPRO_ADAPTER_KEY,
        initial_candidate=launch.initial_candidate,
        control=control,
    )

    assert request.kind_label == "seed_proposal"
    assert request.step_output_contract == OutputContract(
        returned_proposal_count=control.breadth - 1,
    )
    assert request.step_output_contract.terminal_proposal_count is None
    assert not request.pools["attempt_history"]


# --- honoring the run terminal contract -----------------------------------


def test_honors_terminal_accepts_a_differing_continuing_count() -> None:
    """The intended relaxation: only the terminal side has to agree."""
    terminal = OutputContract(returned_proposal_count=1)
    step = OutputContract(
        returned_proposal_count=0,
        terminal_proposal_count=1,
    )
    assert step.honors_terminal(terminal)


def test_honors_terminal_rejects_a_differing_terminal_count() -> None:
    terminal = OutputContract(returned_proposal_count=2)
    step = OutputContract(
        returned_proposal_count=0,
        terminal_proposal_count=1,
    )
    assert not step.honors_terminal(terminal)


def test_honors_terminal_rejects_a_differing_distinct_base_rule() -> None:
    terminal = OutputContract(
        returned_proposal_count=1,
        require_distinct_bases=True,
    )
    step = OutputContract(
        returned_proposal_count=0,
        terminal_proposal_count=1,
        require_distinct_bases=False,
    )
    assert not step.honors_terminal(terminal)


def test_a_complete_step_with_a_foreign_terminal_count_is_rejected() -> None:
    """The predicate is a real guard on the harness path, not decoration."""
    experiment = build_toy_experiment(num_seeds=1)
    base = experiment.initial_candidate
    run = _distinct_base_run(
        "honors-terminal",
        OutputContract(returned_proposal_count=1),
    )
    # The step promises to terminalize on 2 accepted candidates; the run
    # terminal contract says 1.
    step_contract = OutputContract(
        returned_proposal_count=0,
        terminal_proposal_count=2,
    )
    request = OptimStepRequest(
        run=optimization_run_reference(run),
        step_id="honors-terminal:0",
        kind=StepKind.PROPOSAL,
        step_index=0,
        candidates=(base,),
        step_output_contract=step_contract,
    )
    first = _derived(base, candidate_id="one")
    second = _root_candidate("other-base")
    output = AdapterOutput(
        proposed_candidates=(first, second),
        accepted_candidates=(first, second),
        proposed_status=StepStatus.COMPLETE,
    )

    with pytest.raises(ValueError, match="honor the run terminal"):
        OptimHarness._validate_output(request, output)


def test_a_complete_step_result_with_a_foreign_terminal_count_is_rejected(
) -> None:
    """The same guard binds the persisted Step Result."""
    experiment = build_toy_experiment(num_seeds=1)
    base = experiment.initial_candidate
    run = _distinct_base_run(
        "honors-terminal-result",
        OutputContract(returned_proposal_count=1),
    )
    step_contract = OutputContract(
        returned_proposal_count=0,
        terminal_proposal_count=1,
        require_distinct_bases=True,
    )
    request = OptimStepRequest(
        run=optimization_run_reference(run),
        step_id="honors-terminal-result:0",
        kind=StepKind.PROPOSAL,
        step_index=0,
        candidates=(base,),
        step_output_contract=step_contract,
    )
    proposed = candidate_reference(_derived(base, candidate_id="one"))

    # Cardinality matches; only the distinct-base rule disagrees with the run.
    with pytest.raises(ValueError, match="honor the run terminal"):
        OptimStepResult(
            request=step_request_reference(request),
            proposed_candidates=(proposed,),
            accepted_candidates=(proposed,),
            status=StepStatus.COMPLETE,
        )


# --- no-improvement termination -------------------------------------------


def _seed_retained_step(tmp_path, *, best_candidate: dict[str, str]):
    control = _toy_gepa_control(
        max_metric_calls=1,
        sqlite_path=str(tmp_path / "gepa-retain.sqlite"),
    )
    run = _gepa_run(control)
    run_ref = optimization_run_reference(run)
    experiment = build_toy_experiment(num_seeds=1)
    factory = MagicMock()
    factory.create.return_value = MagicMock()
    factory.search_evidence.return_value = ()
    factory.skipped_mutations.return_value = ()
    factory.persist_result.return_value = TypedRef(
        schema_name="whetstone.gepa.result",
        content_hash="a" * 64,
    )
    detailed = GepaDetailedResult(
        candidates=(dict(SEED_COMPONENTS),),
        parents=((None,),),
        val_aggregate_scores=(0.0,),
        val_subscores=({"task": 0.0},),
        per_val_instance_best_candidates={"task": (0,)},
        discovery_eval_counts=(1,),
        seed=0,
        best_idx=0,
        control_identity_hash=control.identity_hash(),
    )
    adapter = GepaHarnessAdapter(
        control=control,
        seed_candidate=dict(SEED_COMPONENTS),
        trainset=(),
        valset=None,
        adapter_factory=GepaHarnessAdapterFactory(factory=factory),
    )
    request = OptimStepRequest(
        run=run_ref,
        step_id=f"{run.run_id}:gepa:0",
        kind=StepKind.PROPOSAL,
        kind_label="gepa_iteration",
        step_index=0,
        candidates=(experiment.initial_candidate,),
        hyperparameters=ImmutableJsonObject(
            control.step_hyperparameters(iteration=0)
        ),
        budget=request_budget(control),
        step_output_contract=gepa_step_output_contract(run_ref),
    )
    with patch(
        "whetstone.optim.gepa.harness_adapter.run_one_gepa_iteration",
        return_value=(
            detailed,
            GepaStepCheckpoint(metric_calls_consumed=1, terminal=True),
        ),
    ), patch(
        "whetstone.optim.gepa.harness_adapter.project_gepa_terminal",
        return_value=GepaTerminalResult(
            best_candidate=best_candidate,
            control_identity_hash=control.identity_hash(),
            artifact_ref=TypedRef(
                schema_name="whetstone.gepa.result",
                content_hash="a" * 64,
            ),
        ),
    ):
        return adapter.invoke(request, ())


def request_budget(control):
    from whetstone.optim.contracts import BudgetState

    return BudgetState(
        remaining=ImmutableJsonObject(
            {"metric_calls": control.resolved_max_metric_calls}
        ),
    )


def test_gepa_terminalizes_with_the_seed_retained(tmp_path) -> None:
    """No accepted improvement is a clean completion, not a fake candidate."""
    output = _seed_retained_step(tmp_path, best_candidate=dict(SEED_COMPONENTS))

    assert output.proposed_status is StepStatus.COMPLETE
    assert output.seed_retained is True
    assert output.accepted_candidates == ()
    assert output.proposed_candidates == ()


def test_gepa_terminalizes_with_an_improvement(tmp_path) -> None:
    output = _seed_retained_step(
        tmp_path,
        best_candidate={"generate": "Answer {prompt} in one sentence."},
    )

    assert output.proposed_status is StepStatus.COMPLETE
    assert output.seed_retained is False
    assert len(output.accepted_candidates) == 1


def test_a_seed_retaining_output_cannot_accept_candidates() -> None:
    experiment = build_toy_experiment(num_seeds=1)
    with pytest.raises(ValueError, match="accepts no candidates"):
        AdapterOutput(
            proposed_candidates=(experiment.initial_candidate,),
            accepted_candidates=(experiment.initial_candidate,),
            proposed_status=StepStatus.COMPLETE,
            seed_retained=True,
        )


def test_only_a_complete_output_may_retain_the_seed() -> None:
    with pytest.raises(ValueError, match="only a COMPLETE"):
        AdapterOutput(
            proposed_status=StepStatus.CONTINUE,
            seed_retained=True,
        )


# --- distinct bases -------------------------------------------------------


def _root_candidate(candidate_id: str) -> Candidate:
    """A second root candidate, so a request can offer two distinct bases."""
    candidate = Candidate(
        candidate_id=candidate_id,
        base_ref=typed_ref_for_record(
            "whetstone.test.root", {"root": candidate_id}
        ),
        payload={
            TOY_MUTATION_FIELD: f"{candidate_id}, reply to: {{prompt}}"
        },
    )
    return candidate_reference(candidate).record


def _derived(base: Candidate, *, candidate_id: str) -> Candidate:
    """A candidate mutated from ``base``, so it binds ``base`` as its base."""
    candidate = Candidate(
        candidate_id=candidate_id,
        base_ref=candidate_reference(base).record_ref,
        payload={
            TOY_MUTATION_FIELD: f"{candidate_id}, reply to: {{prompt}}"
        },
    )
    return candidate_reference(candidate).record


def _distinct_base_run(run_id: str, contract: OutputContract) -> OptimRun:
    experiment = build_toy_experiment(num_seeds=1)
    return OptimRun(
        run_id=run_id,
        optimizer_config=IdentityRef(
            record_ref=TypedRef(
                schema_name="whetstone.optim_control",
                content_hash="b" * 64,
            ),
            record_hash="c" * 64,
        ),
        adapter_key=COPRO_ADAPTER_KEY,
        mode=StepMode.PROPOSAL_ONLY,
        terminal_output_contract=contract,
        template_render_contract=toy_template_render_contract(),
        initial_candidate_ref=candidate_reference(
            experiment.initial_candidate
        ),
        mutation_field=TOY_MUTATION_FIELD,
        reward_policy=experiment.reward_policy,
    )


def _distinct_base_request(
    run_id: str,
    contract: OutputContract,
    bases: tuple[Candidate, ...],
) -> OptimStepRequest:
    """A step request bound to a require_distinct_bases terminal contract."""
    run = _distinct_base_run(run_id, contract)
    return OptimStepRequest(
        run=optimization_run_reference(run),
        step_id=f"{run_id}:0",
        kind=StepKind.PROPOSAL,
        step_index=0,
        candidates=bases,
        step_output_contract=contract,
    )


def test_distinct_bases_rejects_duplicate_based_proposed_candidates() -> None:
    """The rule the contract states: proposed candidates need distinct bases."""
    experiment = build_toy_experiment(num_seeds=1)
    base = experiment.initial_candidate
    contract = OutputContract(
        returned_proposal_count=1,
        require_distinct_bases=True,
    )
    request = _distinct_base_request("distinct-proposed", contract, (base,))
    # Both proposals are mutations of the one request candidate, so they
    # share a base_ref.
    first = _derived(base, candidate_id="one")
    second = _derived(base, candidate_id="two")
    output = AdapterOutput(
        proposed_candidates=(first, second),
        accepted_candidates=(first,),
        proposed_status=StepStatus.COMPLETE,
    )

    with pytest.raises(ValueError, match="distinct-base output contract"):
        OptimHarness._validate_output(request, output)


def test_distinct_bases_allows_duplicate_based_accepted_candidates() -> None:
    """D1: the rule constrains proposed only, so accepted may repeat a base.

    This is what makes seed-retained termination representable: an adapter
    must not be forced to fabricate a distinct base to accept a candidate.
    """
    base_a = build_toy_experiment(num_seeds=1).initial_candidate
    contract = OutputContract(
        returned_proposal_count=2,
        require_distinct_bases=False,
    )
    request = _distinct_base_request(
        "distinct-accepted", contract, (base_a,)
    )
    # Two distinct candidates mutated from the one base: their base_refs
    # collide, which the pre-D1 rule rejected on accepted_candidates.
    first = _derived(base_a, candidate_id="one")
    second = _derived(base_a, candidate_id="two")
    output = AdapterOutput(
        proposed_candidates=(first, second),
        accepted_candidates=(first, second),
        proposed_status=StepStatus.CONTINUE,
    )

    # Accepted candidates sharing a base are legal; the harness only checks
    # proposed, so this passes and would have raised before D1.
    OptimHarness._validate_output(request, output)

    # Turning the rule on now rejects it, because proposed shares the base.
    strict = request.model_copy(
        update={
            "step_output_contract": contract.model_copy(
                update={"require_distinct_bases": True}
            )
        }
    )
    with pytest.raises(ValueError, match="distinct-base output contract"):
        OptimHarness._validate_output(strict, output)


def test_step_result_rejects_duplicate_based_proposed_candidates() -> None:
    """The Step Result validator enforces the same rule as the harness."""
    experiment = build_toy_experiment(num_seeds=1)
    base = experiment.initial_candidate
    contract = OutputContract(
        returned_proposal_count=1,
        require_distinct_bases=True,
    )
    request = _distinct_base_request("distinct-step-result", contract, (base,))
    first = candidate_reference(_derived(base, candidate_id="one"))
    second = candidate_reference(_derived(base, candidate_id="two"))

    with pytest.raises(ValueError, match="distinct-base output"):
        OptimStepResult(
            request=step_request_reference(request),
            proposed_candidates=(first, second),
            accepted_candidates=(first,),
            status=StepStatus.COMPLETE,
        )


def test_duplicate_based_terminal_proposals_are_unreachable() -> None:
    """No OptimResult-level distinct-base check is needed, and why.

    Terminal proposals equal the final Step's accepted candidates, accepted
    is a sub-multiset of proposed, and a COMPLETE Step must honor the run's
    distinct-base flag. So the Step Result rule already excludes duplicate
    bases from the terminal proposals; this pins that derivation.
    """
    base_a = build_toy_experiment(num_seeds=1).initial_candidate
    base_b = _root_candidate("second-base")
    contract = OutputContract(
        returned_proposal_count=2,
        require_distinct_bases=True,
    )
    run = _distinct_base_run("distinct-terminal", contract)
    run_ref = optimization_run_reference(run)
    request = OptimStepRequest(
        run=run_ref,
        step_id="distinct-terminal:0",
        kind=StepKind.PROPOSAL,
        step_index=0,
        candidates=(base_a, base_b),
        step_output_contract=contract,
    )
    # The only way to accept two candidates sharing base_a is to propose
    # them both, which the Step Result rule rejects outright.
    first = candidate_reference(_derived(base_a, candidate_id="one"))
    second = candidate_reference(_derived(base_a, candidate_id="two"))
    with pytest.raises(ValueError, match="distinct-base output"):
        OptimStepResult(
            request=step_request_reference(request),
            proposed_candidates=(first, second),
            accepted_candidates=(first, second),
            status=StepStatus.COMPLETE,
        )

    # A COMPLETE step cannot escape by relaxing its own contract, because
    # honors_terminal compares require_distinct_bases against the run.
    relaxed = contract.model_copy(update={"require_distinct_bases": False})
    assert not relaxed.honors_terminal(run.terminal_output_contract)


def test_optim_result_seed_retained_must_mirror_the_final_step() -> None:
    """The consumer reads seed_retained off the run, so it cannot disagree.

    This is the field that distinguishes a real no-improvement run from a
    fabricated substitute candidate.
    """
    from whetstone.optim.contracts import OptimResult, step_result_reference

    experiment = build_toy_experiment(num_seeds=1)
    contract = OutputContract(returned_proposal_count=1)
    run = _distinct_base_run("seed-retained-mirror", contract)
    run_ref = optimization_run_reference(run)
    request = OptimStepRequest(
        run=run_ref,
        step_id="seed-retained-mirror:0",
        kind=StepKind.PROPOSAL,
        step_index=0,
        candidates=(experiment.initial_candidate,),
        # Search-dependent terminal cardinality: the only contract shape
        # that may report seed_retained at all.
        step_output_contract=contract.model_copy(
            update={"terminal_proposal_count": 1}
        ),
    )
    step_result = OptimStepResult(
        request=step_request_reference(request),
        status=StepStatus.COMPLETE,
        seed_retained=True,
        retained_candidate_ref=candidate_reference(
            experiment.initial_candidate
        ),
    )

    with pytest.raises(ValueError, match="seed_retained must match"):
        OptimResult(
            run=run_ref,
            proposals=(),
            step_results=(step_result_reference(step_result),),
            seed_retained=False,
        )


def test_a_seed_retaining_run_terminalizes_through_the_harness(
    tmp_path, sqlite_store
) -> None:
    """A whole run whose best stays the seed reaches a clean OptimResult."""
    from datetime import timedelta

    from whetstone.coordination.eval_service import EvalEngineService
    from whetstone.core.effects.authority import EffectAuthority, ReplayPolicy
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
    from whetstone.optim.adapters import MappingAdapterRegistry
    from whetstone.optim.contracts import step_result_reference
    from whetstone.optim.harness import OptimHarness
    from whetstone.optim.tools.facade import (
        ToolAdmissionAuthority,
        ToolCallStore,
    )

    control = _toy_gepa_control(
        max_metric_calls=1,
        sqlite_path=str(tmp_path / "gepa-retain-e2e.sqlite"),
    )
    run = _gepa_run(control, run_id="gepa-retain-e2e")
    experiment = build_toy_experiment(num_seeds=1)
    engine = ReferenceEvalRuntimeConfig().build_engine(sqlite_store)
    factory = MagicMock()
    factory.create.return_value = MagicMock()
    factory.search_evidence.return_value = ()
    factory.skipped_mutations.return_value = ()
    factory.persist_result.return_value = TypedRef(
        schema_name="whetstone.gepa.result",
        content_hash="a" * 64,
    )
    detailed = GepaDetailedResult(
        candidates=(dict(SEED_COMPONENTS),),
        parents=((None,),),
        val_aggregate_scores=(0.0,),
        val_subscores=({"task": 0.0},),
        per_val_instance_best_candidates={"task": (0,)},
        discovery_eval_counts=(1,),
        seed=0,
        best_idx=0,
        control_identity_hash=control.identity_hash(),
    )
    adapter = GepaHarnessAdapter(
        control=control,
        seed_candidate=dict(SEED_COMPONENTS),
        trainset=(),
        valset=None,
        adapter_factory=GepaHarnessAdapterFactory(factory=factory),
    )
    effect_authority = EffectAuthority.memory()
    harness = OptimHarness(
        store=sqlite_store,
        adapter_registry=MappingAdapterRegistry({GEPA_ADAPTER_KEY: adapter}),
        tool_store=ToolCallStore(
            sqlite_store,
            ToolAdmissionAuthority.memory(),
            effect_authority,
        ),
        effect_authority=effect_authority,
        owner_id="gepa-retain-owner",
        adapter_replay_policy=ReplayPolicy.DURABLE_WORKFLOW,
        lease_duration=timedelta(minutes=5),
        evaluation_service=EvalEngineService(
            store=sqlite_store, engine=engine
        ),
    )
    bound = harness.bind_run(run)
    step_request = StepRequestBuilder(store=sqlite_store).build_first(
        run=bound,
        adapter_key=GEPA_ADAPTER_KEY,
        initial_candidate=experiment.initial_candidate,
        control=control,
    )

    with patch(
        "whetstone.optim.gepa.harness_adapter.run_one_gepa_iteration",
        return_value=(
            detailed,
            GepaStepCheckpoint(metric_calls_consumed=1, terminal=True),
        ),
    ), patch(
        "whetstone.optim.gepa.harness_adapter.project_gepa_terminal",
        return_value=GepaTerminalResult(
            best_candidate=dict(SEED_COMPONENTS),
            control_identity_hash=control.identity_hash(),
            artifact_ref=TypedRef(
                schema_name="whetstone.gepa.result",
                content_hash="a" * 64,
            ),
        ),
    ):
        result, _ref = harness.run_step(step_request)

    assert result.status is StepStatus.COMPLETE
    assert result.seed_retained is True
    assert result.accepted_candidates == ()
    assert result.terminal_failure is None

    terminal, _terminal_ref = harness.terminalize(
        run=bound,
        step_results=(step_result_reference(result),),
    )

    assert terminal.seed_retained is True
    assert terminal.proposals == ()
    assert terminal.terminal_failure is None
    assert terminal.status is StepStatus.COMPLETE


# --- seed retention is a narrow, contract-gated exemption ------------------


def _seed_retaining_output(experiment) -> AdapterOutput:
    return AdapterOutput(
        proposed_status=StepStatus.COMPLETE,
        seed_retained=True,
        retained_candidate=experiment.initial_candidate,
    )


def _step_request_for(run, *, contract: OutputContract, experiment):
    return OptimStepRequest(
        run=optimization_run_reference(run),
        step_id=f"{run.run_id}:0",
        kind=StepKind.PROPOSAL,
        step_index=0,
        candidates=(experiment.initial_candidate,),
        step_output_contract=contract,
    )


def test_a_copro_style_contract_cannot_retain_the_seed() -> None:
    """Unconditional terminal cardinality admits no seed-retained exemption.

    Without this gate any adapter could zero out its own terminal proposal
    cardinality just by setting ``seed_retained``.
    """
    experiment = build_toy_experiment(num_seeds=1)
    contract = OutputContract(returned_proposal_count=1)
    run = _distinct_base_run("copro-style-seed", contract)
    request = _step_request_for(
        run, contract=contract, experiment=experiment
    )

    with pytest.raises(ValueError, match="terminal_proposal_count"):
        OptimHarness._validate_output(
            request, _seed_retaining_output(experiment)
        )


def test_a_gepa_style_contract_may_retain_the_true_seed() -> None:
    experiment = build_toy_experiment(num_seeds=1)
    contract = OutputContract(
        returned_proposal_count=0,
        terminal_proposal_count=1,
    )
    run = _distinct_base_run("gepa-style-seed", contract)
    request = _step_request_for(
        run, contract=contract, experiment=experiment
    )

    OptimHarness._validate_output(
        request, _seed_retaining_output(experiment)
    )


def test_a_gepa_style_contract_cannot_retain_a_nonseed_candidate() -> None:
    """``seed_retained`` names the run's seed, not whatever search liked."""
    experiment = build_toy_experiment(num_seeds=1)
    contract = OutputContract(
        returned_proposal_count=0,
        terminal_proposal_count=1,
    )
    run = _distinct_base_run("gepa-style-nonseed", contract)
    request = _step_request_for(
        run, contract=contract, experiment=experiment
    )
    impostor = _derived(experiment.initial_candidate, candidate_id="impostor")

    with pytest.raises(ValueError, match="exact run initial candidate"):
        OptimHarness._validate_output(
            request,
            AdapterOutput(
                proposed_status=StepStatus.COMPLETE,
                seed_retained=True,
                retained_candidate=impostor,
            ),
        )


# --- skipped reflection mutations are durable per step --------------------


def test_a_nonterminal_step_persists_its_skipped_mutations(tmp_path) -> None:
    """A skip on a continuing step must not wait for the terminal transcript.

    ``GepaSkippedMutation`` used to reach durable state only through the
    terminal effect transcript, so a process death after a non-terminal skip
    lost it entirely.
    """
    from whetstone.optim.gepa.contracts import GepaSkippedMutation
    from whetstone.optim.gepa.harness_adapter import (
        GEPA_SKIPPED_MUTATIONS_KEY,
    )

    control = _toy_gepa_control(
        max_metric_calls=4,
        sqlite_path=str(tmp_path / "gepa-skip.sqlite"),
    )
    run = _gepa_run(control, run_id="gepa-skip-durable")
    run_ref = optimization_run_reference(run)
    experiment = build_toy_experiment(num_seeds=1)
    skipped = GepaSkippedMutation(
        component_name="generate",
        attempt_ordinal=1,
        rejection_detail="omitted required placeholders",
        raw_response="no placeholder here",
        exhausted=True,
    )
    factory = MagicMock()
    factory.create.return_value = MagicMock()
    factory.search_evidence.return_value = ()
    factory.skipped_mutations.return_value = (skipped,)
    adapter = GepaHarnessAdapter(
        control=control,
        seed_candidate=dict(SEED_COMPONENTS),
        trainset=(),
        valset=None,
        adapter_factory=GepaHarnessAdapterFactory(factory=factory),
    )
    request = OptimStepRequest(
        run=run_ref,
        step_id=f"{run.run_id}:gepa:0",
        kind=StepKind.PROPOSAL,
        kind_label="gepa_iteration",
        step_index=0,
        candidates=(experiment.initial_candidate,),
        hyperparameters=ImmutableJsonObject(
            control.step_hyperparameters(iteration=0)
        ),
        budget=request_budget(control),
        step_output_contract=gepa_step_output_contract(run_ref),
    )
    with patch(
        "whetstone.optim.gepa.harness_adapter.run_one_gepa_iteration",
        return_value=(
            MagicMock(),
            GepaStepCheckpoint(metric_calls_consumed=1, terminal=False),
        ),
    ):
        output = adapter.invoke(request, ())

    # A continuing step, so nothing has written a terminal transcript yet.
    assert output.proposed_status is StepStatus.CONTINUE
    persisted = output.state_delta[GEPA_SKIPPED_MUTATIONS_KEY]
    assert list(persisted) == [skipped.model_dump(mode="json")]
    assert persisted[0]["exhausted"] is True
