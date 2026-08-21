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
from whetstone.core.identity import ImmutableJsonObject, TypedRef
from whetstone.optim.adapters import AdapterOutput
from whetstone.optim.contracts import (
    OptimRun,
    OptimStepRequest,
    OutputContract,
    StepKind,
    StepMode,
    StepStatus,
    optimization_run_reference,
)
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


def test_distinct_bases_constrains_proposed_not_accepted() -> None:
    contract = OutputContract(
        returned_proposal_count=1,
        require_distinct_bases=True,
    )
    assert contract.accepted_count_for(StepStatus.COMPLETE) == 1
    assert contract.accepted_count_for(StepStatus.FAILED) == 0


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
