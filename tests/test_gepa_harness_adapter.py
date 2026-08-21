from __future__ import annotations

from unittest.mock import MagicMock, patch

from dr_store.sync import open_sqlite
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.experiment.candidate import Candidate
from whetstone.optim.adapters import AdapterOutput
from whetstone.core.identity import ImmutableJsonObject, TypedRef
from whetstone.optim.contracts import (
    BudgetState,
    OptimRun,
    OptimStepRequest,
    OutputContract,
    StepKind,
    StepMode,
    StepStatus,
    optimization_run_reference,
)
from whetstone.optim.gepa.adapter import GepaTerminalResult
from whetstone.optim.gepa.control import configure_gepa
from whetstone.optim.gepa.engine import GepaDetailedResult
from whetstone.optim.gepa.harness_adapter import (
    GEPA_ADAPTER_KEY,
    GepaHarnessAdapter,
    GepaHarnessAdapterFactory,
)
from whetstone.optim.gepa.step_engine import GepaStepCheckpoint
from whetstone.optim.proposal.proposer import ProposerConfig, prompt_adapter_identity_hash
from whetstone.provider.language_model import PlainPromptAdapter
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    build_toy_experiment,
    toy_template_render_contract,
)


def _toy_gepa_control(*, max_metric_calls: int = 2, sqlite_path: str):
    experiment = build_toy_experiment(num_seeds=1)
    with open_sqlite(sqlite_path) as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(store)
        task_hashes = engine.sampling.task_hashes[:1]
        prompt_adapter = PlainPromptAdapter()
        return configure_gepa(
            reflection_model=ProposerConfig(
                provider_call_config=engine.provider_execution_policy_ref,
            ),
            metric=engine.eval_config_ref,
            reward_policy_hash=experiment.reward_policy.identity_hash(),
            evaluation_execution_policy_hash=engine.execution_policy_identity_hash(),
            proposal_execution_policy_hash=engine.execution_policy_identity_hash(),
            proposal_prompt_adapter_identity_hash=prompt_adapter_identity_hash(
                prompt_adapter
            ),
            proposal_durability_policy_identity_hash="c" * 64,
            task_model_identity_hash=engine.task_model_identity_hash(),
            prompt_format_identity_hash="d" * 64,
            prompt_binding_identity_hash="e" * 64,
            trainset_task_hashes=task_hashes,
            valset_task_hashes=None,
            component_names=("generate",),
            num_predictors=1,
            max_metric_calls=max_metric_calls,
        )


def test_gepa_control_reference_and_step_hyperparameters(tmp_path) -> None:
    control = _toy_gepa_control(
        max_metric_calls=2,
        sqlite_path=str(tmp_path / "gepa.sqlite"),
    )
    ref = control.reference()
    assert ref.record_hash == control.identity_hash()
    hyper = control.step_hyperparameters(iteration=1)
    assert hyper["round_index"] == 1
    assert hyper["max_metric_calls"] == 2


def test_gepa_harness_adapter_two_iterations(tmp_path) -> None:
    control = _toy_gepa_control(
        max_metric_calls=2,
        sqlite_path=str(tmp_path / "gepa-harness.sqlite"),
    )
    experiment = build_toy_experiment(num_seeds=1)
    run = OptimRun(
        run_id="gepa-run",
        optimizer_config=control.reference(),
        adapter_key=GEPA_ADAPTER_KEY,
        mode=StepMode.PROPOSAL_ONLY,
        terminal_output_contract=OutputContract(returned_proposal_count=1),
        template_render_contract=toy_template_render_contract(),
        mutation_field=TOY_MUTATION_FIELD,
        reward_policy=experiment.reward_policy,
    )
    run_ref = optimization_run_reference(run)
    candidate = experiment.initial_candidate
    factory = MagicMock()
    factory.create.return_value = MagicMock()
    detailed = GepaDetailedResult(
        candidates=({"generate": "hello"},),
        parents=((None,),),
        val_aggregate_scores=(1.0,),
        val_subscores=({"task": 1.0},),
        per_val_instance_best_candidates={"task": (0,)},
        discovery_eval_counts=(1,),
        seed=0,
        best_idx=0,
        control_identity_hash=control.identity_hash(),
    )
    adapter = GepaHarnessAdapter(
        control=control,
        seed_candidate={"generate": "seed"},
        trainset=(),
        valset=None,
        adapter_factory=GepaHarnessAdapterFactory(factory=factory),
    )
    request0 = OptimStepRequest(
        run=run_ref,
        step_id="gepa-run:gepa:0",
        kind=StepKind.PROPOSAL,
        step_index=0,
        candidates=(candidate,),
        hyperparameters=ImmutableJsonObject(
            control.step_hyperparameters(iteration=0)
        ),
        budget=BudgetState(
            remaining=ImmutableJsonObject({"metric_calls": 2}),
        ),
        step_output_contract=OutputContract(returned_proposal_count=0),
    )
    with patch(
        "whetstone.optim.gepa.harness_adapter.run_one_gepa_iteration",
        side_effect=[
            (detailed, GepaStepCheckpoint(metric_calls_consumed=1, terminal=False)),
            (detailed, GepaStepCheckpoint(metric_calls_consumed=2, terminal=True)),
        ],
    ) as run_iteration:
        output0 = adapter.invoke(request0, ())
        assert output0.proposed_status is StepStatus.CONTINUE
        request1 = OptimStepRequest(
            run=run_ref,
            step_id="gepa-run:gepa:1",
            kind=StepKind.PROPOSAL,
            step_index=1,
            prior_step_result_ref=TypedRef(
                schema_name="whetstone.optim_step_result",
                content_hash="c" * 64,
            ),
            candidates=(candidate,),
            pools=ImmutableJsonObject(
                {"gepa_checkpoint": {"metric_calls_consumed": 1, "terminal": False}}
            ),
            hyperparameters=ImmutableJsonObject(
                control.step_hyperparameters(iteration=1)
            ),
            budget=BudgetState(
                remaining=ImmutableJsonObject({"metric_calls": 1}),
            ),
            step_output_contract=OutputContract(returned_proposal_count=1),
        )
        factory.persist_result.return_value = TypedRef(
            schema_name="whetstone.gepa.result",
            content_hash="a" * 64,
        )
        with patch(
            "whetstone.optim.gepa.harness_adapter.project_gepa_terminal",
            return_value=GepaTerminalResult(
                best_candidate={"generate": "hello"},
                control_identity_hash=control.identity_hash(),
                artifact_ref=TypedRef(
                    schema_name="whetstone.gepa.result",
                    content_hash="a" * 64,
                ),
            ),
        ):
            output1 = adapter.invoke(request1, ())
        assert output1.proposed_status is StepStatus.COMPLETE
        assert len(output1.accepted_candidates) == 1
        assert run_iteration.call_count == 2


def test_gepa_harness_adapter_terminal_assembles_all_components(tmp_path) -> None:
    control = _toy_gepa_control(
        max_metric_calls=1,
        sqlite_path=str(tmp_path / "gepa-multi.sqlite"),
    )
    control = control.model_copy(
        update={
            "component_names": ("generate", "critique"),
            "num_predictors": 2,
        }
    )
    experiment = build_toy_experiment(num_seeds=1)
    run = OptimRun(
        run_id="gepa-multi-run",
        optimizer_config=control.reference(),
        adapter_key=GEPA_ADAPTER_KEY,
        mode=StepMode.PROPOSAL_ONLY,
        terminal_output_contract=OutputContract(returned_proposal_count=1),
        template_render_contract=toy_template_render_contract(),
        mutation_field=TOY_MUTATION_FIELD,
        reward_policy=experiment.reward_policy,
    )
    run_ref = optimization_run_reference(run)
    candidate = experiment.initial_candidate
    factory = MagicMock()
    factory.create.return_value = MagicMock()
    detailed = GepaDetailedResult(
        candidates=({"generate": "hello", "critique": "be concise"},),
        parents=((None, None),),
        val_aggregate_scores=(1.0,),
        val_subscores=({"task": 1.0},),
        per_val_instance_best_candidates={"task": (0,)},
        discovery_eval_counts=(1,),
        seed=0,
        best_idx=0,
        control_identity_hash=control.identity_hash(),
    )
    adapter = GepaHarnessAdapter(
        control=control,
        seed_candidate={"generate": "seed", "critique": "seed critique"},
        trainset=(),
        valset=None,
        adapter_factory=GepaHarnessAdapterFactory(factory=factory),
    )
    request = OptimStepRequest(
        run=run_ref,
        step_id="gepa-multi-run:gepa:0",
        kind=StepKind.PROPOSAL,
        step_index=0,
        candidates=(candidate,),
        hyperparameters=ImmutableJsonObject(
            control.step_hyperparameters(iteration=0)
        ),
        budget=BudgetState(
            remaining=ImmutableJsonObject({"metric_calls": 1}),
        ),
        step_output_contract=OutputContract(returned_proposal_count=1),
    )
    factory.persist_result.return_value = TypedRef(
        schema_name="whetstone.gepa.result",
        content_hash="a" * 64,
    )
    with patch(
        "whetstone.optim.gepa.harness_adapter.run_one_gepa_iteration",
        return_value=(
            detailed,
            GepaStepCheckpoint(metric_calls_consumed=1, terminal=True),
        ),
    ):
        with patch(
            "whetstone.optim.gepa.harness_adapter.project_gepa_terminal",
            return_value=GepaTerminalResult(
                best_candidate={"generate": "hello", "critique": "be concise"},
                control_identity_hash=control.identity_hash(),
                artifact_ref=TypedRef(
                    schema_name="whetstone.gepa.result",
                    content_hash="a" * 64,
                ),
            ),
        ):
            output = adapter.invoke(request, ())
    assert output.proposed_status is StepStatus.COMPLETE
    payload = output.accepted_candidates[0].payload
    assert payload["generate"] == "hello"
    assert payload["critique"] == "be concise"
