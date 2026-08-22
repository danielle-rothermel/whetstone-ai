from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock, patch

from whetstone.coordination.eval_service import EvalEngineService
from whetstone.coordination.harness_run_controller import OptimRunLaunch
from whetstone.coordination.step_request_builder import StepRequestBuilder
from dr_store.sync import open_sqlite
from whetstone.core.leasing import EffectLeaseAuthority, ReplayPolicy
from whetstone.core.identity import TypedRef
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.adapters import MappingAdapterRegistry
from whetstone.optim.contracts import (
    OptimRun,
    OutputContract,
    StepMode,
    StepStatus,
    step_result_reference,
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
from whetstone.optim.harness import OptimHarness
from whetstone.optim.proposal.proposer import ProposerConfig, prompt_adapter_identity_hash
from whetstone.optim.tools.facade import ToolAdmissionAuthority, ToolCallStore
from whetstone.provider.language_model import PlainPromptAdapter
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    build_toy_experiment,
    toy_template_render_contract,
)


def _toy_gepa_control(*, max_metric_calls: int, store) -> object:
    experiment = build_toy_experiment(num_seeds=1)
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


def test_gepa_harness_e2e_terminalizes(tmp_path) -> None:
    sqlite_path = str(tmp_path / "gepa-e2e.sqlite")
    with open_sqlite(sqlite_path) as store:
        control = _toy_gepa_control(max_metric_calls=2, store=store)
        experiment = build_toy_experiment(num_seeds=1)
        engine = ReferenceEvalRuntimeConfig().build_engine(store)
        eval_service = EvalEngineService(store=store, engine=engine)
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
        registry = MappingAdapterRegistry({GEPA_ADAPTER_KEY: adapter})
        effect_authority = EffectLeaseAuthority.memory()
        harness = OptimHarness(
            store=store,
            adapter_registry=registry,
            tool_store=ToolCallStore(
                store,
                ToolAdmissionAuthority.memory(),
                effect_authority,
            ),
            effect_authority=effect_authority,
            owner_id="gepa-e2e-owner",
            adapter_replay_policy=ReplayPolicy.DURABLE_WORKFLOW,
            lease_duration=timedelta(minutes=5),
            evaluation_service=eval_service,
        )
        run = OptimRun(
            run_id="gepa-e2e-run",
            optimizer_config=control.reference(),
            adapter_key=GEPA_ADAPTER_KEY,
            mode=StepMode.PROPOSAL_ONLY,
            terminal_output_contract=OutputContract(returned_proposal_count=1),
            template_render_contract=toy_template_render_contract(),
            mutation_field=TOY_MUTATION_FIELD,
            reward_policy=experiment.reward_policy,
        )
        launch = OptimRunLaunch(
            run=run,
            initial_candidate=experiment.initial_candidate,
            control=control,
        )
        harness.bind_run(launch.run)
        step_builder = StepRequestBuilder(store=store)
        factory.persist_result.return_value = TypedRef(
            schema_name="whetstone.gepa.result",
            content_hash="a" * 64,
        )
        with patch(
            "whetstone.optim.gepa.harness_adapter.run_one_gepa_iteration",
            side_effect=[
                (detailed, GepaStepCheckpoint(metric_calls_consumed=1, terminal=False)),
                (detailed, GepaStepCheckpoint(metric_calls_consumed=2, terminal=True)),
            ],
        ):
            with patch(
                "whetstone.optim.gepa.harness_adapter.project_gepa_terminal",
                return_value=GepaTerminalResult(
                    best_candidate={
                        "generate": "Answer {prompt} in one short friendly sentence.",
                    },
                    control_identity_hash=control.identity_hash(),
                    artifact_ref=TypedRef(
                        schema_name="whetstone.gepa.result",
                        content_hash="a" * 64,
                    ),
                ),
            ):
                bound = harness.bind_run(launch.run)
                step_request = step_builder.build_first(
                    run=bound,
                    adapter_key=GEPA_ADAPTER_KEY,
                    initial_candidate=launch.initial_candidate,
                    control=control,
                )
                result, result_ref = harness.run_step(step_request)
                assert result.status is StepStatus.CONTINUE
                step_request = step_builder.build_next(
                    prior=result,
                    prior_ref=result_ref,
                    prior_results=(result,),
                    control=control,
                    mutation_field=TOY_MUTATION_FIELD,
                )
                terminal_result, _terminal_ref = harness.run_step(step_request)
        assert terminal_result.status is StepStatus.COMPLETE
        assert len(terminal_result.accepted_candidates) == 1
        assert len(terminal_result.proposed_candidates) == 1
        terminal, terminal_ref = harness.terminalize(
            run=bound,
            step_results=(
                step_result_reference(result),
                step_result_reference(terminal_result),
            ),
        )
        assert len(terminal.proposals) == 1
