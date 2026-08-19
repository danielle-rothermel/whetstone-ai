from __future__ import annotations

from unittest.mock import MagicMock

from whetstone.core.roles import EvalRole
from whetstone.core.identity import TypedRef
from whetstone.eval.protocol import EvalRequest
from whetstone.eval.row_slice import RowEvalCompletion
from whetstone.eval.schema import EvalFailureEvidence
from whetstone.eval.schema_names import EVAL_FAILURE_SCHEMA
from whetstone.experiment.candidate import candidate_reference
from whetstone.optim.contracts import IntentOutcome, IntentResolution, ResolutionClass, ResolutionDetail
from whetstone.platform.eval_fanin import (
    build_platform_row_executor,
    execute_eval_fanin_sync,
    execute_eval_row_sync,
)
from whetstone.platform.step_executor import STAGE_EVAL_FANIN, STAGE_EVAL_ROW, execute_optim_step_sync
from whetstone.provider.llm_call import derive_rng_seed


def test_gepa_zero_budget_uses_terminal_contract_and_zero_delta(tmp_path) -> None:
    from tests.test_gepa_harness_adapter import _toy_gepa_control
    from whetstone.coordination.step_request_builder import StepRequestBuilder
    from whetstone.optim.contracts import (
        OptimRun,
        OutputContract,
        StepMode,
        optimization_run_reference,
    )
    from whetstone.optim.gepa.harness_adapter import GEPA_ADAPTER_KEY
    from whetstone.optim.gepa.step_engine import GepaStepCheckpoint, run_one_gepa_iteration
    from whetstone.testing.toy.experiment import (
        TOY_MUTATION_FIELD,
        build_toy_experiment,
        toy_template_render_contract,
    )

    control = _toy_gepa_control(
        max_metric_calls=0,
        sqlite_path=str(tmp_path / "gepa-zero-contract.sqlite"),
    )
    experiment = build_toy_experiment(num_seeds=1)
    run = OptimRun(
        run_id="gepa-zero",
        optimizer_config=control.reference(),
        adapter_key=GEPA_ADAPTER_KEY,
        mode=StepMode.PROPOSAL_ONLY,
        terminal_output_contract=OutputContract(returned_proposal_count=1),
        template_render_contract=toy_template_render_contract(),
        mutation_field=TOY_MUTATION_FIELD,
        reward_policy=experiment.reward_policy,
    )
    run_ref = optimization_run_reference(run)
    request = StepRequestBuilder(store=MagicMock()).build_first(
        run=run_ref,
        adapter_key=GEPA_ADAPTER_KEY,
        initial_candidate=experiment.initial_candidate,
        control=control,
    )
    assert request.step_output_contract == run.terminal_output_contract

    _, checkpoint = run_one_gepa_iteration(
        control=control,
        seed_candidate={"generate": "seed"},
        trainset=(),
        valset=None,
        adapter=MagicMock(),
        checkpoint=GepaStepCheckpoint(),
    )
    assert checkpoint.budget_delta.consumed["metric_calls"] == 0


def test_for_task_seed_preserves_source_rng_seed(copro_launch) -> None:
    runtime, launch = copro_launch
    runtime.controller.bind_launch(launch)
    engine = runtime.eval_service._engine  # noqa: SLF001
    task_id = engine.sampling.tasks[0].task_id
    task_hash = engine.sampling.task_hashes[0]
    source_rng = dict(engine._sampling.seed_plan.rng_seeds)  # noqa: SLF001
    scoped = engine.for_task_seed(task_id, 0)
    derived_rng = dict(scoped._sampling.seed_plan.rng_seeds)  # noqa: SLF001
    source_value = source_rng.get(f"{task_hash}#0")
    expected = (
        source_value
        if source_value is not None
        else derive_rng_seed(task_hash, 0)
    )
    assert derived_rng.get(f"{task_hash}#0") == expected


def test_fanin_binds_failed_when_row_returns_failure_evidence(copro_launch) -> None:
    runtime, launch = copro_launch
    from whetstone.coordination.eval_service import EvalDispatchMode, EvalEngineService
    from whetstone.platform.contracts import OptimWorkInput, persist_work_input

    control = launch.control
    runtime.controller.bind_launch(launch)
    work_input = OptimWorkInput(
        run_id=launch.run.run_id,
        controller_identity_hash=runtime.controller.runtime_hash,
        control_identity_hash=control.identity_hash(),
        dispatch_mode=EvalDispatchMode.PLATFORM,
    )
    input_reference = persist_work_input(runtime.store, work_input)
    step_completion = execute_optim_step_sync(
        runtime,
        input_reference=input_reference,
        stage_index=0,
    )
    row_successors = [
        successor
        for successor in step_completion.successors
        if successor.stage_key.value == STAGE_EVAL_ROW
    ]
    fanin_successors = [
        successor
        for successor in step_completion.successors
        if successor.stage_key.value == STAGE_EVAL_FANIN
    ]
    service = runtime.eval_service
    assert isinstance(service, EvalEngineService)
    failure_ref: TypedRef | None = None

    def failing_executor(*, intent, **kwargs) -> RowEvalCompletion:
        nonlocal failure_ref
        _ = kwargs
        failure = EvalFailureEvidence(
            candidate=candidate_reference(intent.eval_request.candidate),
            eval_config_ref=service._engine.eval_config_ref,  # noqa: SLF001
            eval_role=EvalRole.INTERNAL,
            provider_execution_policy_ref=service._engine.provider_execution_policy_ref,  # noqa: SLF001
            metadata=intent.eval_request.metadata,
            exception_type="RuntimeError",
            message="row failed",
        )
        failure_ref_obj, _ = runtime.store.put(
            EVAL_FAILURE_SCHEMA, failure.record_content()
        )
        failure_ref = TypedRef(
            schema_name=failure_ref_obj.schema,
            content_hash=failure_ref_obj.content_hash,
        )
        return RowEvalCompletion(evidence_ref=failure_ref)

    for row_successor in row_successors:
        execute_eval_row_sync(
            runtime,
            input_reference=row_successor.input_reference,
            stage_index=row_successor.stage_index,
            row_executor=failing_executor,
        )

    execute_eval_fanin_sync(
        runtime,
        input_reference=fanin_successors[0].input_reference,
        stage_index=fanin_successors[0].stage_index,
    )
    intent = runtime.harness.last_deferred_platform_intents[0]  # noqa: SLF001
    bound = runtime.store.resolve(service._key(intent))
    assert bound is not None
    resolution = IntentResolution.model_validate(runtime.store.get(bound))
    assert resolution.outcome is IntentOutcome.FAILED


def test_fanin_binds_rejected_when_row_returns_rejection(copro_launch) -> None:
    runtime, launch = copro_launch
    from whetstone.coordination.eval_service import EvalDispatchMode, EvalEngineService
    from whetstone.platform.contracts import OptimWorkInput, persist_work_input

    control = launch.control
    runtime.controller.bind_launch(launch)
    work_input = OptimWorkInput(
        run_id=launch.run.run_id,
        controller_identity_hash=runtime.controller.runtime_hash,
        control_identity_hash=control.identity_hash(),
        dispatch_mode=EvalDispatchMode.PLATFORM,
    )
    input_reference = persist_work_input(runtime.store, work_input)
    step_completion = execute_optim_step_sync(
        runtime,
        input_reference=input_reference,
        stage_index=0,
    )
    row_successors = [
        successor
        for successor in step_completion.successors
        if successor.stage_key.value == STAGE_EVAL_ROW
    ]
    fanin_successors = [
        successor
        for successor in step_completion.successors
        if successor.stage_key.value == STAGE_EVAL_FANIN
    ]
    service = runtime.eval_service
    assert isinstance(service, EvalEngineService)
    rejected = ResolutionDetail(
        classification=ResolutionClass.VALIDATION,
        message="candidate rejected",
    )

    def rejecting_executor(**kwargs) -> RowEvalCompletion:
        _ = kwargs
        return RowEvalCompletion(rejected_detail=rejected)

    for row_successor in row_successors:
        execute_eval_row_sync(
            runtime,
            input_reference=row_successor.input_reference,
            stage_index=row_successor.stage_index,
            row_executor=rejecting_executor,
        )

    execute_eval_fanin_sync(
        runtime,
        input_reference=fanin_successors[0].input_reference,
        stage_index=fanin_successors[0].stage_index,
    )
    intent = runtime.harness.last_deferred_platform_intents[0]  # noqa: SLF001
    bound = runtime.store.resolve(service._key(intent))
    assert bound is not None
    resolution = IntentResolution.model_validate(runtime.store.get(bound))
    assert resolution.outcome is IntentOutcome.REJECTED


def test_platform_row_executor_returns_completion_not_raise_on_reject(
    copro_launch, monkeypatch
) -> None:
    runtime, launch = copro_launch
    runtime.controller.bind_launch(launch)
    scoped_engine = MagicMock()
    scoped_engine.evaluate_row.return_value = RowEvalCompletion(
        rejected_detail=ResolutionDetail(
            classification=ResolutionClass.VALIDATION,
            message="rejected",
        )
    )
    monkeypatch.setattr(
        runtime.eval_service._engine,
        "for_task_seed",
        lambda task_id, seed_index: scoped_engine,
    )
    executor = build_platform_row_executor(runtime)
    from whetstone.testing.toy.experiment import build_toy_experiment
    from whetstone.optim.contracts import OptimEvalRequest

    experiment = build_toy_experiment(num_seeds=2)
    intent = OptimEvalRequest(
        optim_run_id=launch.run.run_id,
        optim_step_index=0,
        eval_request=EvalRequest(
            request_id="row-eval",
            candidate=experiment.initial_candidate,
        ),
        expected_reward_policy_hash=experiment.reward_policy.identity_hash(),
    )
    completion = executor(intent=intent, task_id="task-a", seed_index=1)
    assert completion is not None
    assert completion.rejected_detail is not None
    scoped_engine.evaluate_row.assert_called_once()


def test_assembly_includes_supplemental_aggregates_from_row_refs(
    sqlite_store,
) -> None:
    from unittest.mock import patch

    from whetstone.eval import AggregationOutput, AggregationStatus
    from whetstone.eval.aggregate import AGGREGATE_SCHEMA, Aggregate
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
    from whetstone.eval.row_slice import RowEvalSlice
    from whetstone.eval.schema import EvalEvidence
    from whetstone.experiment.reward import apply_reward_policy

    engine = ReferenceEvalRuntimeConfig().build_engine(sqlite_store)
    task_id = engine.sampling.tasks[0].task_id
    scoped = engine.for_task_seed(task_id, 0)
    request = EvalRequest(
        request_id="assembly-supplemental",
        candidate=engine.experiment.initial_candidate,
    )
    row_completion = scoped.evaluate_row(request)
    assert row_completion.evidence_ref is not None
    evidence = EvalEvidence.model_validate(
        sqlite_store.get(row_completion.evidence_ref.reference)
    )
    supplemental = Aggregate(
        name="supplemental",
        graph_hash=engine.experiment.rollout_graph.graph_hash,
        eval_config_hash=scoped.eval_config_ref.config_hash,
        task_count=1,
        num_seeds=1,
        aggregation_output=AggregationOutput(
            value=0.25,
            status=AggregationStatus.OK,
            count_total=1,
            count_applicable=1,
            count_present=1,
        ),
        rows_present=1,
        rows_missing=0,
        rows_failed=0,
        rows_invalid=0,
    )
    supplemental_ref = scoped._put(  # noqa: SLF001
        AGGREGATE_SCHEMA,
        supplemental.record_content(),
    )
    row_slice = RowEvalSlice(
        task_id=task_id,
        seed_index=0,
        evidence=evidence,
        supplemental_aggregate_refs=(supplemental_ref,),
    )
    with patch(
        "whetstone.eval.runtime_engine.apply_reward_policy",
        wraps=apply_reward_policy,
    ) as reward_policy:
        scoped.assemble_from_row_slices(request, row_slices=(row_slice,))
    reward_policy.assert_called_once()
    assert reward_policy.call_args.kwargs["aggregates"]["supplemental"] == 0.25
