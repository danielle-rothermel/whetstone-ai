from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from whetstone.coordination.eval_service import EvalDispatchMode, EvalEngineService
from whetstone.core.identity import ImmutableJsonObject
from whetstone.coordination.step_request_builder import StepRequestBuilder
from whetstone.eval.protocol import EvalRequest
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.contracts import StepStatus
from whetstone.platform.contracts import OptimWorkInput, persist_work_input
from whetstone.platform.eval_fanin import (
    build_inline_row_executor,
    build_platform_row_executor,
    execute_eval_fanin_sync,
    execute_eval_row_sync,
)
from whetstone.platform.step_executor import (
    STAGE_EVAL_FANIN,
    STAGE_EVAL_ROW,
    execute_optim_step_sync,
)
from whetstone.provider.llm_call import derive_rng_seed, resolve_eval_rng_seed


def _run_platform_deferral_to_fanin(copro_launch):
    runtime, launch = copro_launch
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
    for row_successor in row_successors:
        execute_eval_row_sync(
            runtime,
            input_reference=row_successor.input_reference,
            stage_index=row_successor.stage_index,
            row_executor=build_platform_row_executor(runtime),
        )
    return runtime, fanin_successors[0]


def test_fanin_retry_after_intent_resolved_completes(copro_launch) -> None:
    runtime, fanin_successor = _run_platform_deferral_to_fanin(copro_launch)
    service = runtime.eval_service
    assert isinstance(service, EvalEngineService)

    first = execute_eval_fanin_sync(
        runtime,
        input_reference=fanin_successor.input_reference,
        stage_index=fanin_successor.stage_index,
    )
    intent = runtime.harness.last_deferred_platform_intents[0]  # noqa: SLF001
    assert runtime.store.resolve(service._key(intent)) is not None

    service._store.evict_bindings([service._key(intent)])  # noqa: SLF001
    second = execute_eval_fanin_sync(
        runtime,
        input_reference=fanin_successor.input_reference,
        stage_index=fanin_successor.stage_index,
    )
    assert second.output_reference == first.output_reference
    assert second.successors[0].stage_key.value == "optim_step"


def test_fanin_retry_is_idempotent_after_finalize(copro_launch) -> None:
    runtime, fanin_successor = _run_platform_deferral_to_fanin(copro_launch)
    first = execute_eval_fanin_sync(
        runtime,
        input_reference=fanin_successor.input_reference,
        stage_index=fanin_successor.stage_index,
    )
    second = execute_eval_fanin_sync(
        runtime,
        input_reference=fanin_successor.input_reference,
        stage_index=fanin_successor.stage_index,
    )
    assert second.output_reference == first.output_reference


def test_fanin_raises_when_batch_missing_row_outcomes(copro_launch) -> None:
    runtime, launch = copro_launch
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
    fanin_successor = next(
        successor
        for successor in step_completion.successors
        if successor.stage_key.value == STAGE_EVAL_FANIN
    )
    with pytest.raises(ValueError, match="incomplete|not bound"):
        execute_eval_fanin_sync(
            runtime,
            input_reference=fanin_successor.input_reference,
            stage_index=fanin_successor.stage_index,
        )


def test_eval_row_sync_is_idempotent(copro_launch) -> None:
    runtime, launch = copro_launch
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
    row_successor = next(
        successor
        for successor in step_completion.successors
        if successor.stage_key.value == STAGE_EVAL_ROW
    )
    calls: list[int] = []
    real_executor = build_inline_row_executor(runtime)

    def counting_executor(**kwargs):
        calls.append(1)
        return real_executor(**kwargs)

    first = execute_eval_row_sync(
        runtime,
        input_reference=row_successor.input_reference,
        stage_index=row_successor.stage_index,
        row_executor=counting_executor,
    )
    second = execute_eval_row_sync(
        runtime,
        input_reference=row_successor.input_reference,
        stage_index=row_successor.stage_index,
        row_executor=counting_executor,
    )
    assert first.output_reference == second.output_reference
    assert len(calls) == 1


def test_deferred_handoff_resumes_from_pending_batch(copro_launch) -> None:
    runtime, launch = copro_launch
    control = launch.control
    runtime.controller.bind_launch(launch)
    work_input = OptimWorkInput(
        run_id=launch.run.run_id,
        controller_identity_hash=runtime.controller.runtime_hash,
        control_identity_hash=control.identity_hash(),
        dispatch_mode=EvalDispatchMode.PLATFORM,
    )
    input_reference = persist_work_input(runtime.store, work_input)
    first = execute_optim_step_sync(
        runtime,
        input_reference=input_reference,
        stage_index=0,
    )
    assert any(
        successor.stage_key.value == STAGE_EVAL_ROW for successor in first.successors
    )
    recovered = execute_optim_step_sync(
        runtime,
        input_reference=input_reference,
        stage_index=0,
    )
    assert len(recovered.successors) == len(first.successors)


def test_for_task_ids_preserves_rng_seeds(copro_launch) -> None:
    runtime, _launch = copro_launch
    runtime.controller.bind_launch(_launch)
    engine = runtime.eval_service._engine  # noqa: SLF001
    task_id = engine.sampling.tasks[0].task_id
    task_hash = engine.sampling.task_hashes[0]
    source_rng = dict(engine._sampling.seed_plan.rng_seeds)  # noqa: SLF001
    scoped = engine.for_task_ids((task_id,))
    derived_rng = dict(scoped._sampling.seed_plan.rng_seeds)  # noqa: SLF001
    source_value = source_rng.get(f"{task_hash}#0")
    expected = (
        source_value
        if source_value is not None
        else derive_rng_seed(task_hash, 0)
    )
    assert derived_rng.get(f"{task_hash}#0") == expected


def test_resolve_eval_rng_seed_uses_plan_provenance(sqlite_store) -> None:
    engine = ReferenceEvalRuntimeConfig().build_engine(sqlite_store)
    sampling = engine._sampling  # noqa: SLF001
    task = sampling.tasks[0]
    task_id = task.task_id
    task_hash = sampling.task_set.task_hashes[0]
    provenance_seed = 424242
    seed_plan = sampling.seed_plan.model_copy(
        update={"rng_seeds": {f"{task_hash}#1": provenance_seed}}
    )
    resolved = resolve_eval_rng_seed(
        candidate_id=engine.experiment.initial_candidate.candidate_id,
        task_id=task_id,
        task_hash=task_hash,
        seed_index=1,
        seed_plan=seed_plan,
    )
    assert resolved == provenance_seed
    assert resolved != derive_rng_seed(
        engine.experiment.initial_candidate.candidate_id,
        task_id,
        1,
    )


def test_for_task_seed_uses_source_rng_in_driver(sqlite_store) -> None:
    from dataclasses import replace

    engine = ReferenceEvalRuntimeConfig().build_engine(sqlite_store)
    task_id = engine.sampling.tasks[0].task_id
    task_hash = engine.sampling.task_hashes[0]
    provenance_seed = 8675309
    patched_seed_plan = engine._sampling.seed_plan.model_copy(  # noqa: SLF001
        update={"rng_seeds": {f"{task_hash}#0": provenance_seed}}
    )
    patched_sampling = replace(
        engine._sampling,  # noqa: SLF001
        seed_plan=patched_seed_plan,
    )
    scoped = engine.for_task_seed(task_id, 0)
    scoped._sampling = patched_sampling  # noqa: SLF001
    captured: list[int] = []

    def capture_run_node(*, llm_deps, eval_deps):
        captured.append(llm_deps.rng_seed)
        return MagicMock()

    request = EvalRequest(
        request_id="driver-rng",
        candidate=engine.experiment.initial_candidate,
    )
    with patch(
        "whetstone.eval.drivers.graph_rollout.build_run_node",
        side_effect=capture_run_node,
    ):
        scoped.evaluate_row(request)
    assert captured[0] == provenance_seed


def test_supplemental_reaggregation_is_order_independent(sqlite_store) -> None:
    from unittest.mock import patch

    from whetstone.eval import AggregationOutput, AggregationStatus
    from whetstone.eval.aggregate import AGGREGATE_SCHEMA, Aggregate
    from whetstone.eval.row_slice import RowEvalSlice
    from whetstone.eval.schema import EvalEvidence
    from whetstone.experiment.reward import apply_reward_policy

    engine = ReferenceEvalRuntimeConfig().build_engine(sqlite_store)
    task_ids = tuple(task.task_id for task in engine.sampling.tasks[:2])
    scoped = engine.for_task_ids(task_ids)
    request = EvalRequest(
        request_id="supplemental-reaggregate",
        candidate=engine.experiment.initial_candidate,
    )

    def make_slice(task_id: str, seed_index: int, value: float) -> RowEvalSlice:
        row_engine = engine.for_task_seed(task_id, seed_index)
        row_completion = row_engine.evaluate_row(request)
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
                value=value,
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
        return RowEvalSlice(
            task_id=task_id,
            seed_index=seed_index,
            evidence=evidence,
            supplemental_aggregate_refs=(supplemental_ref,),
        )

    slice_a = make_slice(task_ids[0], 0, 0.2)
    slice_b = make_slice(task_ids[1], 0, 0.8)
    with patch(
        "whetstone.eval.runtime_engine.apply_reward_policy",
        wraps=apply_reward_policy,
    ) as reward_policy:
        scoped.assemble_from_row_slices(request, row_slices=(slice_a, slice_b))
        first_value = reward_policy.call_args.kwargs["aggregates"]["supplemental"]
        reward_policy.reset_mock()
        scoped.assemble_from_row_slices(request, row_slices=(slice_b, slice_a))
        second_value = reward_policy.call_args.kwargs["aggregates"]["supplemental"]
    assert first_value == pytest.approx(0.5)
    assert second_value == pytest.approx(0.5)


def test_gepa_build_next_zero_budget_binds_the_step_contract(tmp_path) -> None:
    from tests.test_gepa_harness_adapter import _toy_gepa_control
    from dr_store.sync import open_sqlite
    from whetstone.core.identity import TypedRef
    from whetstone.optim.contracts import (
        BudgetState,
        OptimRun,
        OptimStepResult,
        OutputContract,
        StepMode,
        optimization_run_reference,
        step_request_reference,
        step_result_reference,
    )
    from whetstone.optim.gepa.harness_adapter import GEPA_ADAPTER_KEY
    from whetstone.optim.gepa.step_contract import gepa_step_output_contract
    from whetstone.experiment.candidate import Candidate, candidate_reference
    from whetstone.optim.gepa.step_engine import GEPA_STATE_KEY
    from whetstone.testing.toy.experiment import (
        TOY_MUTATION_FIELD,
        build_toy_experiment,
        toy_template_render_contract,
    )

    control = _toy_gepa_control(
        max_metric_calls=0,
        sqlite_path=str(tmp_path / "gepa-next-zero.sqlite"),
    )
    experiment = build_toy_experiment(num_seeds=1)
    run = OptimRun(
        run_id="gepa-next-zero",
        optimizer_config=control.reference(),
        adapter_key=GEPA_ADAPTER_KEY,
        mode=StepMode.PROPOSAL_ONLY,
        terminal_output_contract=OutputContract(returned_proposal_count=1),
        template_render_contract=toy_template_render_contract(),
        mutation_field=TOY_MUTATION_FIELD,
        reward_policy=experiment.reward_policy,
    )
    run_ref = optimization_run_reference(run)
    with open_sqlite(str(tmp_path / "builder.sqlite")) as store:
        builder = StepRequestBuilder(store=store)
        first = builder.build_first(
            run=run_ref,
            adapter_key=GEPA_ADAPTER_KEY,
            initial_candidate=experiment.initial_candidate,
            control=control,
        )
        state_ref_obj, _ = store.put(
            "whetstone.optim_step_state",
            {GEPA_STATE_KEY: {"metric_calls_consumed": 0, "terminal": False}},
        )
        state_ref = TypedRef(
            schema_name=state_ref_obj.schema,
            content_hash=state_ref_obj.content_hash,
        )
        base_ref = candidate_reference(experiment.initial_candidate)
        mutated_template = (
            f"{experiment.initial_candidate.payload[TOY_MUTATION_FIELD]} v2"
        )
        proposed_ref = candidate_reference(
            Candidate(
                candidate_id="gepa-terminal-proposal",
                base_ref=base_ref.record_ref,
                payload={
                    **experiment.initial_candidate.payload,
                    TOY_MUTATION_FIELD: mutated_template,
                },
            )
        )
        prior = OptimStepResult(
            request=step_request_reference(first),
            proposed_candidates=(proposed_ref,),
            accepted_candidates=(proposed_ref,),
            status=StepStatus.COMPLETE,
            state_ref=state_ref,
            budget=BudgetState(
                remaining=ImmutableJsonObject({"metric_calls": 0}),
            ),
        )
        prior_ref = step_result_reference(prior).record_ref
        next_request = builder.build_next(
            prior=prior,
            prior_ref=prior_ref,
            prior_results=(prior,),
            control=control,
            mutation_field=TOY_MUTATION_FIELD,
        )
    assert next_request.step_output_contract == gepa_step_output_contract(
        run_ref
    )
    assert next_request.step_output_contract.honors_terminal(
        run.terminal_output_contract
    )


def test_preemptible_retry_through_deferral_fanin_resume(copro_launch) -> None:
    runtime, fanin_successor = _run_platform_deferral_to_fanin(copro_launch)
    fanin_completion = execute_eval_fanin_sync(
        runtime,
        input_reference=fanin_successor.input_reference,
        stage_index=fanin_successor.stage_index,
    )
    retry_fanin = execute_eval_fanin_sync(
        runtime,
        input_reference=fanin_successor.input_reference,
        stage_index=fanin_successor.stage_index,
    )
    assert retry_fanin.output_reference == fanin_completion.output_reference
    resume = execute_optim_step_sync(
        runtime,
        input_reference=fanin_completion.successors[0].input_reference,
        stage_index=fanin_completion.successors[0].stage_index,
    )
    assert resume.successors == () or all(
        successor.stage_key.value != STAGE_EVAL_ROW for successor in resume.successors
    )
