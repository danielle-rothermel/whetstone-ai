"""GEPA platform wiring: continuation pools, fan-out, seed-retained completion."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from whetstone.coordination.eval_service import EvalDispatchMode
from whetstone.coordination.step_request_builder import StepRequestBuilder
from whetstone.core.identity import ImmutableJsonObject, TypedRef
from whetstone.eval.metadata import PURPOSE_METADATA_KEY
from whetstone.eval.protocol import EvalRequest
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.experiment.candidate import Candidate, candidate_reference
from whetstone.optim.contracts import (
    IntentOutcome,
    OptimEvalRequest,
    OptimStepResultRef,
    StepStatus,
)
from whetstone.optim.gepa.authorities import (
    CanonicalGepaCandidateAssembler,
    GepaCandidateFieldBinding,
)
from whetstone.optim.gepa.contracts import GepaCandidateComponent
from whetstone.optim.gepa.harness_adapter import (
    GEPA_ADAPTER_KEY,
    GEPA_SKIPPED_MUTATIONS_KEY,
)
from whetstone.optim.gepa.step_engine import GEPA_STATE_KEY
from whetstone.platform.contracts import OptimWorkInput, persist_work_input
from whetstone.platform.step_executor import (
    _deferred_row_count,
    _expand_eval_rows,
    _load_work_state,
    _task_ids_for_intent,
    execute_optim_step_sync,
    execute_run_completion_sync,
)
from whetstone.testing.runtime import (
    TOY_GEPA_COMPONENT,
    build_toy_copro_control,
    build_toy_gepa_adapter,
    build_toy_gepa_control,
    prepare_toy_gepa_run,
    register_toy_runtime,
)
from whetstone.testing.toy.experiment import (
    DEFAULT_TOY_TEMPLATE,
    TOY_MUTATION_FIELD,
    ToyTask,
    build_toy_experiment,
)


def _gepa_runtime(
    store,
    *,
    run_id: str,
    max_metric_calls: int,
    reflection_bodies=None,
    bind_platform_eval_service: bool = False,
):
    experiment = build_toy_experiment(num_seeds=1)
    engine = ReferenceEvalRuntimeConfig().build_engine(store, experiment=experiment)
    control = build_toy_gepa_control(
        engine=engine,
        max_metric_calls=max_metric_calls,
    )
    adapter = build_toy_gepa_adapter(
        store=store,
        engine=engine,
        control=control,
        run_id=run_id,
        initial_candidate=experiment.initial_candidate,
        reflection_bodies=reflection_bodies
        if reflection_bodies is not None
        else (DEFAULT_TOY_TEMPLATE,),
    )
    runtime = register_toy_runtime(
        store=store,
        engine=engine,
        copro_control=build_toy_copro_control(engine=engine),
        extra_adapters={GEPA_ADAPTER_KEY: adapter},
    )
    launch = prepare_toy_gepa_run(
        runtime,
        run_id=run_id,
        control=control,
        experiment=experiment,
    )
    if bind_platform_eval_service:
        adapter.bind_evaluation_service(runtime.eval_service)
    return runtime, launch, adapter, experiment


def _assembled_gepa_eval_candidate(experiment):
    assembler = CanonicalGepaCandidateAssembler(
        base_candidate=candidate_reference(experiment.initial_candidate),
        fields=(
            GepaCandidateFieldBinding(
                component_name=TOY_GEPA_COMPONENT,
                candidate_field=TOY_MUTATION_FIELD,
            ),
        ),
    )
    return assembler.assemble(
        (
            GepaCandidateComponent(
                name=TOY_GEPA_COMPONENT,
                text="Answer {prompt} with a mutated greeting.",
            ),
        )
    )


def _gepa_lineage_request(runtime, launch, experiment):
    bound = runtime.harness.bind_run(launch.run)
    return StepRequestBuilder(store=runtime.store).build_first(
        run=bound,
        adapter_key=GEPA_ADAPTER_KEY,
        initial_candidate=experiment.initial_candidate,
        control=launch.control,
    )


def _gepa_eval_intent(request, candidate, *, request_id: str):
    return OptimEvalRequest(
        optim_run_id=request.run_id,
        optim_step_index=request.step_index,
        eval_request=EvalRequest(
            request_id=request_id,
            candidate=candidate,
        ),
    )


def test_persisted_build_next_matches_in_process_continuation_pools(
    sqlite_store,
) -> None:
    """Platform reconstructs GEPA continuation from the last completed prior."""
    run_id = "gepa-persisted-build-next"
    experiment = build_toy_experiment(num_seeds=1)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store, experiment=experiment
    )
    control = build_toy_gepa_control(engine=engine, max_metric_calls=8)
    adapter = build_toy_gepa_adapter(
        store=sqlite_store,
        engine=engine,
        control=control,
        run_id=run_id,
        initial_candidate=experiment.initial_candidate,
    )
    runtime = register_toy_runtime(
        store=sqlite_store,
        engine=engine,
        copro_control=build_toy_copro_control(engine=engine),
        extra_adapters={GEPA_ADAPTER_KEY: adapter},
    )
    launch = prepare_toy_gepa_run(
        runtime,
        run_id=run_id,
        control=control,
        experiment=experiment,
    )
    bound = runtime.harness.bind_run(launch.run)
    builder = StepRequestBuilder(store=sqlite_store)
    first_request = builder.build_first(
        run=bound,
        adapter_key=GEPA_ADAPTER_KEY,
        initial_candidate=experiment.initial_candidate,
        control=control,
    )
    first, first_ref = runtime.harness.run_step(first_request)
    assert first.status is StepStatus.CONTINUE

    in_process = builder.build_next(
        prior=first,
        prior_ref=first_ref,
        prior_results=(first,),
        control=control,
        mutation_field=TOY_MUTATION_FIELD,
    )
    persisted_prior = OptimStepResultRef(record=first, record_ref=first_ref)
    reconstructed = builder.build_next(
        prior=persisted_prior.record,
        prior_ref=persisted_prior.record_ref,
        prior_results=(persisted_prior.record,),
        control=control,
        mutation_field=TOY_MUTATION_FIELD,
    )
    in_process_pools = dict(in_process.pools)
    reconstructed_pools = dict(reconstructed.pools)
    assert reconstructed_pools[GEPA_STATE_KEY] == in_process_pools[GEPA_STATE_KEY]
    assert (
        reconstructed_pools[GEPA_SKIPPED_MUTATIONS_KEY]
        == in_process_pools[GEPA_SKIPPED_MUTATIONS_KEY]
    )

    resumed = builder.build_next(
        prior=first,
        prior_ref=first_ref,
        prior_results=(first,),
        control=control,
        mutation_field=TOY_MUTATION_FIELD,
        extra_pools={"platform_stage_index": 4},
    )
    resumed_pools = dict(resumed.pools)
    assert resumed_pools[GEPA_STATE_KEY] == in_process_pools[GEPA_STATE_KEY]
    assert resumed_pools["platform_stage_index"] == 4
    assert dict(in_process.pools) != resumed_pools

    authority = adapter._adapter_factory._factory._evaluation_authority
    prefix_hits = len(first.search_evidence)
    second, _ = runtime.harness.run_step(reconstructed)
    replayed = sum(1 for flag in authority.replayed_flags if flag)
    assert prefix_hits
    assert replayed == prefix_hits
    assert second.search_evidence


def test_seed_retained_completes_a_platform_run(sqlite_store) -> None:
    """A GEPA run that keeps the seed still terminalizes the platform loop."""
    run_id = f"gepa-seed-retained-{uuid4().hex[:8]}"
    runtime, launch, _adapter, _experiment = _gepa_runtime(
        sqlite_store,
        run_id=run_id,
        max_metric_calls=1,
        reflection_bodies=(DEFAULT_TOY_TEMPLATE,),
    )
    control = launch.control
    assert control is not None
    runtime.controller.bind_launch(launch)
    work_input = OptimWorkInput(
        run_id=launch.run.run_id,
        controller_identity_hash=runtime.controller.runtime_hash,
        control_identity_hash=control.identity_hash(),
        dispatch_mode=EvalDispatchMode.INLINE,
    )
    current_ref = persist_work_input(runtime.store, work_input)
    completion = execute_optim_step_sync(runtime, input_reference=current_ref)
    while completion.successors:
        current_ref = completion.successors[0].input_reference
        completion = execute_optim_step_sync(
            runtime,
            input_reference=current_ref,
            stage_index=completion.successors[0].stage_index,
        )
    terminal_ref = execute_run_completion_sync(
        runtime,
        input_reference=completion.output_reference,
    )
    from whetstone.optim.contracts import OPTIM_RESULT_SCHEMA, OptimResult
    from dr_store.content_addressing import parse_object_reference

    parsed = parse_object_reference(terminal_ref)
    assert parsed.schema == OPTIM_RESULT_SCHEMA
    result = OptimResult.model_validate(runtime.store.get(parsed))
    assert result.seed_retained is True
    assert not result.proposals
    assert result.step_results
    for step_ref in result.step_results:
        assert step_ref.record.search_evidence


def test_gepa_two_intents_expand_per_intent_task_sets(sqlite_store) -> None:
    """Fan-out uses each GEPA intent's task hashes; COPRO-shaped intents stay full."""
    experiment = build_toy_experiment(
        num_seeds=1,
        internal_tasks=(
            ToyTask(task_id="task-a", prompt_inputs={"prompt": "hello A"}, gold="A"),
            ToyTask(task_id="task-b", prompt_inputs={"prompt": "hello B"}, gold="B"),
            ToyTask(task_id="task-c", prompt_inputs={"prompt": "hello C"}, gold="C"),
        ),
    )
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store, experiment=experiment
    )
    runtime = SimpleNamespace(eval_service=SimpleNamespace(_engine=engine))
    hashes = engine.sampling.task_hashes
    assert len(hashes) == 3

    def make_intent(*, request_id: str, task_hashes: tuple[str, ...] | None):
        return OptimEvalRequest(
            optim_run_id="gepa-fanout",
            optim_step_index=0,
            eval_request=EvalRequest(
                request_id=request_id,
                candidate=experiment.initial_candidate,
                metadata=ImmutableJsonObject(
                    {PURPOSE_METADATA_KEY: "gepa_metric"}
                ),
            ),
            expected_reward_policy_hash=experiment.reward_policy.identity_hash(),
            task_hashes=task_hashes,
        )

    scoped = (
        make_intent(request_id="gepa-a", task_hashes=(hashes[0],)),
        make_intent(request_id="gepa-bc", task_hashes=hashes[1:]),
    )
    assert _deferred_row_count(runtime, scoped) == 3
    rows = _expand_eval_rows(
        runtime,
        scoped,
        deferral_origin_stage_index=0,
        work_state_ref="ws-ref",
    )
    assert [row.task_id for row in rows] == ["task-a", "task-b", "task-c"]
    assert [row.row_ordinal for row in rows] == [0, 1, 2]

    full = make_intent(request_id="copro-full", task_hashes=None)
    assert _task_ids_for_intent(runtime, full) == ("task-a", "task-b", "task-c")
    assert _deferred_row_count(runtime, (full,)) == 3


def test_gepa_platform_deferral_same_step_resume(sqlite_store) -> None:
    """PLATFORM fan-in re-queues the same step and then the run can finish."""
    from whetstone.platform.contracts import (
        STAGE_EVAL_FANIN,
        STAGE_EVAL_ROW,
        STAGE_OPTIM_STEP,
    )
    from whetstone.platform.eval_fanin import (
        build_platform_row_executor,
        execute_eval_fanin_sync,
        execute_eval_row_sync,
    )

    run_id = f"gepa-platform-e2e-{uuid4().hex[:8]}"
    runtime, launch, adapter, _experiment = _gepa_runtime(
        sqlite_store,
        run_id=run_id,
        max_metric_calls=2,
    )
    control = launch.control
    assert control is not None
    runtime.controller.bind_launch(launch)
    work_input = OptimWorkInput(
        run_id=launch.run.run_id,
        controller_identity_hash=runtime.controller.runtime_hash,
        control_identity_hash=control.identity_hash(),
        dispatch_mode=EvalDispatchMode.PLATFORM,
    )
    current_ref = persist_work_input(runtime.store, work_input)
    episodes = 0
    completion = None
    while episodes < 8:
        completion = execute_optim_step_sync(
            runtime,
            input_reference=current_ref,
        )
        row_successors = [
            successor
            for successor in completion.successors
            if successor.stage_key.value == STAGE_EVAL_ROW
        ]
        fanin_successors = [
            successor
            for successor in completion.successors
            if successor.stage_key.value == STAGE_EVAL_FANIN
        ]
        if not row_successors and not fanin_successors:
            assert not completion.successors or (
                completion.successors[0].stage_key.value == STAGE_OPTIM_STEP
            )
            if not completion.successors:
                current_ref = completion.output_reference
                break
            current_ref = completion.successors[0].input_reference
            continue
        episodes += 1
        assert fanin_successors and fanin_successors[0].barrier is True
        platform_executor = build_platform_row_executor(runtime)
        for row_successor in row_successors:
            execute_eval_row_sync(
                runtime,
                input_reference=row_successor.input_reference,
                stage_index=row_successor.stage_index,
                row_executor=platform_executor,
            )
        fanin_completion = execute_eval_fanin_sync(
            runtime,
            input_reference=fanin_successors[0].input_reference,
            stage_index=fanin_successors[0].stage_index,
        )
        assert fanin_completion.successors
        successor = fanin_completion.successors[0]
        assert successor.stage_key.value == STAGE_OPTIM_STEP
        current_ref = successor.input_reference
    assert 1 <= episodes < 8
    assert adapter.invocations > 1
    terminal_ref = execute_run_completion_sync(
        runtime,
        input_reference=current_ref,
    )
    from dr_store.content_addressing import parse_object_reference
    from whetstone.optim.contracts import OPTIM_RESULT_SCHEMA, OptimResult

    parsed = parse_object_reference(terminal_ref)
    assert parsed.schema == OPTIM_RESULT_SCHEMA
    result = OptimResult.model_validate(runtime.store.get(parsed))
    assert result.proposals or result.seed_retained
    assert result.step_results
    for step_ref in result.step_results:
        assert step_ref.record.search_evidence


def test_gepa_search_eval_candidate_passes_lineage_check(sqlite_store) -> None:
    runtime, launch, _adapter, experiment = _gepa_runtime(
        sqlite_store,
        run_id="gepa-lineage-ok",
        max_metric_calls=2,
    )
    request = _gepa_lineage_request(runtime, launch, experiment)
    assembled = _assembled_gepa_eval_candidate(experiment)
    intent = _gepa_eval_intent(
        request,
        assembled.record,
        request_id="gepa-lineage-ok",
    )
    runtime.harness._require_eval_request_on_step(request, intent, {})


def test_gepa_eval_candidate_rejects_alien_base_ref(sqlite_store) -> None:
    runtime, launch, _adapter, experiment = _gepa_runtime(
        sqlite_store,
        run_id="gepa-lineage-base-ref",
        max_metric_calls=2,
    )
    request = _gepa_lineage_request(runtime, launch, experiment)
    assembled = _assembled_gepa_eval_candidate(experiment)
    alien = assembled.record.model_copy(
        update={
            "base_ref": TypedRef(
                schema_name="whetstone.alien_base",
                content_hash="ab" * 32,
            )
        }
    )
    intent = _gepa_eval_intent(
        request,
        alien,
        request_id="gepa-lineage-base-ref",
    )
    with pytest.raises(ValueError, match="assembled from the run"):
        runtime.harness._require_eval_request_on_step(request, intent, {})


def test_gepa_eval_candidate_rejects_extra_payload_key(sqlite_store) -> None:
    runtime, launch, _adapter, experiment = _gepa_runtime(
        sqlite_store,
        run_id="gepa-lineage-extra-key",
        max_metric_calls=2,
    )
    request = _gepa_lineage_request(runtime, launch, experiment)
    assembled = _assembled_gepa_eval_candidate(experiment)
    payload = dict(assembled.record.payload)
    payload["alien"] = "extra"
    alien = Candidate(
        candidate_id=assembled.record.candidate_id,
        base_ref=assembled.record.base_ref,
        payload=payload,
    )
    intent = _gepa_eval_intent(
        request,
        alien,
        request_id="gepa-lineage-extra-key",
    )
    with pytest.raises(ValueError, match="assembled from the run"):
        runtime.harness._require_eval_request_on_step(request, intent, {})


def _run_gepa_platform_to_first_fanin(runtime, launch):
    from whetstone.platform.contracts import STAGE_EVAL_FANIN, STAGE_EVAL_ROW
    from whetstone.platform.eval_fanin import (
        build_platform_row_executor,
        execute_eval_row_sync,
    )

    control = launch.control
    assert control is not None
    runtime.controller.bind_launch(launch)
    work_input = OptimWorkInput(
        run_id=launch.run.run_id,
        controller_identity_hash=runtime.controller.runtime_hash,
        control_identity_hash=control.identity_hash(),
        dispatch_mode=EvalDispatchMode.PLATFORM,
    )
    current_ref = persist_work_input(runtime.store, work_input)
    for _ in range(8):
        completion = execute_optim_step_sync(
            runtime,
            input_reference=current_ref,
        )
        row_successors = [
            successor
            for successor in completion.successors
            if successor.stage_key.value == STAGE_EVAL_ROW
        ]
        fanin_successors = [
            successor
            for successor in completion.successors
            if successor.stage_key.value == STAGE_EVAL_FANIN
        ]
        if row_successors or fanin_successors:
            assert fanin_successors and fanin_successors[0].barrier is True
            platform_executor = build_platform_row_executor(runtime)
            for row_successor in row_successors:
                execute_eval_row_sync(
                    runtime,
                    input_reference=row_successor.input_reference,
                    stage_index=row_successor.stage_index,
                    row_executor=platform_executor,
                )
            return runtime, row_successors, fanin_successors[0]
        if not completion.successors:
            raise AssertionError("GEPA PLATFORM run finished before deferral")
        current_ref = completion.successors[0].input_reference
    raise AssertionError("GEPA PLATFORM run did not defer")


def test_gepa_fanin_retry_is_idempotent(sqlite_store) -> None:
    from whetstone.platform.eval_fanin import execute_eval_fanin_sync

    run_id = f"gepa-fanin-retry-{uuid4().hex[:8]}"
    runtime, launch, adapter, _experiment = _gepa_runtime(
        sqlite_store,
        run_id=run_id,
        max_metric_calls=2,
    )
    runtime, _rows, fanin_successor = _run_gepa_platform_to_first_fanin(
        runtime, launch
    )
    first = execute_eval_fanin_sync(
        runtime,
        input_reference=fanin_successor.input_reference,
        stage_index=fanin_successor.stage_index,
    )
    invocations = adapter.invocations
    second = execute_eval_fanin_sync(
        runtime,
        input_reference=fanin_successor.input_reference,
        stage_index=fanin_successor.stage_index,
    )
    assert second.successors[0].input_reference == first.successors[0].input_reference
    assert second.successors[0].stage_index == first.successors[0].stage_index
    assert adapter.invocations == invocations


def test_gepa_failed_deferred_row_resumes_without_redeferral(sqlite_store) -> None:
    from whetstone.coordination.eval_service import EvalEngineService
    from whetstone.core.roles import EvalRole
    from whetstone.eval.row_slice import RowEvalCompletion
    from whetstone.eval.schema import EvalFailureEvidence
    from whetstone.eval.schema_names import EVAL_FAILURE_SCHEMA
    from whetstone.optim.contracts import IntentResolution
    from whetstone.platform.contracts import STAGE_EVAL_FANIN, STAGE_EVAL_ROW
    from whetstone.platform.eval_fanin import (
        execute_eval_fanin_sync,
        execute_eval_row_sync,
    )

    run_id = f"gepa-failed-row-{uuid4().hex[:8]}"
    runtime, launch, adapter, _experiment = _gepa_runtime(
        sqlite_store,
        run_id=run_id,
        max_metric_calls=1,
    )
    control = launch.control
    assert control is not None
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
    assert row_successors and fanin_successors
    service = runtime.eval_service
    assert isinstance(service, EvalEngineService)

    def failing_executor(*, intent, **kwargs) -> RowEvalCompletion:
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
        return RowEvalCompletion(
            evidence_ref=TypedRef(
                schema_name=failure_ref_obj.schema,
                content_hash=failure_ref_obj.content_hash,
            )
        )

    for row_successor in row_successors:
        execute_eval_row_sync(
            runtime,
            input_reference=row_successor.input_reference,
            stage_index=row_successor.stage_index,
            row_executor=failing_executor,
        )
    fanin_completion = execute_eval_fanin_sync(
        runtime,
        input_reference=fanin_successors[0].input_reference,
        stage_index=fanin_successors[0].stage_index,
    )
    intent = runtime.harness.last_deferred_platform_intents[0]
    bound = runtime.store.resolve(service._key(intent))
    assert bound is not None
    resolution = IntentResolution.model_validate(runtime.store.get(bound))
    assert resolution.outcome is IntentOutcome.FAILED

    resume = execute_optim_step_sync(
        runtime,
        input_reference=fanin_completion.successors[0].input_reference,
        stage_index=fanin_completion.successors[0].stage_index,
    )
    assert not any(
        successor.stage_key.value in {STAGE_EVAL_ROW, STAGE_EVAL_FANIN}
        for successor in resume.successors
    )
    authority = adapter._adapter_factory._factory._evaluation_authority
    assert authority.resolved_intents
    assert all(
        item.outcome is IntentOutcome.FAILED
        for item in authority.resolved_intents
    )


def test_gepa_stale_fanin_does_not_regress_later_head(sqlite_store) -> None:
    from whetstone.platform.contracts import (
        STAGE_EVAL_FANIN,
        STAGE_EVAL_ROW,
        load_deferral_join_input,
    )
    from whetstone.platform.eval_fanin import (
        build_platform_row_executor,
        execute_eval_fanin_sync,
        execute_eval_row_sync,
    )
    from whetstone.platform.work_state_head import resolve_work_state_head

    run_id = f"gepa-stale-fanin-{uuid4().hex[:8]}"
    runtime, launch, _adapter, _experiment = _gepa_runtime(
        sqlite_store,
        run_id=run_id,
        max_metric_calls=2,
    )
    runtime, _rows, fanin_successor = _run_gepa_platform_to_first_fanin(
        runtime, launch
    )
    join_input = load_deferral_join_input(
        runtime.store, fanin_successor.input_reference
    )
    deferred_state = _load_work_state(runtime, join_input.work_state_ref)
    first = execute_eval_fanin_sync(
        runtime,
        input_reference=fanin_successor.input_reference,
        stage_index=fanin_successor.stage_index,
    )
    current_ref = first.successors[0].input_reference
    for _ in range(8):
        completion = execute_optim_step_sync(
            runtime,
            input_reference=current_ref,
        )
        row_successors = [
            successor
            for successor in completion.successors
            if successor.stage_key.value == STAGE_EVAL_ROW
        ]
        fanin_successors = [
            successor
            for successor in completion.successors
            if successor.stage_key.value == STAGE_EVAL_FANIN
        ]
        if row_successors or fanin_successors:
            platform_executor = build_platform_row_executor(runtime)
            for row_successor in row_successors:
                execute_eval_row_sync(
                    runtime,
                    input_reference=row_successor.input_reference,
                    stage_index=row_successor.stage_index,
                    row_executor=platform_executor,
                )
            later = execute_eval_fanin_sync(
                runtime,
                input_reference=fanin_successors[0].input_reference,
                stage_index=fanin_successors[0].stage_index,
            )
            current_ref = later.successors[0].input_reference
            continue
        if completion.successors:
            current_ref = completion.successors[0].input_reference
            continue
        current_ref = completion.output_reference
        break
    head_ref = resolve_work_state_head(
        runtime.store,
        run_id=run_id,
        work_key="",
    )
    assert head_ref is not None
    head = _load_work_state(runtime, head_ref)
    assert head.step_index > deferred_state.step_index
    replayed = execute_eval_fanin_sync(
        runtime,
        input_reference=fanin_successor.input_reference,
        stage_index=fanin_successor.stage_index,
    )
    replayed_state = _load_work_state(
        runtime, replayed.successors[0].input_reference
    )
    assert replayed_state.step_index >= head.step_index
    assert len(replayed_state.step_result_refs) >= len(head.step_result_refs)
