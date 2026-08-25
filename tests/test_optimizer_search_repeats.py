"""In-search evaluation at ``num_seeds > 1`` for COPRO, MIPROv2, and GEPA.

The Step 10 protocol pre-registers ``K_REPEAT`` for *every* evaluation, the
in-search ones included, so each optimizer must evaluate each task on
``num_seeds`` repeats and consume the repeat-mean. These tests pin the two
halves of that contract:

* the **planned row matrix** is ``tasks x num_seeds`` -- the repeats are
  actually paid for, not silently collapsed to one row per task;
* the **score the search consumes** is the mean over those repeats, which is
  the canonical reduction (``whetstone.eval.drivers.eval_result.per_task_score``,
  surfaced as ``EvalEvidence.per_task_values``) rather than a second one.

Before repeats were plumbed, MIPROv2 raised ``engine sampling repeats (2) do
not match the requested num_seeds (1)`` and GEPA raised ``GEPA evaluation
engine must use a single-repeat plan``.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from whetstone.coordination.harness_run_controller import RunRequest
from whetstone.core.identity import ImmutableJsonObject
from whetstone.eval.aggregate import (
    RowValue,
    TaskRows,
    aggregation_definition,
)
from whetstone.eval.drivers.eval_result import per_task_count, per_task_score
from whetstone.eval.metadata import PURPOSE_METADATA_KEY
from whetstone.eval.protocol import EvalRequest
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.eval.schema import EvalEvidence
from whetstone.optim.contracts import OptimEvalRequest
from whetstone.optim.miprov2.adapter import (
    MIPROV2_ADAPTER_KEY,
    MIPROV2_STATE_KEY,
)
from whetstone.optim.miprov2.runtime import Miprov2State
from whetstone.platform.step_executor import (
    _deferred_row_count,
    _expand_eval_rows,
)
from whetstone.testing.runtime import (
    build_miprov2_adapter,
    build_toy_copro_control,
    prepare_toy_copro_run,
    prepare_toy_miprov2_run,
    register_toy_runtime,
)
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    ToyTask,
    build_toy_experiment,
)
from whetstone.testing.toy.miprov2 import build_toy_miprov2_control

REPEATS = 2


def _three_task_experiment(*, num_seeds: int):
    """A toy experiment whose internal split is three distinct tasks."""

    return build_toy_experiment(
        num_seeds=num_seeds,
        internal_tasks=(
            ToyTask(
                task_id="task-a",
                prompt_inputs={"prompt": "hello A"},
                gold="A",
            ),
            ToyTask(
                task_id="task-b",
                prompt_inputs={"prompt": "hello B"},
                gold="B",
            ),
            ToyTask(
                task_id="task-c",
                prompt_inputs={"prompt": "hello C"},
                gold="C",
            ),
        ),
        official_tasks=(
            ToyTask(
                task_id="task-o",
                prompt_inputs={"prompt": "hello O"},
                gold="O",
            ),
        ),
    )


def _step_results(store, run_id: str):
    """Every persisted Step result of ``run_id``, in step order."""

    from whetstone.optim.contracts import OptimStepResult
    from whetstone.optim.harness import OptimHarness

    results = []
    index = 0
    while True:
        key = OptimHarness._result_binding_key(run_id, index)  # noqa: SLF001
        bound = store.resolve(key)
        if bound is None:
            return tuple(results)
        results.append(OptimStepResult.model_validate(store.get(bound)))
        index += 1


def _evidence_records(store, run_id: str) -> tuple[EvalEvidence, ...]:
    """Evaluation Evidence for every evaluation the run paid for.

    COPRO and MIPROv2 evaluate through harness Intents, so their evidence
    lands in ``resolved_intents``. GEPA's upstream search evaluates
    intermediate candidates the run never proposes, so its evidence lands in
    ``search_evidence`` instead. Both are in-search evaluations and both must
    honour the repeat count.
    """

    records = []
    for result in _step_results(store, run_id):
        entries = (*result.resolved_intents, *result.search_evidence)
        for entry in entries:
            if entry.eval_result_ref is None:
                continue
            records.append(
                EvalEvidence.model_validate(
                    store.get(entry.eval_result_ref.reference)
                )
            )
    return tuple(records)


# --- the planned matrix is tasks x repeats --------------------------------


def test_planned_rows_are_tasks_times_repeats(sqlite_store) -> None:
    """Fan-out plans one row per (task, repeat) at the split's repeat count."""

    experiment = _three_task_experiment(num_seeds=REPEATS)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store, experiment=experiment
    )
    runtime = SimpleNamespace(eval_service=SimpleNamespace(_engine=engine))
    hashes = engine.sampling.task_hashes
    assert len(hashes) == 3

    intent = OptimEvalRequest(
        optim_run_id="repeats-fanout",
        optim_step_index=0,
        eval_request=EvalRequest(
            request_id="repeats-all",
            candidate=experiment.initial_candidate,
            metadata=ImmutableJsonObject(
                {PURPOSE_METADATA_KEY: "gepa_metric"}
            ),
        ),
        expected_reward_policy_hash=(
            experiment.reward_policy.identity_hash()
        ),
        task_hashes=hashes,
    )

    assert _deferred_row_count(runtime, (intent,)) == 3 * REPEATS
    rows = _expand_eval_rows(
        runtime,
        (intent,),
        deferral_origin_stage_index=0,
        work_state_ref="ws-ref",
    )
    assert len(rows) == 3 * REPEATS
    # Rows are ordered task-major: every task carries each repeat exactly once.
    assert [row.task_id for row in rows] == [
        "task-a",
        "task-a",
        "task-b",
        "task-b",
        "task-c",
        "task-c",
    ]
    assert [row.seed_index for row in rows] == [0, 1, 0, 1, 0, 1]


# --- the reduction is the repeat mean -------------------------------------


def _aggregation_config(missing_data: str):
    """A mean aggregation config under the named missing-row policy."""

    return aggregation_definition("test.per_task.aggregation").materialize(
        {"reduction": "mean", "missing_data": missing_data}
    )


SKIP_CONFIG = _aggregation_config("skip")
PROPAGATE_CONFIG = _aggregation_config("propagate")


def test_per_task_score_is_the_mean_over_repeats() -> None:
    """The canonical reduction averages a task's repeat rows."""

    task = TaskRows(
        task_hash="a" * 64,
        rows=(RowValue(value=1.0), RowValue(value=0.0)),
    )
    assert per_task_score(task, 2, SKIP_CONFIG) == pytest.approx(0.5)
    assert per_task_count(task, 2) == 2


# --- the three row shapes --------------------------------------------------
#
# ``per_task_score`` means over *present* rows and ``per_task_count`` counts
# them. A lost repeat is skipped, never scored 0.0, and a task with no present
# row is ``None`` (unobserved) rather than a measured zero.


def test_per_task_all_rows_present_is_the_plain_mean() -> None:
    task = TaskRows(
        task_hash="a" * 64,
        rows=(RowValue(value=1.0), RowValue(value=0.5)),
    )
    assert per_task_score(task, 2, SKIP_CONFIG) == pytest.approx(0.75)
    assert per_task_count(task, 2) == 2


@pytest.mark.parametrize(
    "lost",
    [
        RowValue(failed=True),
        RowValue(missing=True),
        RowValue(invalid=True),
    ],
    ids=["failed", "missing", "invalid"],
)
def test_per_task_some_rows_missing_skips_them(lost: RowValue) -> None:
    """A lost repeat is skipped, not averaged in as a zero.

    Scoring it 0.0 over ``num_seeds`` would report 0.5 here -- the exact
    disagreement with the tolerant evaluation-level aggregate this contract
    removes.
    """

    task = TaskRows(task_hash="a" * 64, rows=(RowValue(value=1.0), lost))
    assert per_task_score(task, 2, SKIP_CONFIG) == pytest.approx(1.0)
    assert per_task_count(task, 2) == 1


def test_per_task_short_row_tuple_counts_only_present_rows() -> None:
    """A repeat that never produced a row is padded in as missing."""

    task = TaskRows(task_hash="a" * 64, rows=(RowValue(value=1.0),))
    assert per_task_score(task, 2, SKIP_CONFIG) == pytest.approx(1.0)
    assert per_task_count(task, 2) == 1


@pytest.mark.parametrize(
    "config", [SKIP_CONFIG, PROPAGATE_CONFIG], ids=["skip", "propagate"]
)
def test_per_task_all_rows_missing_is_unobserved_not_zero(config) -> None:
    """A fully lost task has no score under either policy."""

    task = TaskRows(
        task_hash="a" * 64,
        rows=(RowValue(failed=True), RowValue(missing=True)),
    )
    assert per_task_score(task, 2, config) is None
    assert per_task_count(task, 2) == 0


def test_per_task_propagate_policy_withholds_a_partial_task() -> None:
    """Under ``propagate`` a lost repeat withholds the task's score."""

    task = TaskRows(
        task_hash="a" * 64,
        rows=(RowValue(value=1.0), RowValue(failed=True)),
    )
    assert per_task_score(task, 2, PROPAGATE_CONFIG) is None
    # The count still reports the one row that was actually observed.
    assert per_task_count(task, 2) == 1


# --- COPRO -----------------------------------------------------------------


def test_copro_evaluates_every_repeat_and_scores_the_mean(
    sqlite_store,
) -> None:
    """COPRO's in-search evaluation pays for repeats and reduces them."""

    experiment = _three_task_experiment(num_seeds=REPEATS)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store, experiment=experiment
    )
    control = build_toy_copro_control(engine=engine)
    runtime = register_toy_runtime(
        store=sqlite_store,
        engine=engine,
        copro_control=control,
    )
    run_id = f"copro-repeats-{uuid4().hex[:8]}"
    prepare_toy_copro_run(
        runtime,
        run_id=run_id,
        control=control,
        experiment=experiment,
    )
    runtime.controller.drive(
        RunRequest(
            controller_identity_hash=runtime.controller.runtime_hash,
            run_id=run_id,
            control_identity_hash=control.identity_hash(),
        )
    )

    records = _evidence_records(sqlite_store, run_id)
    assert records, "the COPRO run recorded no evaluation evidence"
    for evidence in records:
        assert evidence.num_seeds == REPEATS
        assert evidence.row_accounting.planned == (
            len(evidence.task_hashes) * REPEATS
        )
        # One reduced score per task, each backed by every repeat.
        assert len(evidence.per_task_values) == len(evidence.task_hashes)
        assert evidence.per_task_counts == tuple(
            REPEATS for _ in evidence.task_hashes
        )


# --- MIPROv2 ---------------------------------------------------------------


def test_miprov2_runs_its_search_at_multiple_repeats(sqlite_store) -> None:
    """MIPROv2 terminalizes at repeats and every evaluation pays for them.

    Fails before repeats were plumbed with ``engine sampling repeats (2) do
    not match the requested num_seeds (1)``.
    """

    experiment = build_toy_experiment(num_seeds=REPEATS)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store, experiment=experiment
    )
    control = build_toy_miprov2_control(engine=engine)
    # The control records the repeat count the bound engine samples.
    assert control.num_seeds == REPEATS

    adapter = build_miprov2_adapter(
        store=sqlite_store, control=control, engine=engine
    )
    runtime = register_toy_runtime(
        store=sqlite_store,
        engine=engine,
        copro_control=build_toy_copro_control(engine=engine),
        extra_adapters={MIPROV2_ADAPTER_KEY: adapter},
    )
    run_id = f"miprov2-repeats-{uuid4().hex[:8]}"
    prepare_toy_miprov2_run(
        runtime,
        run_id=run_id,
        control=control,
        engine=engine,
        experiment=experiment,
    )
    runtime.controller.drive(
        RunRequest(
            controller_identity_hash=runtime.controller.runtime_hash,
            run_id=run_id,
            control_identity_hash=control.identity_hash(),
        )
    )

    records = _evidence_records(sqlite_store, run_id)
    assert records, "the MIPROv2 run recorded no evaluation evidence"
    for evidence in records:
        assert evidence.num_seeds == REPEATS
        assert evidence.row_accounting.planned == (
            len(evidence.task_hashes) * REPEATS
        )
        assert evidence.per_task_counts == tuple(
            REPEATS for _ in evidence.task_hashes
        )


def test_miprov2_records_the_repeat_count_in_its_study_transcript(
    sqlite_store,
) -> None:
    """The persisted study contract states the repeats it ran under."""

    experiment = build_toy_experiment(num_seeds=REPEATS)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store, experiment=experiment
    )
    control = build_toy_miprov2_control(engine=engine)
    adapter = build_miprov2_adapter(
        store=sqlite_store, control=control, engine=engine
    )
    runtime = register_toy_runtime(
        store=sqlite_store,
        engine=engine,
        copro_control=build_toy_copro_control(engine=engine),
        extra_adapters={MIPROV2_ADAPTER_KEY: adapter},
    )
    run_id = f"miprov2-transcript-{uuid4().hex[:8]}"
    prepare_toy_miprov2_run(
        runtime,
        run_id=run_id,
        control=control,
        engine=engine,
        experiment=experiment,
    )
    runtime.controller.drive(
        RunRequest(
            controller_identity_hash=runtime.controller.runtime_hash,
            run_id=run_id,
            control_identity_hash=control.identity_hash(),
        )
    )

    from whetstone.optim.harness import OptimHarness
    from whetstone.optim.contracts import OptimStepResult

    index = 0
    final_state: Miprov2State | None = None
    while True:
        key = OptimHarness._result_binding_key(run_id, index)  # noqa: SLF001
        bound = sqlite_store.resolve(key)
        if bound is None:
            break
        result = OptimStepResult.model_validate(sqlite_store.get(bound))
        snapshot = sqlite_store.get(result.state_ref.reference)
        final_state = Miprov2State.model_validate(
            snapshot[MIPROV2_STATE_KEY]
        )
        index += 1

    assert final_state is not None
    transcript = final_state.study_transcript
    assert transcript is not None
    # An audit reads the repeat count off the record rather than inferring
    # it from row counts.
    assert transcript.validation_num_seeds == REPEATS


# --- GEPA ------------------------------------------------------------------


def test_gepa_runs_its_search_at_multiple_repeats(sqlite_store) -> None:
    """GEPA terminalizes at repeats and every evaluation pays for them.

    Fails before repeats were plumbed with ``GEPA evaluation engine must use
    a single-repeat plan``, raised while building the eval authority.
    """

    from whetstone.optim.gepa.harness_adapter import GEPA_ADAPTER_KEY
    from whetstone.testing.runtime import (
        build_toy_gepa_adapter,
        build_toy_gepa_control,
        prepare_toy_gepa_run,
    )

    experiment = build_toy_experiment(num_seeds=REPEATS)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store, experiment=experiment
    )
    assert engine.sampling.num_seeds == REPEATS

    run_id = f"gepa-repeats-{uuid4().hex[:8]}"
    control = build_toy_gepa_control(engine=engine, max_metric_calls=2)
    # Building the adapter constructs the eval authority, which is what used
    # to refuse a multi-repeat plan outright.
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
    prepare_toy_gepa_run(
        runtime,
        run_id=run_id,
        control=control,
        experiment=experiment,
    )
    runtime.controller.drive(
        RunRequest(
            controller_identity_hash=runtime.controller.runtime_hash,
            run_id=run_id,
            control_identity_hash=control.identity_hash(),
        )
    )

    records = _evidence_records(sqlite_store, run_id)
    assert records, "the GEPA run recorded no evaluation evidence"
    for evidence in records:
        assert evidence.num_seeds == REPEATS
        assert evidence.row_accounting.planned == (
            len(evidence.task_hashes) * REPEATS
        )
        assert evidence.per_task_counts == tuple(
            REPEATS for _ in evidence.task_hashes
        )
        # The score fed to the Pareto front is one reduced value per task,
        # not one per repeat row.
        assert len(evidence.per_task_values) == len(evidence.task_hashes)

    # A metric call stays one candidate-task evaluation: the run charged one
    # search-evidence entry per evaluation, and each evaluation's evidence
    # covers whole tasks, so repeats did not multiply the budget unit.
    entries = [
        entry
        for result in _step_results(sqlite_store, run_id)
        for entry in result.search_evidence
    ]
    assert entries, "the GEPA search recorded no evidence entries"
    assert len(entries) == len(set(entry.eval_request_id for entry in entries))


def test_gepa_metric_calls_stay_in_task_units_under_repeats(
    sqlite_store,
) -> None:
    """A metric call is one candidate-task evaluation, repeat-independent.

    ``max_metric_calls = 200`` was pre-registered *in metric calls*, and the
    budget the code already computes sums task counts (valset size plus
    minibatch sizes). Repeats multiply provider rows, not metric calls, so
    the pinned budget keeps its meaning: a GEPA evaluation effect reports one
    logical metric call per requested task at any repeat count.
    """

    from whetstone.optim.gepa.control import gepa_auto_budget

    # The resolved budget is a pure function of *task* counts: it scales
    # with the valset size and is untouched by any repeat count, which is
    # what makes "200 metric calls" mean the same thing at K_REPEAT 1 and 3.
    ten = gepa_auto_budget(
        num_predictors=1, num_candidates=4, valset_size=10
    )
    twenty = gepa_auto_budget(
        num_predictors=1, num_candidates=4, valset_size=20
    )
    assert twenty > ten, "the budget must scale with the task count"
    assert "num_seeds" not in gepa_auto_budget.__annotations__, (
        "the metric-call budget must not take a repeat term"
    )

    # A toy control resolves the budget it was handed, unchanged, against a
    # split that carries repeats.
    from whetstone.testing.runtime import build_toy_gepa_control

    experiment = build_toy_experiment(num_seeds=REPEATS)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store, experiment=experiment
    )
    control = build_toy_gepa_control(engine=engine, max_metric_calls=7)
    assert control.resolved_max_metric_calls == 7


# --- a failed in-search evaluation debits the full repeat matrix ----------


def test_miprov2_evaluation_failure_debits_tasks_times_repeats(
    sqlite_store,
) -> None:
    """A failed evaluation's ledger entry counts task x repeat rows.

    ``resolve_evaluation_failure`` writes ``row_accounting.planned`` straight
    into the completed-effect ledger, while the canonical replay in
    ``_canonical_completed_effects`` recomputes it as
    ``len(task_batch_hashes) * num_seeds``. Counting tasks alone made the two
    disagree, so at ``num_seeds > 1`` folding a non-COMPLETED in-search
    evaluation wedged the run with "completed-effect ledger is not the
    canonical evidence replay", and the ledger under-reported the rows that
    evaluation had already debited.
    """

    from whetstone.core.identity import TypedRef
    from whetstone.core.roles import EvalRole
    from whetstone.eval.schema import EvalFailureEvidence
    from whetstone.eval.schema_names import EVAL_FAILURE_SCHEMA
    from whetstone.optim.contracts import (
        INTENT_RESOLUTION_SCHEMA_VERSION,
        IntentOutcome,
        IntentResolution,
        ResolutionClass,
        ResolutionDetail,
        TerminalFailure,
    )
    from whetstone.optim.contracts import OptimStepResult
    from whetstone.optim.miprov2.adapter import fold_resolution
    from whetstone.optim.miprov2.evidence import (
        Miprov2EvidenceResolver,
        load_miprov2_intent_context,
    )

    experiment = build_toy_experiment(num_seeds=REPEATS)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store, experiment=experiment
    )
    control = build_toy_miprov2_control(engine=engine)
    assert control.num_seeds == REPEATS
    adapter = build_miprov2_adapter(
        store=sqlite_store, control=control, engine=engine
    )
    runtime = register_toy_runtime(
        store=sqlite_store,
        engine=engine,
        copro_control=build_toy_copro_control(engine=engine),
        extra_adapters={MIPROV2_ADAPTER_KEY: adapter},
    )
    run_id = f"miprov2-failure-{uuid4().hex[:8]}"
    prepare_toy_miprov2_run(
        runtime,
        run_id=run_id,
        control=control,
        engine=engine,
        experiment=experiment,
    )
    runtime.controller.drive(
        RunRequest(
            controller_identity_hash=runtime.controller.runtime_hash,
            run_id=run_id,
            control_identity_hash=control.identity_hash(),
        )
    )

    # Replay the run's own Steps to recover a state that has a *pending*
    # evaluation -- the exact state a failed in-search evaluation folds into.
    from whetstone.optim.harness import OptimHarness

    pending_state: Miprov2State | None = None
    pending_request = None
    index = 0
    while True:
        key = OptimHarness._result_binding_key(run_id, index)  # noqa: SLF001
        bound = sqlite_store.resolve(key)
        if bound is None:
            break
        result = OptimStepResult.model_validate(sqlite_store.get(bound))
        snapshot = sqlite_store.get(result.state_ref.reference)
        state = Miprov2State.model_validate(snapshot[MIPROV2_STATE_KEY])
        if state.pending_evaluation is not None and result.resolved_intents:
            pending_state = state
            pending_request = result.resolved_intents[0].optim_eval_request
            break
        index += 1

    assert pending_state is not None and pending_request is not None, (
        "the MIPROv2 run never left an evaluation pending"
    )
    context = load_miprov2_intent_context(sqlite_store, pending_request)
    assert context.num_seeds == REPEATS
    assert context.task_batch_hashes

    failure = EvalFailureEvidence(
        candidate=context.candidate,
        eval_config_ref=context.eval_config,
        eval_role=EvalRole.INTERNAL,
        provider_execution_policy_ref=context.provider_execution_policy_ref,
        metadata=pending_request.eval_request.metadata,
        exception_type="RuntimeError",
        message="in-search evaluation failed",
    )
    failure_obj, _ = sqlite_store.put(
        EVAL_FAILURE_SCHEMA, failure.record_content()
    )
    resolution = IntentResolution(
        schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
        optim_eval_request=pending_request,
        outcome=IntentOutcome.FAILED,
        detail=ResolutionDetail(
            classification=ResolutionClass.INFRASTRUCTURE,
            message="the provider refused the request",
        ),
        eval_result_ref=TypedRef(
            schema_name=failure_obj.schema,
            content_hash=failure_obj.content_hash,
        ),
        resolved_eval_config=context.eval_config,
        terminal_failure=TerminalFailure(
            code="provider_error", message="provider_error"
        ),
    )

    # The resolver's planned rows are the whole task x repeat matrix.
    resolved = Miprov2EvidenceResolver(sqlite_store).resolve_evaluation_failure(
        resolution
    )
    expected_rows = len(context.task_batch_hashes) * REPEATS
    assert resolved.row_accounting.planned == expected_rows
    assert resolved.row_accounting.failed == expected_rows

    # Folding it reaches a state that re-validates against the canonical
    # replay. Counting tasks alone raised here instead.
    folded = fold_resolution(sqlite_store, pending_state, resolution)
    ledgered = [
        effect
        for effect in folded.completed_effects
        if effect.kind == "evaluations"
        and effect.identity_hash == context.effect_identity_hash
    ]
    assert len(ledgered) == 1
    assert ledgered[0].task_rows == expected_rows

    # The folded state round-trips its own ledger validation, which is what
    # the next Step and the step contract both do before continuing.
    assert Miprov2State.model_validate(
        folded.model_dump(mode="json")
    ).completed_effects == folded.completed_effects


# --- the score the search consumes is the repeat *mean* -------------------
#
# The tests above pin row *counts*. Counts alone cannot tell a mean apart
# from repeat 0, the max, or the sum, because the default toy scorer ignores
# ``seed_index`` and every repeat of a task scores identically. These tests
# run the same optimizers against ``RepeatVaryingEvalProcedureRunner``, whose
# repeats genuinely differ, and assert the optimizer-visible score is the
# arithmetic mean of the differing rows -- and is *not* any of the three
# reductions it would be easy to write by accident.


def _unmatched_gold_experiment(*, num_seeds: int):
    """A toy experiment whose golds never appear in the rendered prompt.

    ``score_generation`` returns exactly 1.0 whenever the gold is a substring
    of the generation, at every repeat -- it is the toy scorer's exact-match
    anchor. The default toy tasks all hit it, which would pin every repeat to
    1.0 and make the repeat-mean assertions vacuous again. These golds do not
    appear in the rendered prompt, so scoring falls to the seed-varying hash.
    """

    return build_toy_experiment(
        num_seeds=num_seeds,
        internal_tasks=(
            ToyTask(
                task_id="task-a",
                prompt_inputs={"prompt": "alpha"},
                gold="unmatched-gold-a",
            ),
            ToyTask(
                task_id="task-b",
                prompt_inputs={"prompt": "beta"},
                gold="unmatched-gold-b",
            ),
        ),
        official_tasks=(
            ToyTask(
                task_id="task-o",
                prompt_inputs={"prompt": "omega"},
                gold="unmatched-gold-o",
            ),
        ),
    )


def _varying_engine(store, *, experiment):
    """An engine whose repeats of one task score differently."""

    from whetstone.testing.fakes import RepeatVaryingEvalProcedureRunner

    return ReferenceEvalRuntimeConfig().build_engine(
        store,
        experiment=experiment,
        eval_runner=RepeatVaryingEvalProcedureRunner(),
    )


def _rows_by_task(store, evidence: EvalEvidence):
    """Raw per-(task, repeat) scores behind one evaluation, task-major."""

    from whetstone.eval.schema import EvalOutputsRecord

    outputs = EvalOutputsRecord.model_validate(
        store.get(evidence.outputs_ref.reference)
    )
    grouped: dict[str, list[float]] = {}
    for row in outputs.outputs:
        grouped.setdefault(row.task_hash, []).append(float(row.score or 0.0))
    return grouped


def _assert_mean_not_other_reductions(store, evidence: EvalEvidence) -> None:
    """``per_task_values`` is the mean, distinguishably so.

    Asserts per task that the reduced value equals the arithmetic mean of
    that task's repeat rows and differs from repeat 0, the max, and the sum.
    The inequalities are the point: they are what fails if the reduction is
    silently changed to any of those.
    """

    grouped = _rows_by_task(store, evidence)
    assert grouped, "the evaluation recorded no output rows"
    differed = False
    for task_hash, reduced in zip(
        evidence.task_hashes, evidence.per_task_values, strict=True
    ):
        repeats = grouped[task_hash]
        assert len(repeats) == REPEATS
        mean = sum(repeats) / len(repeats)
        assert reduced == pytest.approx(mean)
        if len(set(repeats)) > 1:
            differed = True
            # A vacuous fixture would satisfy every reduction at once.
            assert reduced != pytest.approx(repeats[0])
            assert reduced != pytest.approx(max(repeats))
            assert reduced != pytest.approx(sum(repeats))
    assert differed, (
        "no task's repeats differed, so the mean assertion proves nothing"
    )


def test_copro_candidate_reward_is_the_mean_of_differing_repeats(
    sqlite_store,
) -> None:
    """COPRO's candidate reward is the repeat mean, not repeat 0."""

    experiment = _unmatched_gold_experiment(num_seeds=REPEATS)
    engine = _varying_engine(sqlite_store, experiment=experiment)
    control = build_toy_copro_control(engine=engine)
    runtime = register_toy_runtime(
        store=sqlite_store,
        engine=engine,
        copro_control=control,
    )
    run_id = f"copro-mean-{uuid4().hex[:8]}"
    prepare_toy_copro_run(
        runtime,
        run_id=run_id,
        control=control,
        experiment=experiment,
    )
    runtime.controller.drive(
        RunRequest(
            controller_identity_hash=runtime.controller.runtime_hash,
            run_id=run_id,
            control_identity_hash=control.identity_hash(),
        )
    )

    records = _evidence_records(sqlite_store, run_id)
    assert records, "the COPRO run recorded no evaluation evidence"
    for evidence in records:
        assert evidence.num_seeds == REPEATS
        _assert_mean_not_other_reductions(sqlite_store, evidence)

        # The reward COPRO's search ranks candidates by is built from those
        # reduced per-task values, so it inherits the mean.
        assert evidence.reward_ref is not None
        assert evidence.aggregate_value == pytest.approx(
            sum(evidence.per_task_values) / len(evidence.per_task_values)
        )
        assert evidence.reward_ref.record.value == pytest.approx(
            evidence.aggregate_value
        )


def test_miprov2_normalized_score_is_the_mean_of_differing_repeats(
    sqlite_store,
) -> None:
    """MIPROv2's ``normalized_score`` is the repeat mean, not repeat 0.

    ``normalized_score`` is the number the study's ``param_score_dict``
    stores and the sampler optimizes, so it is exactly "the reward the
    search consumed".
    """

    experiment = _unmatched_gold_experiment(num_seeds=REPEATS)
    engine = _varying_engine(sqlite_store, experiment=experiment)
    control = build_toy_miprov2_control(engine=engine)
    adapter = build_miprov2_adapter(
        store=sqlite_store, control=control, engine=engine
    )
    runtime = register_toy_runtime(
        store=sqlite_store,
        engine=engine,
        copro_control=build_toy_copro_control(engine=engine),
        extra_adapters={MIPROV2_ADAPTER_KEY: adapter},
    )
    run_id = f"miprov2-mean-{uuid4().hex[:8]}"
    prepare_toy_miprov2_run(
        runtime,
        run_id=run_id,
        control=control,
        engine=engine,
        experiment=experiment,
    )
    runtime.controller.drive(
        RunRequest(
            controller_identity_hash=runtime.controller.runtime_hash,
            run_id=run_id,
            control_identity_hash=control.identity_hash(),
        )
    )

    records = _evidence_records(sqlite_store, run_id)
    assert records, "the MIPROv2 run recorded no evaluation evidence"
    by_reward: dict[str, float] = {}
    for evidence in records:
        assert evidence.num_seeds == REPEATS
        _assert_mean_not_other_reductions(sqlite_store, evidence)
        assert evidence.reward_ref is not None
        by_reward[evidence.candidate.identity_hash] = (
            evidence.reward_ref.record.value
        )

    # Every score the study recorded is the normalization of a repeat-mean
    # reward: ``normalized_score = round(reward * 100, 2)``.
    from whetstone.optim.contracts import OptimStepResult
    from whetstone.optim.harness import OptimHarness

    final_state: Miprov2State | None = None
    index = 0
    while True:
        key = OptimHarness._result_binding_key(run_id, index)  # noqa: SLF001
        bound = sqlite_store.resolve(key)
        if bound is None:
            break
        result = OptimStepResult.model_validate(sqlite_store.get(bound))
        snapshot = sqlite_store.get(result.state_ref.reference)
        raw = snapshot.get(MIPROV2_STATE_KEY)
        if raw is not None:
            final_state = Miprov2State.model_validate(raw)
        index += 1

    assert final_state is not None
    transcript = final_state.study_transcript
    assert transcript is not None
    assert transcript.validation_num_seeds == REPEATS

    observations = [transcript.baseline.evaluation]
    for sample in transcript.samples:
        observations.append(sample.evaluation)
        if sample.promotion is not None:
            observations.append(sample.promotion.evaluation)
    assert observations
    checked = 0
    for observation in observations:
        reward = by_reward.get(observation.candidate.identity_hash)
        if reward is None:
            continue
        assert observation.normalized_score == pytest.approx(
            round(reward * 100, 2)
        )
        checked += 1
    assert checked, "no study observation matched a recorded reward"


def test_gepa_pareto_score_is_the_mean_of_differing_repeats(
    sqlite_store,
) -> None:
    """The score GEPA's Pareto front ranks is the repeat mean.

    ``GepaEvalAuthority`` projects one row per *task*, taking the output and
    trace from repeat 0 but the ``score`` from ``per_task_values[index]`` --
    the reduced mean. This pins that the score is the mean and not the
    repeat-0 row it sits beside.
    """

    from whetstone.optim.gepa.harness_adapter import GEPA_ADAPTER_KEY
    from whetstone.testing.runtime import (
        build_toy_gepa_adapter,
        build_toy_gepa_control,
        prepare_toy_gepa_run,
    )

    experiment = _unmatched_gold_experiment(num_seeds=REPEATS)
    engine = _varying_engine(sqlite_store, experiment=experiment)
    run_id = f"gepa-mean-{uuid4().hex[:8]}"
    control = build_toy_gepa_control(engine=engine, max_metric_calls=2)
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
    prepare_toy_gepa_run(
        runtime,
        run_id=run_id,
        control=control,
        experiment=experiment,
    )
    runtime.controller.drive(
        RunRequest(
            controller_identity_hash=runtime.controller.runtime_hash,
            run_id=run_id,
            control_identity_hash=control.identity_hash(),
        )
    )

    records = _evidence_records(sqlite_store, run_id)
    assert records, "the GEPA run recorded no evaluation evidence"
    for evidence in records:
        assert evidence.num_seeds == REPEATS
        _assert_mean_not_other_reductions(sqlite_store, evidence)

    # The scores GEPA's Pareto front ranks are those same reduced means:
    # ``_project_evaluation`` takes ``score=per_task_values[index]``, one row
    # per task, so pinning ``per_task_values`` above pins the Pareto input.


# --- one flaky repeat must not wedge the GEPA evaluation ------------------


def test_gepa_representative_repeat_is_the_lowest_completed_seed(
    sqlite_store,
) -> None:
    """Row selection skips failed repeats and reports the failure count.

    The representative repeat GEPA reflects on is the lowest ``seed_index``
    that *completed*, not unconditionally repeat 0, so that one flaky repeat
    does not decide what the reflection sees. Selection is deterministic
    under replay because "lowest completed" is a total order on the block.
    """

    from whetstone.optim.gepa.authorities import CanonicalGepaEvalAuthority

    task_hash = "t" * 64
    # A failed repeat is one carrying no score -- the same rule the eval
    # plane uses for presence.
    block = [
        {
            "seed_index": 0,
            "task_hash": task_hash,
            "score": None,
            "failure_code": "provider_error",
        },
        {"seed_index": 1, "task_hash": task_hash, "score": 0.5, "failure_code": ""},
    ]
    representative, failed_repeats, all_failed = (
        CanonicalGepaEvalAuthority._representative_repeat(  # noqa: SLF001
            raw_rows=block,
            index=0,
            num_seeds=REPEATS,
            expected_task_hash=task_hash,
        )
    )
    assert representative["seed_index"] == 1
    assert failed_repeats == 1
    assert all_failed is False

    # Every repeat failed: there is no completed row, so the task is failed
    # and repeat 0 stands in as the representative.
    all_bad = [
        {
            "seed_index": 0,
            "task_hash": task_hash,
            "score": None,
            "failure_code": "provider_error",
        },
        {
            "seed_index": 1,
            "task_hash": task_hash,
            "score": None,
            "failure_code": "provider_error",
        },
    ]
    representative, failed_repeats, all_failed = (
        CanonicalGepaEvalAuthority._representative_repeat(  # noqa: SLF001
            raw_rows=all_bad,
            index=0,
            num_seeds=REPEATS,
            expected_task_hash=task_hash,
        )
    )
    assert representative["seed_index"] == 0
    assert failed_repeats == REPEATS
    assert all_failed is True


def test_gepa_row_block_must_belong_to_its_own_task(sqlite_store) -> None:
    """Mis-attributed output rows are loud, not silent (item 4 guard).

    Without this, a misaligned outputs record would hand one task's
    generation and trace to another task's score.
    """

    from whetstone.optim.gepa.authorities import CanonicalGepaEvalAuthority

    mine, theirs = "a" * 64, "b" * 64
    with pytest.raises(
        ValueError, match="belongs to another task"
    ):
        CanonicalGepaEvalAuthority._representative_repeat(  # noqa: SLF001
            raw_rows=[
                {"seed_index": 0, "task_hash": mine, "failure_code": ""},
                {"seed_index": 1, "task_hash": theirs, "failure_code": ""},
            ],
            index=0,
            num_seeds=REPEATS,
            expected_task_hash=mine,
        )

    with pytest.raises(
        ValueError, match="not ordered task-major by seed_index"
    ):
        CanonicalGepaEvalAuthority._representative_repeat(  # noqa: SLF001
            raw_rows=[
                {"seed_index": 1, "task_hash": mine, "failure_code": ""},
                {"seed_index": 0, "task_hash": mine, "failure_code": ""},
            ],
            index=0,
            num_seeds=REPEATS,
            expected_task_hash=mine,
        )


def test_gepa_flaky_repeat_zero_still_projects_a_scored_row(
    sqlite_store,
) -> None:
    """Repeat 0 failing beside a successful repeat 1 yields a *scored* row.

    Fails before with ``failed GEPA evaluation row score must be zero before
    the bound failure_score projection``: the projection took repeat 0
    unconditionally, minted a ``failure_ref`` from its failure code, and then
    handed the row the task's nonzero repeat-mean, which
    ``GepaEvaluationRow._validate`` rejects. One flaky repeat wedged the
    whole GEPA evaluation.
    """

    from whetstone.optim.gepa.authorities import CanonicalGepaEvalAuthority
    from whetstone.optim.gepa.contracts import GepaEvaluationRow

    # A task whose repeat 0 failed but whose repeat 1 completed. The
    # canonical per-task score averages the completed repeats, so it is a
    # real, nonzero score -- the exact pairing the row validator rejected.
    task_hash = "c" * 64
    block = [
        {
            "seed_index": 0,
            "task_hash": task_hash,
            "score": None,
            "failure_code": "provider_error",
        },
        {"seed_index": 1, "task_hash": task_hash, "score": 0.5, "failure_code": ""},
    ]
    representative, failed_repeats, all_failed = (
        CanonicalGepaEvalAuthority._representative_repeat(  # noqa: SLF001
            raw_rows=block,
            index=0,
            num_seeds=REPEATS,
            expected_task_hash=task_hash,
        )
    )
    # The scored (non-failed) repeat represents the task ...
    assert representative["failure_code"] == ""
    # ... so the row is scored, not failed, and carries the mean.
    assert all_failed is False
    assert failed_repeats == 1

    # And the contract itself agrees: a scored row with a nonzero score and
    # no failure_ref validates, while the old pairing does not.
    from whetstone.core.identity import TypedRef
    from whetstone.optim.gepa.contracts import GepaDataInstance

    ref = TypedRef(schema_name="whetstone.eval_evidence", content_hash="d" * 64)
    data = GepaDataInstance(
        upstream_position=0,
        data_id="task-a",
        task_hash=task_hash,
        data_ref=TypedRef(
            schema_name="whetstone.gepa.reflection_input",
            content_hash="e" * 64,
        ),
        loader_identity_hash="9" * 64,
    )
    scored = GepaEvaluationRow(
        data=data,
        output={"output_text": "ok"},
        score=0.5,
        evidence_refs=(ref,),
    )
    assert scored.score == 0.5 and scored.failure_ref is None
    with pytest.raises(
        ValueError, match="failed GEPA evaluation row score must be zero"
    ):
        GepaEvaluationRow(
            data=data,
            output={"output_text": "ok"},
            score=0.5,
            evidence_refs=(ref,),
            failure_ref=TypedRef(
                schema_name="whetstone.gepa.evaluation_row_failure",
                content_hash="f" * 64,
            ),
        )


# --- a blank repeat is a completed, scored repeat -------------------------


def test_gepa_blank_repeat_is_a_completed_scored_repeat() -> None:
    """A blank generation is a sample GEPA reflects on, not a failed repeat.

    Fails before: ``_representative_repeat`` keyed "failed" off any non-empty
    ``failure_code``, so a blank row -- which now completes with a real score
    while retaining ``blank-provider-generation`` as explanation -- was
    counted in ``failed_repeats`` and excluded from being the representative.
    That hid from reflection exactly the blank-output signal it should learn
    from.
    """

    from whetstone.optim.gepa.authorities import CanonicalGepaEvalAuthority

    task_hash = "b" * 64
    # Repeat 0 is blank: scored at the family floor, code retained.
    block = [
        {
            "seed_index": 0,
            "task_hash": task_hash,
            "score": 0.0,
            "output_text": "",
            "failure_code": "blank-provider-generation",
        },
        {"seed_index": 1, "task_hash": task_hash, "score": 0.6, "failure_code": ""},
    ]
    representative, failed_repeats, all_failed = (
        CanonicalGepaEvalAuthority._representative_repeat(  # noqa: SLF001
            raw_rows=block,
            index=0,
            num_seeds=REPEATS,
            expected_task_hash=task_hash,
        )
    )

    # The blank repeat completed, so it is representative-eligible and, being
    # the lowest completed seed, it *is* the representative.
    assert representative["seed_index"] == 0
    assert representative["failure_code"] == "blank-provider-generation"
    # It is not counted as a failed repeat.
    assert failed_repeats == 0
    assert all_failed is False


def test_gepa_all_blank_task_is_scored_not_wedged() -> None:
    """Every repeat blank: a real 0.0 score, no failure, no wedge.

    Fails before: all-blank set ``all_repeats_failed=True`` while the
    canonical per-task score was a real 0.0. That pairing only survived
    because blanks happen to score exactly 0.0 -- any eval family whose blank
    score is nonzero would hard-wedge the GEPA evaluation on
    ``GepaEvaluationRow._validate``.
    """

    from whetstone.optim.gepa.authorities import (
        CanonicalGepaEvalAuthority,
        _gepa_row_score,
    )

    task_hash = "a" * 64
    all_blank = [
        {
            "seed_index": 0,
            "task_hash": task_hash,
            "score": 0.0,
            "failure_code": "blank-provider-generation",
        },
        {
            "seed_index": 1,
            "task_hash": task_hash,
            "score": 0.0,
            "failure_code": "blank-provider-generation",
        },
    ]
    representative, failed_repeats, all_failed = (
        CanonicalGepaEvalAuthority._representative_repeat(  # noqa: SLF001
            raw_rows=all_blank,
            index=0,
            num_seeds=REPEATS,
            expected_task_hash=task_hash,
        )
    )
    assert failed_repeats == 0
    assert all_failed is False

    # The task carries its measured mean rather than the inert failed-row
    # stand-in, and asking for the score does not raise.
    assert _gepa_row_score(0.0, all_repeats_failed=all_failed) == 0.0


def test_gepa_all_blank_task_does_not_wedge_at_a_nonzero_blank_score() -> None:
    """The wedge the old predicate hid: a nonzero blank score.

    An eval family whose empty-output score is not 0.0 would, under the old
    ``failure_code`` predicate, mint ``all_repeats_failed=True`` beside a
    nonzero canonical score -- and ``_gepa_row_score`` raises on exactly that
    combination. Score-based failure makes the family's floor irrelevant.
    """

    from whetstone.optim.gepa.authorities import (
        CanonicalGepaEvalAuthority,
        _gepa_row_score,
    )

    task_hash = "9" * 64
    # A family that floors empty output at 0.25 rather than 0.0.
    all_blank = [
        {
            "seed_index": seed,
            "task_hash": task_hash,
            "score": 0.25,
            "failure_code": "blank-provider-generation",
        }
        for seed in range(REPEATS)
    ]
    _representative, failed_repeats, all_failed = (
        CanonicalGepaEvalAuthority._representative_repeat(  # noqa: SLF001
            raw_rows=all_blank,
            index=0,
            num_seeds=REPEATS,
            expected_task_hash=task_hash,
        )
    )
    assert failed_repeats == 0
    assert all_failed is False
    # Before the fix this raised "cannot score a task whose canonical
    # per-task value is withheld" / minted a failure_ref beside 0.25.
    assert _gepa_row_score(0.25, all_repeats_failed=all_failed) == 0.25


def test_gepa_blank_prediction_reaches_reflection_as_a_failing_prediction() -> None:
    """A blank output is a *completed* row whose prediction still failed.

    Reflection must see the blank as a failing prediction -- that is the
    signal to learn from -- without the row being treated as a lost repeat.
    """

    from whetstone.optim.gepa.submission_projection import (
        DefaultGepaSubmissionProjector,
    )

    projector = DefaultGepaSubmissionProjector()

    # The row completed (row_failed=False) but its submission did not pass.
    assert (
        projector.prediction_failed(
            row_failed=False,
            submission={"score": {"passed": False}},
        )
        is True
    )
    # A row carrying no score is failed outright.
    assert (
        projector.prediction_failed(row_failed=True, submission=None) is True
    )
    # A completed, passing row is not a failed prediction.
    assert (
        projector.prediction_failed(
            row_failed=False,
            submission={"score": {"passed": True}},
        )
        is False
    )


# --- GEPA records the repeat count it ran under --------------------------


def test_gepa_detailed_result_pins_its_repeat_provenance_key() -> None:
    """The GEPA run record's wire keys include ``validation_num_seeds``.

    The record is content-addressed, so its serialized key set is its
    persisted identity. This literal is written out by hand: deriving it from
    the model would agree with any silent drift instead of catching it.
    """

    from whetstone.optim.gepa.engine import GepaDetailedResult

    detailed = GepaDetailedResult(
        candidates=({"generate": "seed"},),
        parents=((None,),),
        val_aggregate_scores=(0.0,),
        val_subscores=({},),
        per_val_instance_best_candidates={},
        discovery_eval_counts=(0,),
        seed=0,
        validation_num_seeds=REPEATS,
        best_idx=0,
        control_identity_hash="a" * 64,
    )
    wire = detailed.model_dump(mode="json")
    assert set(wire) == {
        "candidates",
        "parents",
        "val_aggregate_scores",
        "val_subscores",
        "per_val_instance_best_candidates",
        "discovery_eval_counts",
        "best_outputs_valset",
        "val_aggregate_subscores",
        "per_objective_best_candidates",
        "objective_pareto_front",
        "total_metric_calls",
        "num_full_val_evals",
        "seed",
        "validation_num_seeds",
        "best_idx",
        "control_identity_hash",
        "source_manifest_hash",
        "result_schema_version",
        "upstream_validation_schema_version",
    }
    assert wire["validation_num_seeds"] == REPEATS

    # It defaults to one repeat, so a record written before repeats existed
    # still parses and reads as the single-repeat run it was.
    assert (
        GepaDetailedResult(
            candidates=({"generate": "seed"},),
            parents=((None,),),
            val_aggregate_scores=(0.0,),
            val_subscores=({},),
            per_val_instance_best_candidates={},
            discovery_eval_counts=(0,),
            seed=0,
            best_idx=0,
            control_identity_hash="a" * 64,
        ).validation_num_seeds
        == 1
    )


def test_gepa_run_record_states_the_resolved_repeat_count(
    sqlite_store,
) -> None:
    """A GEPA run records the repeats its evaluations resolved to.

    An audit diffs this against the envs manifest's pre-registered
    ``K_REPEAT`` directly, without walking evidence -- the same thing
    MIPROv2's ``validation_num_seeds`` is for.
    """

    from whetstone.optim.gepa.harness_adapter import GEPA_ADAPTER_KEY
    from whetstone.optim.gepa.step_engine import GepaStepCheckpoint
    from whetstone.testing.runtime import (
        build_toy_gepa_adapter,
        build_toy_gepa_control,
    )

    experiment = build_toy_experiment(num_seeds=REPEATS)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store, experiment=experiment
    )
    assert engine.sampling.num_seeds == REPEATS
    run_id = f"gepa-provenance-{uuid4().hex[:8]}"
    # A zero-budget control terminalizes immediately and still writes the
    # run record, which is exactly the record under test.
    control = build_toy_gepa_control(engine=engine, max_metric_calls=0)
    adapter = build_toy_gepa_adapter(
        store=sqlite_store,
        engine=engine,
        control=control,
        run_id=run_id,
        initial_candidate=experiment.initial_candidate,
    )
    _ = GEPA_ADAPTER_KEY

    from whetstone.optim.gepa.step_engine import run_one_gepa_iteration

    detailed, checkpoint = run_one_gepa_iteration(
        control=control,
        seed_candidate={
            name: experiment.initial_candidate.payload[TOY_MUTATION_FIELD]
            for name in control.component_names
        },
        trainset=(),
        valset=None,
        adapter=adapter,
        checkpoint=GepaStepCheckpoint(),
        validation_num_seeds=engine.sampling.num_seeds,
    )
    assert checkpoint.terminal
    # The run record states the repeat count, so it can be diffed against the
    # manifest without reconstructing it from row counts.
    assert detailed.validation_num_seeds == REPEATS
    # Repeats multiply provider rows, not metric calls.
    assert detailed.total_metric_calls == 0
