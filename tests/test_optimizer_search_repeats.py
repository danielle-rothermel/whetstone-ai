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
from whetstone.eval.aggregate import RowValue, TaskRows
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
from whetstone.testing.toy.experiment import ToyTask, build_toy_experiment
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


def test_per_task_score_is_the_mean_over_repeats() -> None:
    """The canonical reduction averages a task's repeat rows."""

    task = TaskRows(
        task_hash="a" * 64,
        rows=(RowValue(value=1.0), RowValue(value=0.0)),
    )
    assert per_task_score(task, 2) == pytest.approx(0.5)
    assert per_task_count(task, 2) == 2


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
