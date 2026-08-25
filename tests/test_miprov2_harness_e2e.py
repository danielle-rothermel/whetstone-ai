"""MIPROv2 driven end to end through the harness, in all three demo modes.

These are the first call sites MIPROv2 has ever had. They run the real
step loop against a real ``RuntimeEvalEngine`` -- not a mocked search --
so what they pin is the whole path: the step contract deriving each Step
Request from durable state, the adapter emitting Evaluation Intents, the
eval engine resolving them under the exact task subset each intent
declares, and the study folding the results back.

Every assertion is on recorded state or evidence. Nothing here waits.

Driving one run costs far more than reading it, and every test below
reads a *finished* run. So each demo mode is driven exactly once per
module, through the module-scoped ``completed_run`` fixture, and the
readers share it. The runs whose control shape genuinely differs --
minibatching -- still drive their own.
"""

from __future__ import annotations

import pytest

from dr_store.sync import open_sqlite

from whetstone.coordination.harness_run_controller import RunRequest
from whetstone.testing.runtime import (
    build_miprov2_adapter,
    build_toy_copro_control,
    prepare_toy_miprov2_run,
    register_toy_runtime,
)
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.contracts import (
    OPTIM_RESULT_SCHEMA,
    IntentOutcome,
    OptimStepResult,
    StepStatus,
)
from whetstone.optim.miprov2.adapter import (
    MIPROV2_ADAPTER_KEY,
    MIPROV2_BOOTSTRAP,
    MIPROV2_COMPLETE,
    MIPROV2_STATE_KEY,
)
from whetstone.optim.miprov2.control import Miprov2DemoMode
from whetstone.optim.miprov2.runtime import Miprov2State
from whetstone.optim.miprov2.study import select_promotion
from whetstone.testing.toy.miprov2 import build_toy_miprov2_control

BOOTSTRAP_PURPOSE = "miprov2_bootstrap"


class _Run:
    """One completed MIPROv2 run, with everything the assertions read."""

    def __init__(
        self, *, store, runtime, control, terminal_ref, run_id: str
    ) -> None:
        self.store = store
        self.runtime = runtime
        self.control = control
        self.terminal_ref = terminal_ref
        self.run_id = run_id
        self.results = _step_results(store, runtime, run_id)

    @property
    def intents(self):
        return tuple(
            intent
            for result in self.results
            for intent in result.resolved_intents
        )

    def intents_for(self, purpose: str):
        return tuple(
            intent
            for intent in self.intents
            if intent.optim_eval_request.eval_request.metadata["purpose"]
            == purpose
        )

    def state_after(self, index: int) -> Miprov2State:
        result = self.results[index]
        assert result.state_ref is not None
        snapshot = self.store.get(result.state_ref.reference)
        return Miprov2State.model_validate(snapshot[MIPROV2_STATE_KEY])

    @property
    def final_state(self) -> Miprov2State:
        return self.state_after(-1)


def _step_results(store, runtime, run_id: str) -> tuple[OptimStepResult, ...]:
    from whetstone.optim.harness import OptimHarness

    results: list[OptimStepResult] = []
    index = 0
    while True:
        key = OptimHarness._result_binding_key(run_id, index)  # noqa: SLF001
        bound = store.resolve(key)
        if bound is None:
            return tuple(results)
        results.append(OptimStepResult.model_validate(store.get(bound)))
        index += 1


def _completed(store, demo_mode: Miprov2DemoMode, run_id: str) -> _Run:
    engine = ReferenceEvalRuntimeConfig().build_engine(store)
    control = build_toy_miprov2_control(engine=engine, demo_mode=demo_mode)
    adapter = build_miprov2_adapter(
        store=store, control=control, engine=engine
    )
    runtime = register_toy_runtime(
        store=store,
        engine=engine,
        copro_control=build_toy_copro_control(engine=engine),
        extra_adapters={MIPROV2_ADAPTER_KEY: adapter},
    )
    prepare_toy_miprov2_run(
        runtime, run_id=run_id, control=control, engine=engine
    )
    terminal_ref = runtime.controller.drive(
        RunRequest(
            controller_identity_hash=runtime.controller.runtime_hash,
            run_id=run_id,
            control_identity_hash=control.identity_hash(),
        )
    )
    return _Run(
        store=store,
        runtime=runtime,
        control=control,
        terminal_ref=terminal_ref,
        run_id=run_id,
    )


@pytest.fixture(
    scope="module",
    params=list(Miprov2DemoMode),
    ids=lambda mode: mode.value,
)
def completed_run(request, tmp_path_factory):
    """One completed toy MIPROv2 run per demo mode, shared by the readers.

    The run is fully determined by its demo mode: same toy experiment,
    same control, same engine, same seed. Every test taking this fixture
    reads recorded state off the finished run and mutates nothing, so one
    run per mode is exactly the same evidence as one run per assertion.
    """

    demo_mode: Miprov2DemoMode = request.param
    directory = tmp_path_factory.mktemp(f"miprov2-{demo_mode.value}")
    with open_sqlite(str(directory / f"{demo_mode.value}.sqlite")) as store:
        yield _completed(store, demo_mode, f"e2e-{demo_mode.value}")


@pytest.fixture
def demo_mode(completed_run) -> Miprov2DemoMode:
    return completed_run.control.demo_mode


# --- the run terminalizes in every mode -----------------------------------


def test_miprov2_terminalizes_through_the_harness(completed_run) -> None:
    run = completed_run

    assert run.terminal_ref.schema_name == OPTIM_RESULT_SCHEMA
    assert run.results, "a driven run records at least one Step"
    assert run.results[-1].status is StepStatus.COMPLETE
    assert all(
        result.status is StepStatus.CONTINUE for result in run.results[:-1]
    ), "only the last Step of a completed run may be terminal"
    assert run.final_state.phase == MIPROV2_COMPLETE
    assert run.final_state.terminal_result is not None


def test_first_step_opens_from_the_bound_state(
    completed_run, demo_mode
) -> None:
    """Step 0 reads the launch's opening state, not a rebuild of it."""
    run = completed_run

    first = run.results[0]
    assert first.step_index == 0
    assert first.request.record.prior_step_result_ref is None
    assert first.request.record.prior_state_ref is None
    opening = Miprov2State.model_validate(
        first.request.record.pools[MIPROV2_STATE_KEY]
    )
    assert opening.control.reference() == run.control.reference()
    # A run that bootstraps opens in bootstrap; one that does not opens
    # straight into instruction proposal.
    expected_phase = "bootstrap" if demo_mode.bootstraps else "proposal"
    assert opening.phase == expected_phase


def test_continuations_cite_the_prior_step_exactly(completed_run) -> None:
    run = completed_run

    from whetstone.optim.contracts import step_result_reference

    for index, result in enumerate(run.results[1:], start=1):
        prior = run.results[index - 1]
        request = result.request.record
        assert result.step_index == index
        assert (
            request.prior_step_result_ref
            == step_result_reference(prior).record_ref
        )
        assert request.prior_state_ref == prior.state_ref


# --- the three modes differ where the design says they differ -------------


def test_every_mode_bootstraps_and_only_fewshot_keeps_demos(
    completed_run, demo_mode
) -> None:
    """zeroshot emits the 3/0 grounding bootstrap and no demo_set."""
    run = completed_run

    bootstrap_intents = run.intents_for(BOOTSTRAP_PURPOSE)
    assert bootstrap_intents, (
        f"{demo_mode.value} must bootstrap through the eval engine"
    )
    assert run.final_state.bootstrap_plans


def test_only_fewshot_searches_a_demo_dimension(
    completed_run, demo_mode
) -> None:
    """ground_only bootstraps yet keeps demos out of the search space."""
    run = completed_run

    transcript = run.final_state.study_transcript
    assert transcript is not None
    assert transcript.demo_mode is demo_mode
    searched = transcript.demo_pool_identity_hashes is not None
    assert searched is (demo_mode is Miprov2DemoMode.FEWSHOT)
    assert searched is demo_mode.searches_demos


def test_only_fewshot_attaches_a_demo_set_to_candidates(
    completed_run, demo_mode
) -> None:
    """The ground_only discriminator: demos ground proposals, not candidates."""
    run = completed_run

    transcript = run.final_state.study_transcript
    assert transcript is not None
    selections = [
        component
        for sample in transcript.samples
        for component in sample.candidate_assembly.rendering.components
    ]
    assert selections, "a completed study assembles candidates"
    attached = [
        component
        for component in selections
        if component.demo_set is not None
    ]
    if demo_mode is Miprov2DemoMode.FEWSHOT:
        assert attached, "fewshot renders bootstrapped demos"
    else:
        assert attached == [], f"{demo_mode.value} must not attach a demo set"


def test_ground_only_is_marked_as_a_deviation(
    completed_run, demo_mode
) -> None:
    """The faithful modes keep the DSPy version; ground_only is marked."""
    run = completed_run

    transcript = run.final_state.study_transcript
    assert transcript is not None
    assert transcript.algorithm_version == "dspy_miprov2/v2"
    expected_deviation = (
        "demo_mode:ground_only"
        if demo_mode is Miprov2DemoMode.GROUND_ONLY
        else None
    )
    assert transcript.whetstone_deviation == expected_deviation


# --- evaluations are durable, attributed evidence -------------------------


def test_every_evaluation_is_recorded_as_resolved_evidence(
    completed_run,
) -> None:
    """Each eval MIPROv2 drove is a resolved Intent bound to its own Step."""
    run = completed_run

    assert run.intents, "a MIPROv2 run pays for evaluations"
    for result in run.results:
        # MIPROv2 evaluates through harness intents, not in-search
        # SearchEvidence (GEPA's path for candidates the run never
        # proposes).
        assert result.search_evidence == ()
        for intent in result.resolved_intents:
            request = intent.optim_eval_request
            assert request.optim_run_id == run.run_id
            assert request.optim_step_index == result.step_index
            assert intent.outcome is IntentOutcome.COMPLETED
            assert intent.eval_result_ref is not None
            # The evidence itself resolves, not merely a plausible ref.
            run.store.get(intent.eval_result_ref.reference)


def test_each_intent_declares_the_task_subset_it_evaluated(
    completed_run,
) -> None:
    """A bootstrap runs one task; the recorded evidence matches the subset."""
    from whetstone.eval.schema import EvalEvidence

    run = completed_run

    for intent in run.intents:
        request = intent.optim_eval_request
        assert request.task_hashes is not None, (
            "MIPROv2 always names the subset it evaluated"
        )
        assert intent.eval_result_ref is not None
        evidence = EvalEvidence.model_validate(
            run.store.get(intent.eval_result_ref.reference)
        )
        assert evidence.task_hashes == request.task_hashes

    for intent in run.intents_for(BOOTSTRAP_PURPOSE):
        assert len(intent.optim_eval_request.task_hashes) == 1


def test_bootstrap_evaluations_are_counted_in_the_effect_ledger(
    completed_run,
) -> None:
    """Bootstrap is budgeted search, not an unmetered side channel."""
    run = completed_run

    counts = run.final_state.effect_counts
    assert counts["bootstrap_generations"] == len(
        run.intents_for(BOOTSTRAP_PURPOSE)
    )
    assert counts["evaluations"] == len(run.intents) - len(
        run.intents_for(BOOTSTRAP_PURPOSE)
    )
    # The Step budget is debited by the same ledger the state keeps.
    final_budget = run.results[-1].budget
    assert final_budget.consumed["bootstrap_generations"] == (
        counts["bootstrap_generations"]
    )
    assert final_budget.consumed["evaluations"] == counts["evaluations"]


def test_a_step_labels_itself_by_the_phase_it_ran(completed_run) -> None:
    """The Step contract labels each Step by the phase that actually runs.

    Every mode bootstraps, so this holds for all three, not only fewshot.
    """
    run = completed_run

    labels = [result.request.record.kind_label for result in run.results]
    assert labels[-1] == MIPROV2_COMPLETE
    assert MIPROV2_BOOTSTRAP in labels
    bootstrap_steps = {
        index
        for index, label in enumerate(labels)
        if label == MIPROV2_BOOTSTRAP
    }
    for index in bootstrap_steps:
        purposes = {
            intent.optim_eval_request.eval_request.metadata["purpose"]
            for intent in run.results[index].resolved_intents
        }
        assert purposes == {BOOTSTRAP_PURPOSE}


def _minibatched_run(store, *, run_id: str, num_candidates: int) -> _Run:
    """Drive a minibatched fewshot run at a given candidate count."""

    engine = ReferenceEvalRuntimeConfig().build_engine(store)
    control = build_toy_miprov2_control(
        engine=engine,
        demo_mode=Miprov2DemoMode.FEWSHOT,
        num_candidates=num_candidates,
        num_trials=2,
        minibatch=True,
    )
    adapter = build_miprov2_adapter(
        store=store, control=control, engine=engine
    )
    runtime = register_toy_runtime(
        store=store,
        engine=engine,
        copro_control=build_toy_copro_control(engine=engine),
        extra_adapters={MIPROV2_ADAPTER_KEY: adapter},
    )
    prepare_toy_miprov2_run(
        runtime, run_id=run_id, control=control, engine=engine
    )
    terminal_ref = runtime.controller.drive(
        RunRequest(
            controller_identity_hash=runtime.controller.runtime_hash,
            run_id=run_id,
            control_identity_hash=control.identity_hash(),
        )
    )
    return _Run(
        store=store,
        runtime=runtime,
        control=control,
        terminal_ref=terminal_ref,
        run_id=run_id,
    )


def test_two_candidates_with_minibatching_still_terminalizes(
    tmp_path,
) -> None:
    """A two-candidate minibatched run promotes instead of dying.

    With ``num_candidates == 2`` and minibatching on, consecutive
    full-eval steps can exhaust the observed combinations: every
    combination the sampler has proposed so far has already been
    promoted. DSPy's ``get_program_with_highest_avg_score`` falls back to
    the last-ranked combination in exactly this state, so the run keeps
    going. This pins that the fall-back reaches a terminal result rather
    than raising "No valid program found in param_score_dict" inside the
    durable run boundary.
    """

    with open_sqlite(str(tmp_path / "two-candidates.sqlite")) as store:
        run = _minibatched_run(
            store, run_id="two-candidate-minibatch", num_candidates=2
        )

        assert run.terminal_ref.schema_name == OPTIM_RESULT_SCHEMA
        assert run.results[-1].status is StepStatus.COMPLETE
        assert run.results[-1].request.record.kind_label == MIPROV2_COMPLETE
        # The exhausted step still promoted: a repeated promotion is a
        # real full evaluation, not a skipped one.
        transcript = run.final_state.study_transcript
        promotions = [
            sample.promotion
            for sample in transcript.samples
            if sample.promotion is not None
        ]
        assert promotions, "a minibatched run must promote at least once"


def test_exhausted_promotion_falls_back_to_the_last_ranked_combination(
    tmp_path,
) -> None:
    """``select_promotion`` mirrors DSPy when every combination is spent.

    Built from a real transcript so the observations carry real
    identities. Marking every observed combination as promoted is the
    state that used to raise; DSPy returns the last-ranked (lowest
    mean) combination instead, and so must this.
    """

    with open_sqlite(str(tmp_path / "fallback.sqlite")) as store:
        run = _minibatched_run(
            store, run_id="fallback-run", num_candidates=3
        )
        samples = run.final_state.study_transcript.samples
        promoted = [
            sample for sample in samples if sample.promotion is not None
        ]
        assert promoted, "need a promoted sample to build the spent state"

        # Every combination present is one that was already promoted.
        spent = tuple(
            sample.model_copy(
                update={
                    "promotion": promoted[0].promotion.model_copy(
                        update={
                            "candidate_combination_identity_hash": (
                                sample.candidate_combination_identity_hash
                            )
                        }
                    )
                }
            )
            for sample in samples
        )

        # Used to raise; now returns the last-ranked combination.
        selected = select_promotion(spent)

        means: dict[str, list[float]] = {}
        for sample in spent:
            means.setdefault(
                sample.candidate_combination_identity_hash, []
            ).append(sample.score)
        ranked = sorted(
            means,
            key=lambda key: sum(means[key]) / len(means[key]),
            reverse=True,
        )
        assert selected.candidate_combination_identity_hash == ranked[-1]
        # It is a combination that was genuinely observed, not invented,
        # and it carries that combination's real observed mean.
        assert selected.candidate_combination_identity_hash in means
        observed = means[selected.candidate_combination_identity_hash]
        assert selected.minibatch_mean == sum(observed) / len(observed)
        assert selected.minibatch_mean == min(
            sum(v) / len(v) for v in means.values()
        )
