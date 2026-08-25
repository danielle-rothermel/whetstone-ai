"""COPRO retains the seed when nothing it proposed beat the baseline.

COPRO measures the run's own initial candidate alongside its proposals: the
seed round carries it as an occurrence, so it sits in ranked history like any
other measured candidate. A search whose proposals all scored at or below the
baseline therefore terminalizes on the seed.

The seed is not a proposal. It binds the *consumer's* root candidate as its
base -- ``whetstone.toy.root_candidate`` here, ``whetstone_envs.c19.
root_candidate`` in a real study -- rather than a candidate the finalizing
Step was asked about, so handing it back as an accepted proposal fails the
harness base check. The honest shape is the one MIPROv2 and GEPA already use:
COMPLETE with ``seed_retained``, retaining the run's initial candidate and
accepting nothing.

Ties matter here rather than being exotic. Ranking is a stable sort on
descending reward across the whole history, so the seed's placement last
within round 0 only breaks ties against that round -- a tying proposal from
any later round ranks *below* the seed. Exact-match rewards quantize to k/N
over a task set, so equal rewards are common.
"""

from __future__ import annotations

from uuid import uuid4


from whetstone.coordination.runtime_bootstrap import copro_run_request
from whetstone.coordination.step_request_builder import StepRequestBuilder
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.experiment.candidate import candidate_reference
from whetstone.optim.contracts import (
    OPTIM_RESULT_SCHEMA,
    OptimResult,
    OptimStepRequest,
    StepStatus,
)
from whetstone.optim.copro.adapter import (
    COPRO_ADAPTER_KEY,
    attempt_history_entries,
)
from whetstone.optim.harness import OptimHarness
from whetstone.optim.proposal.proposer import ProposalDraft
from whetstone.testing.fakes.proposer import DummyProposerTransport
from whetstone.testing.runtime import (
    build_toy_copro_control,
    prepare_toy_copro_run,
    register_toy_runtime,
)
from whetstone.testing.toy.experiment import ToyTask, build_toy_experiment

# The toy scorer awards an exact 1.0 only when the gold token appears in the
# generation, and the fake transport echoes the rendered prompt. Choosing a
# gold token that does NOT appear in any prompt input makes the anchor
# reachable only by a template that literally spells it -- so the seed can
# score strictly above proposals that merely interpolate ``{prompt}``.
GOLD = "ZQX"
INTERNAL_TASKS = (
    ToyTask(task_id="task-a", prompt_inputs={"prompt": "greet one"}, gold=GOLD),
    ToyTask(task_id="task-b", prompt_inputs={"prompt": "greet two"}, gold=GOLD),
)
OFFICIAL_TASKS = (
    ToyTask(
        task_id="task-c", prompt_inputs={"prompt": "greet three"}, gold=GOLD
    ),
)
# Renders to a generation containing the gold token: the perfect seed.
GOLD_SEED_TEMPLATE = f"Reply {GOLD} to: {{prompt}}"
# Render without the gold token, so they genuinely score below the seed.
PLAIN_BODIES = (
    "Answer {prompt} in one short friendly sentence.",
    "Respond to {prompt} in a single clear sentence.",
    "Give {prompt} one short helpful answer.",
    "Offer {prompt} a brief plain reply.",
)


def _gold_experiment(*, initial_template: str = GOLD_SEED_TEMPLATE):
    return build_toy_experiment(
        internal_tasks=INTERNAL_TASKS,
        official_tasks=OFFICIAL_TASKS,
        initial_template=initial_template,
    )


def _runtime(store, *, bodies=PLAIN_BODIES, breadth=3, depth=1, experiment=None):
    resolved = experiment if experiment is not None else _gold_experiment()
    engine = ReferenceEvalRuntimeConfig().build_engine(
        store, experiment=resolved
    )
    control = build_toy_copro_control(
        breadth=breadth, depth=depth, engine=engine
    )
    runtime = register_toy_runtime(
        store=store,
        engine=engine,
        copro_control=control,
        proposal_bodies=bodies,
    )
    return runtime, control, resolved


def _drive(runtime, control, experiment, *, terminal_top_k=1):
    launch = prepare_toy_copro_run(
        runtime,
        run_id=f"seedret-{uuid4().hex[:8]}",
        control=control,
        experiment=experiment,
        terminal_top_k=terminal_top_k,
    )
    result_ref = runtime.controller.drive(
        copro_run_request(
            launch,
            controller_identity_hash=runtime.controller.runtime_hash,
        )
    )
    assert result_ref.schema_name == OPTIM_RESULT_SCHEMA
    return OptimResult.model_validate(runtime.store.get(result_ref.reference))


# --- the full run: a seed that beats every proposal is retained -----------


def test_a_full_run_whose_proposals_all_lose_retains_the_seed(
    sqlite_store,
) -> None:
    """The end-to-end gap this closes.

    Fails before the fix at ``_validate_output_candidates``: finalize sliced
    the seed off the top of ranked history and returned it as an accepted
    proposal, whose base is the consumer's root candidate rather than a
    request candidate, so the harness raised "every proposed candidate must
    bind an exact request candidate as its base" and killed the run.
    """
    runtime, control, experiment = _runtime(sqlite_store)

    result = _drive(runtime, control, experiment)

    assert result.terminal_failure is None
    assert result.seed_retained is True
    assert result.proposals == ()
    final = result.step_results[-1]
    assert final.record.status is StepStatus.COMPLETE
    assert final.record.seed_retained is True
    assert final.record.accepted_candidates == ()
    retained = final.record.retained_candidate_ref
    assert retained is not None
    assert retained == candidate_reference(experiment.initial_candidate)


def test_a_run_whose_proposals_win_still_accepts_a_proposal(
    sqlite_store,
) -> None:
    """The fix does not turn every completion into a retained seed.

    Same fixture with a seed that misses the gold token, so a proposal that
    spells it outranks the baseline and is accepted normally.
    """
    experiment = _gold_experiment(
        initial_template="Reply briefly to: {prompt}"
    )
    winners = (f"Answer {{prompt}} with {GOLD}.",) + PLAIN_BODIES[:2]
    runtime, control, experiment = _runtime(
        sqlite_store, bodies=winners, experiment=experiment
    )

    result = _drive(runtime, control, experiment)

    assert result.terminal_failure is None
    assert result.seed_retained is False
    assert len(result.proposals) == 1
    final = result.step_results[-1]
    assert final.record.seed_retained is False
    assert len(final.record.accepted_candidates) == 1


# --- finalize, against real ranked history --------------------------------

# Proposals that DO spell the gold token, so they hit the same exact-match
# anchor as the seed and every occurrence measures 1.0. This is the tie the
# quantized reward scale makes common, produced by real measurement rather
# than by editing a Reward -- ``Reward`` cross-validates its value against
# its scalarized citation total, so a hand-set scalar is not a real reward.
TIE_BODIES = (
    f"Answer {{prompt}} using {GOLD} exactly.",
    f"Respond to {{prompt}} with the token {GOLD}.",
    f"Give {{prompt}} the reply {GOLD} plainly.",
)


def _finalize_request_shape(runtime, control, experiment, *, terminal_top_k=1):
    """Drive one real round and return the finalizing step's request.

    Every attempt in the resulting history is genuinely measured, so it
    carries the Reward, Eval Config, evidence and policy bindings that
    ``CoproAttempt`` cross-validates. The tests then vary only the *order* of
    that history, which is what ``rank_attempt_history``'s stable sort reads.
    """
    launch = prepare_toy_copro_run(
        runtime,
        run_id=f"seedfin-{uuid4().hex[:8]}",
        control=control,
        experiment=experiment,
        terminal_top_k=terminal_top_k,
    )
    bound = runtime.harness.bind_run(launch.run)
    builder = StepRequestBuilder(store=runtime.store)
    first = builder.build_first(
        run=bound,
        adapter_key=COPRO_ADAPTER_KEY,
        initial_candidate=launch.initial_candidate,
        control=control,
    )
    result, result_ref = runtime.harness.run_step(first)
    assert result.status is StepStatus.CONTINUE
    return builder.build_next(
        prior=result,
        prior_ref=result_ref,
        prior_results=(result,),
        control=control,
        mutation_field=str(bound.record.mutation_field),
    )


def _split_seed(attempts, seed_ref):
    seed = [
        item
        for item in attempts
        if candidate_reference(item.candidate.record) == seed_ref
    ]
    others = [
        item
        for item in attempts
        if candidate_reference(item.candidate.record) != seed_ref
    ]
    assert len(seed) == 1, "the seed round must carry the initial candidate"
    return seed[0], others


def _finalize_case(store, *, bodies, terminal_top_k=1):
    """A finalizing request, its adapter, and its real measured attempts."""
    runtime, control, experiment = _runtime(
        store, bodies=bodies, breadth=3, depth=1
    )
    request = _finalize_request_shape(
        runtime, control, experiment, terminal_top_k=terminal_top_k
    )
    seed_ref = request.run.record.initial_candidate_ref
    assert seed_ref is not None
    seed, others = _split_seed(attempt_history_entries(request), seed_ref)
    adapter = runtime.adapter_registry.resolve(COPRO_ADAPTER_KEY)
    return adapter, request, seed, others


def _invoke(adapter, request):
    """Invoke the adapter and hold its output to the harness's own checks."""
    output = adapter.invoke(request, ())
    OptimHarness._validate_output(request, output)
    OptimHarness._validate_output_candidates(request, output)
    return output


def _ranked_as(request, ordered):
    """``request`` with its attempt history restated in the order ``ordered``.

    Ranking is a stable sort on descending reward, so history position is the
    tie-break. Reordering is the honest way to express "the seed came from an
    earlier position than the tying proposal" -- it varies only where each
    occurrence sits, leaving every measured reward, Reward record, evidence
    ref and policy binding exactly as the driven round produced them.

    ``occurrence_ordinal`` is renumbered to match, because it *names* the
    position: COPRO requires history ordinals to be contiguous in evaluation
    order, so a reordered history with its original ordinals would be a
    self-contradictory record rather than a different ranking.
    """
    renumbered = [
        attempt.model_copy(update={"occurrence_ordinal": position})
        for position, attempt in enumerate(ordered)
    ]
    pools = dict(request.pools.to_json())
    pools["attempt_history"] = [
        attempt.model_dump(mode="json") for attempt in renumbered
    ]
    return OptimStepRequest.model_validate(
        {**request.model_dump(mode="json"), "pools": pools}
    )


def test_finalize_retains_a_strictly_top_ranked_seed(sqlite_store) -> None:
    """The seed outscores every proposal, so the run keeps the baseline.

    Fails before the fix: ``ranked[:required]`` handed the seed back as an
    accepted proposal and ``_validate_output_candidates`` rejected it with
    "every proposed candidate must bind an exact request candidate as its
    base".
    """
    adapter, request, seed, others = _finalize_case(
        sqlite_store, bodies=PLAIN_BODIES
    )
    # Real measurement: the gold-spelling seed scores 1.0 and the plain
    # proposals score strictly below it.
    assert seed.reward == 1.0
    assert all(item.reward < 1.0 for item in others)

    output = _invoke(adapter, request)

    assert output.proposed_status is StepStatus.COMPLETE
    assert output.seed_retained is True
    assert output.accepted_candidates == ()
    assert output.proposed_candidates == ()
    assert output.retained_candidate is not None
    assert (
        candidate_reference(output.retained_candidate)
        == request.run.record.initial_candidate_ref
    )


def test_finalize_retains_a_seed_that_merely_ties_the_top(
    sqlite_store,
) -> None:
    """An exact tie at the top terminalizes on the seed, not on a proposal.

    Pins the case the old code comment got wrong. It claimed the seed "loses
    every tie" because the seed round places it last -- but the stable sort
    runs over the *whole* history, so the seed only trails proposals from its
    own round. Order it ahead of the tying proposals, as any later round's
    proposal would leave it, and the seed takes the top rank. Terminal
    selection must still retain it rather than accept it.
    """
    adapter, request, seed, others = _finalize_case(
        sqlite_store, bodies=TIE_BODIES
    )
    # Every occurrence hits the exact-match anchor: a genuine tie.
    assert seed.reward == 1.0
    assert [item.reward for item in others] == [1.0] * len(others)

    built = _ranked_as(request, [seed, *others])
    output = _invoke(adapter, built)

    assert output.proposed_status is StepStatus.COMPLETE
    assert output.seed_retained is True
    assert output.accepted_candidates == ()


def test_a_tying_proposal_ordered_first_is_accepted_normally(
    sqlite_store,
) -> None:
    """The mirror case: the seed ties but does not take the top rank.

    Keeps the tie test from passing for the trivial reason that a tie always
    retains the seed. Here the tie resolves to a proposal, which is accepted.
    """
    adapter, request, seed, others = _finalize_case(
        sqlite_store, bodies=TIE_BODIES
    )

    built = _ranked_as(request, [*others, seed])
    output = _invoke(adapter, built)

    assert output.seed_retained is False
    assert len(output.accepted_candidates) == 1
    assert (
        candidate_reference(output.accepted_candidates[0])
        != request.run.record.initial_candidate_ref
    )


def test_finalize_excludes_a_seed_ranked_below_the_top(sqlite_store) -> None:
    """Seed retention is all-or-nothing; a mid-rank seed is simply dropped.

    At ``terminal_top_k > 1`` a tying seed below rank 0 falls inside the
    naive ``ranked[:required]`` slice and would be handed back as an accepted
    proposal. Excluding it leaves the slice to real proposals.
    """
    adapter, request, seed, others = _finalize_case(
        sqlite_store, bodies=TIE_BODIES, terminal_top_k=2
    )
    assert len(others) >= 2, "need two proposals to rank around the seed"

    # One proposal, then the seed, then the rest: at equal rewards the seed
    # lands at rank 1 -- inside a top-2 slice.
    built = _ranked_as(request, [others[0], seed, *others[1:]])
    output = _invoke(adapter, built)

    assert output.seed_retained is False
    assert len(output.accepted_candidates) == 2
    seed_ref = request.run.record.initial_candidate_ref
    assert all(
        candidate_reference(candidate) != seed_ref
        for candidate in output.accepted_candidates
    )


# --- the same guard on the zero-proposal terminalization path -------------

# ``_terminalize_without_proposals`` selects out of ranked history with the
# same ``ranked[:required]`` shape, so it carried the same latent bug: a
# round that realizes nothing new terminalizes on best-so-far, and
# best-so-far may be the seed.


class _SeedRoundOnlyTransport(DummyProposerTransport):
    """Answers the seed round, then fails every later round's slots.

    Drives a run into ``_terminalize_without_proposals``: round 1 realizes no
    usable proposal at all, so the run terminalizes on what round 0 measured
    rather than dying as a contract violation.
    """

    def draft(self, config, request, count):
        if request.proposal_mode == "seed_proposal":
            return super().draft(config, request, count)
        return tuple(
            ProposalDraft.failure(detail="no instruction available")
            for _ in range(count)
        )


def _seed_only_runtime(store, *, bodies, breadth=3, depth=2, experiment=None):
    resolved = experiment if experiment is not None else _gold_experiment()
    engine = ReferenceEvalRuntimeConfig().build_engine(
        store, experiment=resolved
    )
    control = build_toy_copro_control(
        breadth=breadth, depth=depth, engine=engine
    )
    runtime = register_toy_runtime(
        store=store,
        engine=engine,
        copro_control=control,
        proposer_transport=_SeedRoundOnlyTransport(
            scripted_bodies=bodies,
            execution_policy_hash=engine.execution_policy_identity_hash(),
            prompt_adapter_identity_hash=(
                control.prompt_adapter_identity_hash
            ),
        ),
    )
    return runtime, control, resolved


def test_an_empty_round_retains_a_top_ranked_seed(sqlite_store) -> None:
    """Best-so-far is the seed, so the empty round retains it.

    Fails before the fix the same way finalize did: the empty round returned
    the seed as an accepted proposal and the harness rejected its base.
    """
    runtime, control, experiment = _seed_only_runtime(
        sqlite_store, bodies=PLAIN_BODIES[:2]
    )

    result = _drive(runtime, control, experiment)

    assert result.terminal_failure is None
    assert result.seed_retained is True
    assert result.proposals == ()
    final = result.step_results[-1]
    assert final.record.status is StepStatus.COMPLETE
    assert final.record.seed_retained is True
    assert final.record.accepted_candidates == ()
    assert final.record.retained_candidate_ref == candidate_reference(
        experiment.initial_candidate
    )


def test_an_empty_round_accepts_a_winning_proposal(sqlite_store) -> None:
    """The guard does not swallow a legitimate best-so-far proposal."""
    experiment = _gold_experiment(
        initial_template="Reply briefly to: {prompt}"
    )
    winners = (f"Answer {{prompt}} with {GOLD}.", PLAIN_BODIES[0])
    runtime, control, experiment = _seed_only_runtime(
        sqlite_store, bodies=winners, experiment=experiment
    )

    result = _drive(runtime, control, experiment)

    assert result.terminal_failure is None
    assert result.seed_retained is False
    assert len(result.proposals) == 1
    final = result.step_results[-1]
    assert final.record.seed_retained is False
    assert len(final.record.accepted_candidates) == 1


def test_an_empty_round_excludes_a_seed_below_the_top(sqlite_store) -> None:
    """A mid-rank seed is dropped from the slice, not accepted.

    One proposal ties the gold-spelling seed and the other scores below it,
    so at ``terminal_top_k=2`` the seed lands at rank 1 -- inside the naive
    ``ranked[:2]`` slice. Seed retention is all-or-nothing, so the honest
    outcome is to exclude the seed and fill the slice with both proposals.

    Fails before the fix: the top-2 slice contained the seed, which the
    harness rejected for binding the toy root candidate as its base.
    """
    bodies = (f"Answer {{prompt}} with {GOLD} now.", PLAIN_BODIES[0])
    runtime, control, experiment = _seed_only_runtime(
        sqlite_store, bodies=bodies
    )

    result = _drive(runtime, control, experiment, terminal_top_k=2)

    assert result.terminal_failure is None
    assert result.seed_retained is False
    assert len(result.proposals) == 2
    seed_ref = candidate_reference(experiment.initial_candidate)
    assert all(
        proposal.candidate != seed_ref for proposal in result.proposals
    )
