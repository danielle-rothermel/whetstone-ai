"""Replaying MIPROv2's history stays linear in the history it verifies.

``Miprov2State`` verifies itself by replaying its whole run history from
the canonical RNG cursor. That guarantee is deliberate, and these tests
do not relax it -- they pin its *cost shape*. The driver builds many
content-identical states per step, and replaying the same unchanged
content repeatedly turned a linear verification into a quadratic one:
every state construction re-walked the entire history, so a run paid
O(steps^2) replays for O(steps) evidence.

What is asserted here is state, never elapsed time: the number of times
the replay entry point is actually invoked, and which states went
through it.
"""

from __future__ import annotations

import pytest

from dr_store.sync import open_sqlite

import whetstone.optim.miprov2.runtime as runtime_module
from whetstone.coordination.harness_run_controller import RunRequest
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.miprov2.adapter import (
    MIPROV2_ADAPTER_KEY,
    MIPROV2_STATE_KEY,
)
from whetstone.optim.miprov2.control import Miprov2DemoMode
from whetstone.optim.miprov2.runtime import Miprov2State
from whetstone.testing.runtime import (
    build_miprov2_adapter,
    build_toy_copro_control,
    prepare_toy_miprov2_run,
    register_toy_runtime,
)
from whetstone.testing.toy.miprov2 import build_toy_miprov2_control

from tests.test_miprov2_harness_e2e import _Run


class _ReplayLedger:
    """Every state that actually reached the replay entry point."""

    def __init__(self) -> None:
        self.replayed: list[Miprov2State] = []

    @property
    def count(self) -> int:
        return len(self.replayed)

    def verified(self, state: Miprov2State) -> bool:
        return any(seen == state for seen in self.replayed)


@pytest.fixture
def replay_ledger(monkeypatch) -> _ReplayLedger:
    """Count replays, from a cold memo, without changing what replay does."""

    ledger = _ReplayLedger()
    original = runtime_module.replay_miprov2_state

    def counting(state, planning):
        ledger.replayed.append(state)
        return original(state, planning)

    # The memo is process-local and other tests warm it; a cold memo is
    # what makes the count attributable to this run alone.
    runtime_module._VERIFIED_STATES.clear()  # noqa: SLF001
    monkeypatch.setattr(runtime_module, "replay_miprov2_state", counting)
    return ledger


@pytest.fixture
def driven_run(tmp_path, replay_ledger) -> _Run:
    """One toy MIPROv2 run driven to terminal with replays counted."""

    run_id = "replay-cost"
    with open_sqlite(str(tmp_path / "replay-cost.sqlite")) as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(store)
        control = build_toy_miprov2_control(
            engine=engine, demo_mode=Miprov2DemoMode.FEWSHOT
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
        yield _Run(
            store=store,
            runtime=runtime,
            control=control,
            terminal_ref=terminal_ref,
            run_id=run_id,
        )


def _evidence_items(state: Miprov2State) -> int:
    """The history a full replay has to walk, item by item."""

    proposal_evidence = (
        0 if state.proposal_state is None else len(state.proposal_state.evidence)
    )
    samples = (
        0
        if state.study_transcript is None
        else len(state.study_transcript.samples)
    )
    return (
        len(state.bootstrap_evidence)
        + len(state.completed_effects)
        + proposal_evidence
        + samples
    )


def test_replays_stay_linear_in_the_evidence_they_verify(
    driven_run, replay_ledger
) -> None:
    """A driven run replays O(evidence), not O(evidence^2).

    Before the verified-state memo, one 20-step toy run drove 209
    replays against 41 distinct states: every construction re-walked the
    whole history, and step 17 cost roughly sixteen times step 0. The
    bound below is what a linear cost shape looks like -- a small
    constant factor over the evidence the run actually accumulated -- and
    it fails loudly if per-construction replay ever comes back.
    """

    final = driven_run.final_state
    evidence = _evidence_items(final)
    assert evidence > 0, "the run must accumulate evidence to bound anything"

    # Each distinct state a step reaches is replayed once; the constant
    # absorbs the per-step states that carry no new evidence (a pending
    # effect opening and resolving) plus the opening state. This run
    # accumulates 29 evidence items and drives 41 replays, so the bound
    # holds with room while still sitting well under the 209 the same
    # run drove when every construction replayed.
    bound = 2 * evidence + 16
    assert replay_ledger.count <= bound, (
        f"{replay_ledger.count} replays for {evidence} evidence items "
        f"exceeds the linear bound of {bound}; per-construction replay "
        "has returned"
    )


def test_every_persisted_state_was_replay_verified(
    driven_run, replay_ledger
) -> None:
    """The memo removes repeated work, never a verification.

    Every state the run wrote to the store -- including the terminal one
    -- must have gone through the full replay at least once, by content.
    """

    persisted: list[Miprov2State] = []
    for result in driven_run.results:
        assert result.state_ref is not None
        snapshot = driven_run.store.get(result.state_ref.reference)
        persisted.append(
            Miprov2State.model_validate(snapshot[MIPROV2_STATE_KEY])
        )

    assert persisted, "a driven run persists at least one state"
    for index, state in enumerate(persisted):
        assert replay_ledger.verified(state), (
            f"the state persisted at step {index} was never replay-verified"
        )


def test_a_state_carrying_new_evidence_is_always_replayed(
    driven_run, replay_ledger
) -> None:
    """Every distinct evidence length the run reached was replayed.

    The memo keys on full content, so a state that appended evidence
    cannot be answered from a shorter prefix. Pinning it by evidence
    length says the same thing in the terms the guarantee is stated in.
    """

    replayed_lengths = {
        _evidence_items(state) for state in replay_ledger.replayed
    }
    persisted_lengths = set()
    for result in driven_run.results:
        assert result.state_ref is not None
        snapshot = driven_run.store.get(result.state_ref.reference)
        persisted_lengths.add(
            _evidence_items(
                Miprov2State.model_validate(snapshot[MIPROV2_STATE_KEY])
            )
        )

    assert persisted_lengths <= replayed_lengths


def test_the_memo_answers_only_for_content_it_recorded(driven_run) -> None:
    """A state differing anywhere misses the memo and is verified again.

    This is the property the whole optimisation rests on, so it is
    checked directly on a real state rather than inferred from timings:
    recorded content is a hit, and content that differs in any field --
    even one the memo's cheap bucket key does not mention -- is a miss.
    """

    memo = runtime_module._VerifiedStateMemo()  # noqa: SLF001
    state = driven_run.final_state

    assert not memo.verified(state), "an empty memo answers for nothing"
    memo.record(state)
    assert memo.verified(state)

    # Equal content, different object: still a hit. The memo is keyed on
    # content, so a state rebuilt from the store is answered.
    assert memo.verified(
        Miprov2State.model_validate(state.model_dump(mode="json"))
    )

    # A field the bucket key does not mention still forces a miss,
    # because the bucket is only a prefilter for a full equality check.
    assert not memo.verified(_with_changed_failure(state))


def _with_changed_failure(state: Miprov2State) -> Miprov2State:
    """A content-different state, built without re-running validation.

    ``model_copy`` would revalidate and reject the mutation, and the
    point here is only that the memo distinguishes content -- so the
    object is constructed directly.
    """

    altered = object.__new__(type(state))
    fields = dict(state.__dict__)
    fields["failure"] = "not the recorded content"
    object.__setattr__(altered, "__dict__", fields)
    object.__setattr__(
        altered, "__pydantic_fields_set__", set(state.__pydantic_fields_set__)
    )
    object.__setattr__(altered, "__pydantic_extra__", None)
    object.__setattr__(altered, "__pydantic_private__", None)
    return altered
