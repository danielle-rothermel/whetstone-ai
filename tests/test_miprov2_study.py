"""Optuna replay determinism from a persisted StudyTranscript (D10)."""

from __future__ import annotations

from unittest.mock import MagicMock

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
from whetstone.optim.contracts import OptimStepResult
from whetstone.optim.harness import OptimHarness
from whetstone.optim.miprov2.adapter import MIPROV2_ADAPTER_KEY, MIPROV2_STATE_KEY
from whetstone.optim.miprov2.control import Miprov2DemoMode
from whetstone.optim.miprov2.runtime import Miprov2Driver, Miprov2State
from whetstone.optim.miprov2.study import (
    StudyTranscript,
    StudyTranscriptMismatch,
)
from whetstone.testing.toy.miprov2 import build_toy_miprov2_control


def _drive_zeroshot(store, *, run_id: str) -> tuple[Miprov2Driver, list[Miprov2State]]:
    engine = ReferenceEvalRuntimeConfig().build_engine(store)
    control = build_toy_miprov2_control(
        engine=engine, demo_mode=Miprov2DemoMode.ZEROSHOT
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
    runtime.controller.drive(
        RunRequest(
            controller_identity_hash=runtime.controller.runtime_hash,
            run_id=run_id,
            control_identity_hash=control.identity_hash(),
        )
    )
    states: list[Miprov2State] = []
    index = 0
    while True:
        key = OptimHarness._result_binding_key(run_id, index)  # noqa: SLF001
        bound = store.resolve(key)
        if bound is None:
            break
        result = OptimStepResult.model_validate(store.get(bound))
        assert result.state_ref is not None
        snapshot = store.get(result.state_ref.reference)
        states.append(
            Miprov2State.model_validate(snapshot[MIPROV2_STATE_KEY])
        )
        index += 1
    return Miprov2Driver(), states


def _open_transcript(
    states: list[Miprov2State],
) -> tuple[Miprov2State, StudyTranscript]:
    for state in states:
        transcript = state.study_transcript
        if transcript is None:
            continue
        if len(transcript.samples) < state.control.num_trials:
            return state, transcript
    raise AssertionError("no mid-study transcript survived the run")


def test_suggest_next_survives_transcript_round_trip(tmp_path) -> None:
    with open_sqlite(str(tmp_path / "d10.sqlite")) as store:
        driver, states = _drive_zeroshot(store, run_id="d10-replay")
        state, transcript = _open_transcript(states)
        study = driver._study(state)  # noqa: SLF001
        live = study.suggest_next(transcript)

        restored = StudyTranscript.model_validate(
            transcript.model_dump(mode="json")
        )
        replayed_study = driver._study(state)  # noqa: SLF001
        replayed = replayed_study.suggest_next(restored)

        assert replayed == live
        assert replayed.trial_number == live.trial_number
        assert replayed.params == live.params


def test_a_drifted_suggestion_is_a_transcript_mismatch(tmp_path) -> None:
    with open_sqlite(str(tmp_path / "d10-mismatch.sqlite")) as store:
        driver, states = _drive_zeroshot(store, run_id="d10-mismatch")
        state, transcript = _open_transcript(states)
        study = driver._study(state)  # noqa: SLF001
        suggestion = study.suggest_next(transcript)
        drifted = suggestion.model_copy(
            update={"trial_number": suggestion.trial_number + 1}
        )

        with pytest.raises(StudyTranscriptMismatch):
            study.record_sample(
                transcript,
                drifted,
                score=0.0,
                evaluation=transcript.baseline.evaluation,
                candidate_assembly=MagicMock(),
            )
