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


def _drive(
    store,
    *,
    run_id: str,
    demo_mode: Miprov2DemoMode,
) -> tuple[Miprov2Driver, list[Miprov2State]]:
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


def _final_transcript(states: list[Miprov2State]) -> StudyTranscript:
    for state in reversed(states):
        transcript = state.study_transcript
        if transcript is not None and transcript.samples:
            return transcript
    raise AssertionError("no study transcript with samples survived the run")


def test_suggest_next_matches_the_live_runs_next_sample(tmp_path) -> None:
    with open_sqlite(str(tmp_path / "d10.sqlite")) as store:
        driver, states = _drive(
            store, run_id="d10-replay", demo_mode=Miprov2DemoMode.FEWSHOT
        )
        transcript = _final_transcript(states)
        assert transcript.demo_pool_identity_hashes is not None
        assert len(transcript.samples) >= 2
        prefix = transcript.model_copy(update={"samples": transcript.samples[:1]})
        recorded = transcript.samples[1]
        assert any(
            name.endswith("_predictor_demos") for name, _value in recorded.params
        )

        study = driver._study(states[-1])  # noqa: SLF001
        replayed = study.suggest_next(prefix)
        assert replayed.trial_number == recorded.trial_number
        assert replayed.params == recorded.params
        assert (
            replayed.candidate_combination_identity_hash
            == recorded.candidate_combination_identity_hash
        )


def test_a_drifted_suggestion_is_a_transcript_mismatch(tmp_path) -> None:
    with open_sqlite(str(tmp_path / "d10-mismatch.sqlite")) as store:
        driver, states = _drive(
            store, run_id="d10-mismatch", demo_mode=Miprov2DemoMode.FEWSHOT
        )
        transcript = _final_transcript(states)
        prefix = transcript.model_copy(update={"samples": transcript.samples[:1]})
        study = driver._study(states[-1])  # noqa: SLF001
        suggestion = study.suggest_next(prefix)
        drifted = suggestion.model_copy(
            update={"trial_number": suggestion.trial_number + 1}
        )

        with pytest.raises(StudyTranscriptMismatch):
            study.record_sample(
                prefix,
                drifted,
                score=0.0,
                evaluation=transcript.baseline.evaluation,
                candidate_assembly=MagicMock(),
            )
