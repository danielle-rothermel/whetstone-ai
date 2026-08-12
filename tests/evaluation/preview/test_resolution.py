from __future__ import annotations

from dr_store import MemoryBackend, ObjectStore

from tests.evaluation.support import _binding, _engine
from whetstone.evaluation.preview.resolution import (
    build_evaluation_intent,
    build_measured_resolution,
    evaluate_and_resolve,
)
from whetstone.optimization.contracts import IntentOutcome, ResolutionClass


def test_evaluate_and_resolve_returns_measured_resolution(tmp_path) -> None:
    store = ObjectStore(MemoryBackend())
    engine = _engine(tmp_path, store=store)
    binding = _binding(engine)
    candidate = engine.experiment.initial_candidate

    evaluated, resolution = evaluate_and_resolve(
        engine,
        binding,
        candidate,
        purpose="preview-resolution-test",
        run_id="run-1",
        step_index=2,
        occurrence_ordinal=3,
        message="measured in unit test",
    )

    assert evaluated.evidence.reward_ref is not None
    assert resolution.outcome is IntentOutcome.COMPLETED
    assert resolution.detail.classification is ResolutionClass.MEASURED
    assert resolution.intent.intent_id.startswith("run-1:2:3:")


def test_build_helpers_match_evaluate_and_resolve(tmp_path) -> None:
    store = ObjectStore(MemoryBackend())
    engine = _engine(tmp_path, store=store)
    binding = _binding(engine)
    candidate = engine.experiment.initial_candidate

    evaluated, resolution = evaluate_and_resolve(
        engine,
        binding,
        candidate,
        purpose="preview-resolution-test",
        run_id="run-1",
        step_index=0,
        occurrence_ordinal=1,
        message="measured in unit test",
    )
    intent = build_evaluation_intent(
        evaluated,
        binding,
        purpose="preview-resolution-test",
        run_id="run-1",
        step_index=0,
        occurrence_ordinal=1,
    )
    rebuilt = build_measured_resolution(
        evaluated,
        intent,
        message="measured in unit test",
    )

    assert rebuilt.intent == resolution.intent
    assert rebuilt.outcome == resolution.outcome
    assert rebuilt.detail == resolution.detail
