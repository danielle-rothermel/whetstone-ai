"""GEPA effect contract and recorder tests."""

from __future__ import annotations

import pytest
from dr_store import ObjectStore, SqliteBackend

from tests.optimization.gepa.support import (
    data_instance,
    evaluation_request,
    evaluation_result,
)
from whetstone.optimization.gepa.contracts import (
    GepaEffectConflictError,
    GepaEffectRecorder,
    GepaEffectTranscript,
    GepaEvaluationRow,
)


def test_effect_recorder_reuses_exact_result_and_rejects_slot_drift(
    tmp_path,
) -> None:
    database = tmp_path / "gepa-effects.sqlite"
    first = GepaEffectRecorder(
        ObjectStore(SqliteBackend(database)),
    )
    request = evaluation_request()
    result = evaluation_result(request)

    first.record_request(request)
    first.record_request(request)
    assert first.record_evaluation_result(request, result) == result

    fresh = GepaEffectRecorder(
        ObjectStore(SqliteBackend(database)),
    )
    fresh.record_request(request)
    assert fresh.load_evaluation_result(request) == result

    drifted = request.model_copy(update={"capture_traces": True})
    with pytest.raises(GepaEffectConflictError, match="ordinal 0"):
        fresh.record_request(drifted)


def test_evaluation_row_rejects_unauditable_success() -> None:
    with pytest.raises(ValueError, match="canonical evidence"):
        GepaEvaluationRow(
            data=data_instance(0),
            output={"answer": "unproven"},
            score=1.0,
        )


def test_recorder_builds_ordered_semantic_effect_transcript(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "transcript.sqlite"))
    recorder = GepaEffectRecorder(store)
    request = evaluation_request()
    result = evaluation_result(request)
    recorder.record_request(request)
    recorder.record_evaluation_result(request, result)

    transcript = recorder.build_transcript(
        context=request.slot.context,
        effect_count=1,
    )
    transcript_ref = recorder.persist_transcript(transcript)

    assert transcript.entries[0].invocation_ordinal == 0
    assert transcript.entries[0].upstream_candidate_index is None
    assert transcript.entries[0].data_ids == tuple(
        item.data_id for item in request.data
    )
    assert (
        GepaEffectTranscript.model_validate(
            store.get(transcript_ref.reference)
        )
        == transcript
    )
