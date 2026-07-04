from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from tests.test_platform_submission_queue_backoff import (
    DummyEngine,
    _spec,
)
from whetstone.platform import fairness, jsonl_specs, queue_worker, submission
from whetstone.platform.jsonl_specs import JsonlSpecRef
from whetstone.records import BatchSubmitItemInsertStatus, PredictionSpecRecord


def _write_jsonl(path: Path, specs: tuple[PredictionSpecRecord, ...]) -> None:
    lines = [
        spec.model_dump_json() for spec in specs
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_index_jsonl_prediction_specs_rejects_experiment_mismatch(
    tmp_path: Path,
) -> None:
    specs_file = tmp_path / "specs.jsonl"
    _write_jsonl(specs_file, (_spec(task_id="HumanEval/0"),))

    with pytest.raises(
        ValueError,
        match="experiment_name must match submit operation",
    ):
        jsonl_specs.index_jsonl_prediction_specs(
            specs_file,
            experiment_name="other",
        )


def test_index_jsonl_prediction_specs_rejects_duplicate_prediction_id(
    tmp_path: Path,
) -> None:
    spec = _spec(task_id="HumanEval/0")
    specs_file = tmp_path / "specs.jsonl"
    _write_jsonl(specs_file, (spec, spec))

    with pytest.raises(
        ValueError,
        match="duplicate prediction_id in submit operation",
    ):
        jsonl_specs.index_jsonl_prediction_specs(
            specs_file,
            experiment_name="exp",
        )


def test_index_jsonl_prediction_specs_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    specs_file = tmp_path / "specs.jsonl"
    specs_file.write_text("{not json}\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="invalid prediction spec JSON on line 1",
    ):
        jsonl_specs.index_jsonl_prediction_specs(
            specs_file,
            experiment_name="exp",
        )


def test_index_jsonl_prediction_specs_requires_index_fields(
    tmp_path: Path,
) -> None:
    specs_file = tmp_path / "specs.jsonl"
    specs_file.write_text(
        json.dumps({"prediction_id": "abc"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="invalid prediction spec JSON on line 1",
    ):
        jsonl_specs.index_jsonl_prediction_specs(
            specs_file,
            experiment_name="exp",
        )


def test_load_jsonl_prediction_specs_returns_refs_in_fair_order(
    tmp_path: Path,
) -> None:
    specs = (
        _spec(task_id="HumanEval/2"),
        _spec(task_id="HumanEval/0"),
        _spec(task_id="HumanEval/1"),
    )
    specs_file = tmp_path / "specs.jsonl"
    _write_jsonl(specs_file, specs)
    refs = jsonl_specs.index_jsonl_prediction_specs(
        specs_file,
        experiment_name="exp",
    )
    ordered_refs = tuple(
        sorted(refs, key=lambda ref: (ref.fair_order_key, ref.prediction_id))
    )

    loaded = jsonl_specs.load_jsonl_prediction_specs(specs_file, ordered_refs)

    assert [spec.task_id for spec in loaded] == [
        spec.task_id for spec in sorted(
            specs,
            key=lambda item: (item.fair_order_key, item.prediction_id),
        )
    ]


def test_load_jsonl_prediction_specs_surfaces_validation_errors(
    tmp_path: Path,
) -> None:
    spec = _spec(task_id="HumanEval/0")
    payload = json.loads(spec.model_dump_json())
    payload["fair_order_key"] = "wrong-key"
    specs_file = tmp_path / "specs.jsonl"
    specs_file.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    refs = jsonl_specs.index_jsonl_prediction_specs(
        specs_file,
        experiment_name="exp",
    )

    with pytest.raises(
        ValueError,
        match="invalid prediction spec JSON at byte offset",
    ):
        jsonl_specs.load_jsonl_prediction_specs(specs_file, refs)


def test_fair_ordered_spec_ref_windows_interleaves_model_axis() -> None:
    specs = tuple(
        _spec(task_id=f"HumanEval/{task_id}", model=model)
        for model in ("model-a", "model-b")
        for task_id in range(8)
    )
    refs = tuple(
        JsonlSpecRef(
            fair_order_key=spec.fair_order_key,
            prediction_id=spec.prediction_id,
            byte_offset=index,
        )
        for index, spec in enumerate(specs)
    )

    ordered = tuple(
        window
        for window in fairness.fair_ordered_spec_ref_windows(
            refs,
            window_size=4,
        )
    )

    assert len(ordered) == 4
    first_window_keys = {ref.fair_order_key for ref in ordered[0]}
    assert len(first_window_keys) > 1


def test_submit_prediction_specs_jsonl_chunks_persistence_and_enqueue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    specs = tuple(
        _spec(task_id=f"HumanEval/{task_id}")
        for task_id in range(5)
    )
    specs_file = tmp_path / "specs.jsonl"
    _write_jsonl(specs_file, specs)
    engine = DummyEngine()
    chunks: list[int] = []
    max_in_flight_specs = 0
    current_in_flight_specs = 0
    pending_ids = {spec.prediction_id for spec in specs}
    by_id = {spec.prediction_id: spec for spec in specs}

    def prepare(
        connection: object,
        *,
        operation_key: str,
        experiment_name: str,
        ordered_specs: tuple[PredictionSpecRecord, ...],
        submit_spec: dict[str, object] | None,
        metadata: dict[str, object] | None,
        chunk_size: int,
        item_index_offset: int,
    ) -> None:
        nonlocal max_in_flight_specs, current_in_flight_specs
        chunks.append(len(ordered_specs))
        current_in_flight_specs = len(ordered_specs)
        max_in_flight_specs = max(max_in_flight_specs, current_in_flight_specs)

    def candidates(
        connection: object,
        *,
        operation_key: str,
        limit: int,
    ) -> tuple[submission.EnqueueCandidate, ...]:
        prediction_ids = tuple(
            spec.prediction_id
            for spec in sorted(
                specs,
                key=lambda spec: (spec.fair_order_key, spec.prediction_id),
            )
            if spec.prediction_id in pending_ids
        )[:limit]
        return tuple(
            submission.EnqueueCandidate(
                prediction_id=prediction_id,
                fair_order_key=by_id[prediction_id].fair_order_key,
                item_index=index,
                insert_status=BatchSubmitItemInsertStatus.INSERTED,
            )
            for index, prediction_id in enumerate(prediction_ids)
        )

    def enqueue(
        database_url: str,
        prediction_id: str,
        attempt_index: int,
        queue_name: str,
    ) -> queue_worker.EnqueuedPredictionWorkflow:
        return queue_worker.EnqueuedPredictionWorkflow(
            prediction_id=prediction_id,
            generation_run_id="run",
            workflow_id=f"workflow:{prediction_id}",
            enqueued=True,
        )

    def discard_pending(
        connection: object,
        *,
        operation_key: str,
        item: submission.SubmittedPredictionItem,
        **kwargs: object,
    ) -> None:
        pending_ids.discard(item.prediction_id)

    monkeypatch.setattr(submission, "prepare_submission_records", prepare)
    monkeypatch.setattr(
        submission,
        "load_pending_enqueue_candidates",
        candidates,
    )
    monkeypatch.setattr(
        submission,
        "update_batch_item_outcome",
        discard_pending,
    )
    monkeypatch.setattr(
        submission,
        "update_operation_summary",
        lambda connection, *, operation_key, experiment_name, queue_name: (
            submission.SubmitPredictionSpecsResult(
                operation_key=operation_key,
                experiment_name=experiment_name,
                queue_name=queue_name,
                requested_count=len(specs),
                inserted_count=len(specs),
                already_present_count=0,
                enqueued_count=len(specs),
                already_scheduled_count=0,
                failed_count=0,
            )
        ),
    )

    submission.submit_prediction_specs_jsonl(
        cast(Any, engine),
        database_url="postgresql://example/db",
        operation_key="op-1",
        experiment_name="exp",
        specs_file=specs_file,
        chunk_size=2,
        enqueue_workflow=enqueue,
    )

    assert chunks == [2, 2, 1]
    assert max_in_flight_specs <= 2
