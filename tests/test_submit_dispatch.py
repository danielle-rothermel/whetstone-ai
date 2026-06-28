from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from dr_dspy import batch_operation
from dr_dspy import humaneval_direct_dbos as direct
from dr_dspy import humaneval_encdec_dbos as encdec
from dr_dspy.dbos_runtime import EnqueueWorkflowsResult
from dr_dspy.lm_utils import ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(path: Path, module_name: str) -> None:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _submit_progress(
    *,
    total_items: int,
    next_offset: int,
    batch_size: int,
) -> batch_operation.BatchOperationProgress:
    return batch_operation.BatchOperationProgress(
        operation_kind=batch_operation.BatchOperationKind.SUBMIT,
        operation_key="op-key",
        experiment_name="exp",
        script_kind="script",
        workflow_id="workflow",
        attempt=1,
        status=batch_operation.BatchOperationStatus.PENDING,
        total_items=total_items,
        next_offset=next_offset,
        metadata={"submission_id": "sub", "batch_size": batch_size},
        processed_count=0,
        inserted_count=0,
        enqueued_count=0,
        existing_workflow_count=0,
        marked_count=0,
        batch_count=0,
        counters={},
        last_error=None,
        log_file="/tmp/submit.log",
    )


def _direct_samples(count: int) -> list[direct.HumanEvalSample]:
    return [
        direct.HumanEvalSample(
            task_id=f"task/{index}",
            sample_index=index,
            prompt="def f():\n    pass\n",
            canonical_solution="",
            ground_truth_code="def f():\n    return 1\n",
            test="def check(candidate):\n    pass\n",
            entry_point="f",
        )
        for index in range(count)
    ]


def _encdec_samples(count: int) -> list[encdec.EncDecSample]:
    return [
        encdec.EncDecSample(
            task_id=f"task/{index}",
            sample_index=index,
            prompt="def f():\n    pass\n",
            canonical_solution="",
            ground_truth_code="def f():\n    return 1\n",
            test="def check(candidate):\n    pass\n",
            entry_point="f",
        )
        for index in range(count)
    ]


def test_direct_offset_jobs_are_stable_across_chunk_sizes() -> None:
    _load_script(
        REPO_ROOT / "scripts" / "humaneval_dspy_eval_only_dbos_v0.py",
        "direct_script_for_submit_dispatch_test",
    )
    samples = [
        direct.HumanEvalSample(
            task_id=f"task/{index}",
            sample_index=index,
            prompt="def f():\n    pass\n",
            canonical_solution="",
            ground_truth_code="def f():\n    return 1\n",
            test="def check(candidate):\n    pass\n",
            entry_point="f",
        )
        for index in range(2)
    ]
    models = [
        ModelConfig(model="model/a", reasoning={}),
        ModelConfig(model="model/b", reasoning={"effort": "low"}),
    ]
    spec = direct.build_submit_spec(
        experiment_name="exp",
        seed=0,
        sample_count=len(samples),
        model_configs=models,
        temperatures=[0.0, 0.5],
        repetitions=2,
        score_timeout=10.0,
    )
    one_chunk = direct.build_prediction_jobs_for_offsets(
        spec=spec,
        submission_id="sub",
        samples=samples,
        start_offset=0,
        limit=spec.total_jobs(),
    )
    chunked = []
    for offset in range(0, spec.total_jobs(), 3):
        chunked.extend(
            direct.build_prediction_jobs_for_offsets(
                spec=spec,
                submission_id="sub",
                samples=samples,
                start_offset=offset,
                limit=3,
            )
        )
    assert [job.model_dump() for job in chunked] == [
        job.model_dump() for job in one_chunk
    ]


def test_encdec_offset_jobs_are_stable_across_chunk_sizes(
    encdec_configured: None,
) -> None:
    samples = [
        encdec.EncDecSample(
            task_id=f"task/{index}",
            sample_index=index,
            prompt="def f():\n    pass\n",
            canonical_solution="",
            ground_truth_code="def f():\n    return 1\n",
            test="def check(candidate):\n    pass\n",
            entry_point="f",
        )
        for index in range(2)
    ]
    pairs = [
        encdec.EncDecPair(
            encoder=ModelConfig(model="model/a", reasoning={}),
            decoder=ModelConfig(model="model/a", reasoning={}),
        ),
        encdec.EncDecPair(
            encoder=ModelConfig(model="model/b", reasoning={"x": 1}),
            decoder=ModelConfig(model="model/c", reasoning={"y": 2}),
        ),
    ]
    spec = encdec.build_submit_spec(
        experiment_name="exp",
        seed=0,
        sample_count=len(samples),
        model_pairs=pairs,
        encoder_temperatures=[0.0, 0.5],
        decoder_temperatures=[0.0],
        budget_ratios=[None, 0.5],
        repetitions=2,
        score_timeout=10.0,
    )
    one_chunk = encdec.build_prediction_jobs_for_offsets(
        spec=spec,
        submission_id="sub",
        samples=samples,
        start_offset=0,
        limit=spec.total_jobs(),
    )
    chunked = []
    for offset in range(0, spec.total_jobs(), 5):
        chunked.extend(
            encdec.build_prediction_jobs_for_offsets(
                spec=spec,
                submission_id="sub",
                samples=samples,
                start_offset=offset,
                limit=5,
            )
        )
    assert [job.model_dump() for job in chunked] == [
        job.model_dump() for job in one_chunk
    ]


def test_encdec_default_submit_plan_total_is_55104(
    encdec_configured: None,
) -> None:
    config = encdec.experiment_config()
    spec = encdec.build_submit_spec(
        experiment_name="exp",
        seed=config.default_seed,
        sample_count=config.default_sample_count,
        model_pairs=config.default_model_pairs,
        encoder_temperatures=config.default_encoder_temperatures,
        decoder_temperatures=config.default_decoder_temperatures,
        budget_ratios=config.default_budget_ratios,
        repetitions=config.default_repetitions,
        score_timeout=config.default_subprocess_timeout,
    )
    assert spec.total_jobs() == 55104


def test_submit_key_is_deterministic_and_changes_with_axes(
    encdec_configured: None,
) -> None:
    config = encdec.experiment_config()
    spec = encdec.build_submit_spec(
        experiment_name="exp",
        seed=config.default_seed,
        sample_count=1,
        model_pairs=config.default_model_pairs[:1],
        encoder_temperatures=config.default_encoder_temperatures,
        decoder_temperatures=config.default_decoder_temperatures,
        budget_ratios=[None],
        repetitions=1,
        score_timeout=config.default_subprocess_timeout,
    )
    changed = spec.model_copy(update={"budget_ratios": [0.5]})
    assert batch_operation.operation_key(
        spec.model_dump(mode="json")
    ) == batch_operation.operation_key(spec.model_dump(mode="json"))
    assert batch_operation.operation_key(
        spec.model_dump(mode="json")
    ) != batch_operation.operation_key(changed.model_dump(mode="json"))


@pytest.mark.parametrize(
    ("start_offset", "limit", "total_items", "items_per_group", "window"),
    [
        (0, 1, 10, 4, (0, 1)),
        (2, 4, 10, 4, (0, 2)),
        (8, 10, 10, 4, (2, 1)),
        (10, 1, 10, 4, (0, 0)),
        (0, 0, 10, 4, (0, 0)),
    ],
)
def test_operation_item_window_selects_touched_sample_indexes(
    start_offset: int,
    limit: int,
    total_items: int,
    items_per_group: int,
    window: tuple[int, int],
) -> None:
    result = batch_operation.operation_item_window(
        start_offset=start_offset,
        limit=limit,
        total_items=total_items,
        items_per_group=items_per_group,
    )

    assert (result.start_index, result.item_count) == window


def test_direct_offset_jobs_accept_only_needed_sample_window() -> None:
    _load_script(
        REPO_ROOT / "scripts" / "humaneval_dspy_eval_only_dbos_v0.py",
        "direct_script_for_sparse_submit_dispatch_test",
    )
    samples = _direct_samples(3)
    spec = direct.build_submit_spec(
        experiment_name="exp",
        seed=0,
        sample_count=len(samples),
        model_configs=[ModelConfig(model="model/a", reasoning={})],
        temperatures=[0.0],
        repetitions=2,
        score_timeout=10.0,
    )

    expected = direct.build_prediction_jobs_for_offsets(
        spec=spec,
        submission_id="sub",
        samples=samples,
        start_offset=1,
        limit=3,
    )
    sparse = direct.build_prediction_jobs_for_offsets(
        spec=spec,
        submission_id="sub",
        samples=samples[:2],
        start_offset=1,
        limit=3,
    )

    assert [job.model_dump() for job in sparse] == [
        job.model_dump() for job in expected
    ]


def test_encdec_offset_jobs_accept_only_needed_sample_window(
    encdec_configured: None,
) -> None:
    samples = _encdec_samples(3)
    pair = encdec.EncDecPair(
        encoder=ModelConfig(model="model/a", reasoning={}),
        decoder=ModelConfig(model="model/a", reasoning={}),
    )
    spec = encdec.build_submit_spec(
        experiment_name="exp",
        seed=0,
        sample_count=len(samples),
        model_pairs=[pair],
        encoder_temperatures=[0.0],
        decoder_temperatures=[0.0],
        budget_ratios=[None, 0.5],
        repetitions=1,
        score_timeout=10.0,
    )

    expected = encdec.build_prediction_jobs_for_offsets(
        spec=spec,
        submission_id="sub",
        samples=samples,
        start_offset=1,
        limit=3,
    )
    sparse = encdec.build_prediction_jobs_for_offsets(
        spec=spec,
        submission_id="sub",
        samples=samples[:2],
        start_offset=1,
        limit=3,
    )

    assert [job.model_dump() for job in sparse] == [
        job.model_dump() for job in expected
    ]


def test_direct_submit_batch_reads_manifest_not_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _load_script(
        REPO_ROOT / "scripts" / "humaneval_dspy_eval_only_dbos_v0.py",
        "direct_script_for_manifest_submit_batch_test",
    )
    samples = _direct_samples(3)
    spec = direct.build_submit_spec(
        experiment_name="exp",
        seed=0,
        sample_count=len(samples),
        model_configs=[ModelConfig(model="model/a", reasoning={})],
        temperatures=[0.0],
        repetitions=2,
        score_timeout=10.0,
    )
    progress = _submit_progress(
        total_items=spec.total_jobs(), next_offset=1, batch_size=3
    )
    fetched_items: list[dict[str, Any]] = []
    recorded_results: list[batch_operation.BatchOperationResult] = []

    def fetch_items(*_args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        assert kwargs["start_index"] == 0
        assert kwargs["limit"] == 2
        fetched_items.extend(
            sample.model_dump(mode="json") for sample in samples[:2]
        )
        return fetched_items

    def load_dataset(**_kwargs: Any) -> list[direct.HumanEvalSample]:
        raise AssertionError("dataset should not load")

    def fetch_progress(*_args: Any, **_kwargs: Any) -> (
        batch_operation.BatchOperationProgress
    ):
        return progress

    monkeypatch.setattr(
        direct,
        "build_humaneval_samples",
        load_dataset,
    )
    monkeypatch.setattr(
        batch_operation,
        "fetch_operation_progress",
        fetch_progress,
    )
    monkeypatch.setattr(
        batch_operation,
        "fetch_operation_spec",
        lambda *_args, **_kwargs: spec.model_dump(mode="json"),
    )
    monkeypatch.setattr(batch_operation, "fetch_operation_items", fetch_items)
    monkeypatch.setattr(
        direct, "_emit_submit_log", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        direct, "insert_prediction_jobs", lambda _url, jobs: len(jobs)
    )
    monkeypatch.setattr(
        direct,
        "_enqueue_generation_jobs",
        lambda _url, jobs, *, score_timeout: EnqueueWorkflowsResult(
            enqueued=len(jobs), existing=0
        ),
    )
    monkeypatch.setattr(
        batch_operation,
        "record_operation_batch_success",
        lambda *_args, **kwargs: recorded_results.append(kwargs["result"]),
    )

    result = direct.submit_batch_step("postgres://example", "op-key")

    assert result.processed == 3
    assert result.next_offset == 4
    assert recorded_results == [result]


def test_encdec_submit_batch_reads_manifest_not_dataset(
    monkeypatch: pytest.MonkeyPatch,
    encdec_configured: None,
) -> None:
    samples = _encdec_samples(3)
    pair = encdec.EncDecPair(
        encoder=ModelConfig(model="model/a", reasoning={}),
        decoder=ModelConfig(model="model/a", reasoning={}),
    )
    spec = encdec.build_submit_spec(
        experiment_name="exp",
        seed=0,
        sample_count=len(samples),
        model_pairs=[pair],
        encoder_temperatures=[0.0],
        decoder_temperatures=[0.0],
        budget_ratios=[None, 0.5],
        repetitions=1,
        score_timeout=10.0,
    )
    progress = _submit_progress(
        total_items=spec.total_jobs(), next_offset=1, batch_size=3
    )
    fetched_items: list[dict[str, Any]] = []
    recorded_results: list[batch_operation.BatchOperationResult] = []

    def fetch_items(*_args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        assert kwargs["start_index"] == 0
        assert kwargs["limit"] == 2
        fetched_items.extend(
            sample.model_dump(mode="json") for sample in samples[:2]
        )
        return fetched_items

    def load_dataset(**_kwargs: Any) -> list[encdec.EncDecSample]:
        raise AssertionError("dataset should not load")

    def fetch_progress(*_args: Any, **_kwargs: Any) -> (
        batch_operation.BatchOperationProgress
    ):
        return progress

    monkeypatch.setattr(
        encdec,
        "build_humaneval_samples",
        load_dataset,
    )
    monkeypatch.setattr(
        batch_operation,
        "fetch_operation_progress",
        fetch_progress,
    )
    monkeypatch.setattr(
        batch_operation,
        "fetch_operation_spec",
        lambda *_args, **_kwargs: spec.model_dump(mode="json"),
    )
    monkeypatch.setattr(batch_operation, "fetch_operation_items", fetch_items)
    monkeypatch.setattr(
        encdec, "_emit_submit_log", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        encdec, "insert_prediction_jobs", lambda _url, jobs: len(jobs)
    )
    monkeypatch.setattr(
        encdec,
        "_enqueue_generation_jobs",
        lambda _url, jobs, *, score_timeout: EnqueueWorkflowsResult(
            enqueued=len(jobs), existing=0
        ),
    )
    monkeypatch.setattr(
        batch_operation,
        "record_operation_batch_success",
        lambda *_args, **kwargs: recorded_results.append(kwargs["result"]),
    )

    result = encdec.submit_batch_step("postgres://example", "op-key")

    assert result.processed == 3
    assert result.next_offset == 4
    assert recorded_results == [result]
