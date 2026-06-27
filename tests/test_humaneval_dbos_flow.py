from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from dr_dspy import humaneval_dbos_flow as flow
from dr_dspy import humaneval_direct_dbos as direct
from dr_dspy import humaneval_encdec_dbos as encdec


class AnalysisRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    temperature: float
    task_id: str
    score: float
    provider_cost: float | None = None
    raw_compile_ok: bool | None = None
    extracted_compile_ok: bool | None = None


def test_stable_prediction_id_helper_preserves_flow_contracts() -> None:
    direct_id = flow.stable_prediction_id_from_dimensions(
        experiment_name="exp",
        task_id="HumanEval/1",
        dimensions={"model": "model/a", "temperature": 0.0},
        repetition_seed=2,
        digest_length=32,
    )
    encdec_id = flow.stable_prediction_id_from_dimensions(
        experiment_name="exp",
        task_id="HumanEval/1",
        dimensions={
            "encoder_model": "model/a",
            "decoder_model": "model/b",
            "encoder_temperature": 0.0,
            "decoder_temperature": 0.2,
        },
        repetition_seed=2,
    )

    assert direct_id == direct.stable_prediction_id(
        experiment_name="exp",
        task_id="HumanEval/1",
        model="model/a",
        temperature=0.0,
        repetition_seed=2,
    )
    assert len(direct_id) == 32
    assert encdec_id == encdec.stable_prediction_id(
        experiment_name="exp",
        task_id="HumanEval/1",
        encoder_model="model/a",
        decoder_model="model/b",
        encoder_temperature=0.0,
        decoder_temperature=0.2,
        repetition_seed=2,
    )
    assert len(encdec_id) == 64


def test_parse_float_csv_rejects_empty_input() -> None:
    assert flow.parse_float_csv("0, 0.2") == [0.0, 0.2]
    with pytest.raises(ValueError, match="temperature"):
        flow.parse_float_csv(" , ", value_name="temperature")


def test_generation_workflow_enqueues_scoring_after_success() -> None:
    calls: list[tuple[str, Any]] = []

    def generate_prediction(database_url: str, prediction_id: str) -> str:
        calls.append(("generate", (database_url, prediction_id)))
        return "result"

    def record_success(database_url: str, result: str) -> None:
        calls.append(("record_success", (database_url, result)))

    def record_error(*_args: object) -> None:
        raise AssertionError("generation should not fail")

    def enqueue_score(
        database_url: str,
        prediction_id: str,
        experiment_name: str,
        timeout: float,
    ) -> None:
        calls.append(
            (
                "enqueue_score",
                (database_url, prediction_id, experiment_name, timeout),
            )
        )

    def mark_queued(database_url: str, prediction_id: str) -> None:
        calls.append(("mark_queued", (database_url, prediction_id)))

    status = flow.run_generation_workflow(
        database_url="postgresql:///unit",
        prediction_id="pred-1",
        experiment_name="exp",
        score_timeout=7.0,
        generate_prediction=generate_prediction,
        record_generation_success=record_success,
        record_generation_error=record_error,
        enqueue_score=enqueue_score,
        mark_scoring_queued=mark_queued,
    )

    assert status == "generated"
    assert calls == [
        ("generate", ("postgresql:///unit", "pred-1")),
        ("record_success", ("postgresql:///unit", "result")),
        ("enqueue_score", ("postgresql:///unit", "pred-1", "exp", 7.0)),
        ("mark_queued", ("postgresql:///unit", "pred-1")),
    ]


def test_generation_workflow_does_not_enqueue_after_failure() -> None:
    calls: list[tuple[str, Any]] = []

    def generate_prediction(_database_url: str, _prediction_id: str) -> str:
        raise RuntimeError("boom")

    def record_error(
        database_url: str, prediction_id: str, error: str
    ) -> None:
        calls.append(("record_error", (database_url, prediction_id, error)))

    def fail_enqueue(*_args: object) -> None:
        raise AssertionError("scoring should not be enqueued")

    status = flow.run_generation_workflow(
        database_url="postgresql:///unit",
        prediction_id="pred-1",
        experiment_name="exp",
        score_timeout=7.0,
        generate_prediction=generate_prediction,
        record_generation_success=lambda *_args: None,
        record_generation_error=record_error,
        enqueue_score=fail_enqueue,
        mark_scoring_queued=fail_enqueue,
    )

    assert status == "generation_error"
    assert calls == [
        (
            "record_error",
            ("postgresql:///unit", "pred-1", "RuntimeError('boom')"),
        )
    ]


def test_summarize_analysis_records_groups_and_aggregates() -> None:
    records = [
        AnalysisRecord(
            model="model/a",
            temperature=0.0,
            task_id="task/1",
            score=1.0,
            provider_cost=0.01,
            raw_compile_ok=True,
            extracted_compile_ok=True,
        ),
        AnalysisRecord(
            model="model/a",
            temperature=0.0,
            task_id="task/1",
            score=0.0,
            provider_cost=0.03,
            raw_compile_ok=False,
            extracted_compile_ok=True,
        ),
    ]

    summaries = flow.summarize_analysis_records(
        records,
        group_key=lambda record: (record.model, record.temperature),
        model_label=lambda record: record.model,
        temperature=lambda record: record.temperature,
        task_id=lambda record: record.task_id,
        score=lambda record: record.score,
        provider_cost=lambda record: record.provider_cost,
        raw_compile_ok=lambda record: record.raw_compile_ok,
        extracted_compile_ok=lambda record: record.extracted_compile_ok,
        summary_factory=flow.AnalysisSummary,
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.model == "model/a"
    assert summary.sample_count == 1
    assert summary.scored_count == 2
    assert summary.total_price == pytest.approx(0.04)
    assert summary.avg_performance == pytest.approx(0.5)
    assert summary.raw_compile_pass_count == 1
    assert summary.extracted_compile_pass_count == 2
    assert summary.extraction_lift == 1
