from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from dr_dspy import humaneval_dbos_flow as flow
from dr_dspy.eval_reporting import (
    StatusDimension,
    analysis_markdown,
    repair_plan_line,
    write_analysis_csv,
)


class Summary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimensions: dict[str, Any]
    sample_count: int = 1
    scored_count: int = 1
    total_price: float | None = None
    avg_price_per_sample: float | None = None
    price_variance: float | None = None
    avg_performance: float = 0.0
    performance_variance: float | None = None
    avg_repetition_variance: float | None = None
    raw_compile_pass_count: int = 0
    extracted_compile_pass_count: int = 0
    extraction_lift: int = 0
    avg_raw_compression_ratio: float | None = None
    avg_best_compression_ratio: float | None = None
    avg_best_compression_percent_reduction: float | None = None


class Record(BaseModel):
    model: str
    temperature: float
    task_id: str
    score: float


DIMENSIONS = [
    StatusDimension(key="model", title="Model"),
    StatusDimension(key="temperature", title="Temp", justify="right"),
]


def test_markdown_has_dimension_headers_and_values() -> None:
    summaries = [
        Summary(
            dimensions={"model": "gpt", "temperature": 0.0},
            avg_performance=1.0,
        )
    ]
    text = analysis_markdown(
        experiment_name="exp", summaries=summaries, dimensions=DIMENSIONS
    )
    assert "| Model | Temp |" in text
    assert "| gpt | 0 |" in text
    assert "Total" in text


def test_write_csv_flattens_dimensions(tmp_path: Path) -> None:
    summaries = [
        Summary(dimensions={"model": "gpt", "temperature": 0.0}),
        Summary(dimensions={"model": "kimi", "temperature": 0.5}),
    ]
    csv_path = tmp_path / "out.csv"
    fieldnames = [
        "model",
        "temperature",
        *(n for n in Summary.model_fields if n != "dimensions"),
    ]
    write_analysis_csv(
        summaries,
        csv_path=csv_path,
        fieldnames=fieldnames,
    )
    lines = csv_path.read_text().splitlines()
    assert lines[0].startswith("model,temperature,sample_count,scored_count")
    assert lines[1].startswith("gpt,0.0,1,1,")


def test_repair_plan_line_reports_retry_categories() -> None:
    line = repair_plan_line(
        experiment_name="exp",
        gen_stranded=1,
        gen_errors=2,
        gen_legacy_errors=1,
        gen_recoverable_errors=1,
        gen_excluded_errors=3,
        score_pending=4,
        score_stranded=5,
        score_errors=6,
        score_legacy_errors=2,
        score_recoverable_errors=4,
        score_excluded_errors=7,
        apply=False,
    )

    assert "gen_retry=" in line
    assert "legacy=1" in line
    assert "rec=1" in line
    assert "skip=3" in line
    assert "score_retry=" in line
    assert "legacy=2" in line
    assert "rec=4" in line
    assert "skip=7" in line


def test_summarize_groups_and_carries_dimensions() -> None:
    records = [
        Record(model="m", temperature=0.0, task_id="a", score=1.0),
        Record(model="m", temperature=0.0, task_id="b", score=0.0),
        Record(model="m", temperature=0.5, task_id="a", score=1.0),
    ]
    summaries = flow.summarize_analysis_records(
        records,
        group_key=lambda r: (r.model, r.temperature),
        dimension_values=lambda r: {
            "model": r.model,
            "temperature": r.temperature,
        },
        task_id=lambda r: r.task_id,
        score=lambda r: r.score,
        provider_cost=lambda _r: None,
        raw_compile_ok=lambda _r: None,
        extracted_compile_ok=lambda _r: None,
        summary_factory=Summary,
    )
    assert len(summaries) == 2
    first = summaries[0]
    assert first.dimensions == {"model": "m", "temperature": 0.0}
    assert first.scored_count == 2
    assert first.sample_count == 2
    assert first.avg_performance == 0.5
