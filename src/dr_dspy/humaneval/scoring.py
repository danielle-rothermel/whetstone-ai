"""Pure HumanEval scoring primitives.

`GeneratedCodeOutcome` is part of the primitive score contract so later
append-only score attempts can persist why a generation scored zero without
parsing error text. The current v0 experiment writers still persist their
legacy scoring columns only; wiring this outcome into durable score-attempt
records belongs to the schema/scoring-profile stage.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from dr_dspy.humaneval.code_extraction import (
    apply_cleaning,
    validate_python_source,
)
from dr_dspy.humaneval.compression import (
    CompressionMetric,
    CompressionMetrics,
    compression_metrics,
)
from dr_dspy.humaneval.task import (
    EvaluationTaskResult,
    HumanEvalTask,
    evaluate_human_eval_code,
)


class GeneratedCodeOutcome(StrEnum):
    PASSED = "passed"
    TESTS_FAILED = "tests_failed"
    EMPTY_GENERATION = "empty_generation"
    EXTRACTION_FAILED = "extraction_failed"
    NO_TOP_LEVEL_FUNCTIONS = "no_top_level_functions"


class GeneratedCodeScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: GeneratedCodeOutcome
    score: float
    error: str | None
    raw_code: str | None = None
    raw_compile_ok: bool
    raw_compile_error: str | None = None
    extraction_candidate_count: int
    selected_candidate_index: int | None = None
    extracted_compile_ok: bool
    extracted_compile_error: str | None = None
    extraction_error: str | None = None
    evaluation: EvaluationTaskResult | None = None


class HumanEvalScoreResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: str
    score: float
    error: str | None
    generated_code_outcome: GeneratedCodeOutcome
    raw_code: str | None = None
    raw_compile_ok: bool
    raw_compile_error: str | None = None
    extraction_candidate_count: int
    selected_candidate_index: int | None = None
    extracted_compile_ok: bool
    extracted_compile_error: str | None = None
    extraction_error: str | None = None
    evaluation_function_names: list[str] = Field(default_factory=list)
    evaluation_total_cases: int | None = None
    evaluation_failure_count: int | None = None
    evaluation_status_counts: dict[str, int] = Field(default_factory=dict)
    compression_metrics: CompressionMetrics = Field(default_factory=dict)
    raw_compression_ratio: float | None = None
    best_compression_ratio: float | None = None
    best_compression_percent_reduction: float | None = None


def best_compression_metric(
    metrics: CompressionMetrics,
) -> CompressionMetric | None:
    comparable = [
        metric
        for metric in metrics.values()
        if metric.ratio_to_ground_truth is not None
    ]
    if not comparable:
        return None
    return min(
        comparable,
        key=lambda metric: metric.ratio_to_ground_truth or 0,
    )


def score_generated_code_for_humaneval(
    *,
    raw_generation: str,
    task: HumanEvalTask,
    timeout: float,
) -> GeneratedCodeScore:
    if not raw_generation.strip():
        extraction_error = "empty raw generation"
        return GeneratedCodeScore(
            outcome=GeneratedCodeOutcome.EMPTY_GENERATION,
            score=0.0,
            error=extraction_error,
            raw_code=None,
            raw_compile_ok=False,
            raw_compile_error=extraction_error,
            extraction_candidate_count=0,
            selected_candidate_index=None,
            extracted_compile_ok=False,
            extracted_compile_error=None,
            extraction_error=extraction_error,
        )

    raw_validation = validate_python_source(raw_generation)
    candidates = apply_cleaning(raw_generation, apply_dedent=True)
    selected_code: str | None = None
    selected_index: int | None = None
    extracted_compile_error: str | None = None
    for index, candidate in enumerate(candidates):
        candidate_validation = validate_python_source(candidate)
        if candidate_validation.compile_ok:
            selected_code = candidate
            selected_index = index
            extracted_compile_error = None
            break
        if extracted_compile_error is None:
            extracted_compile_error = candidate_validation.compile_error

    if selected_code is None:
        extraction_error = (
            "no code candidates extracted"
            if not candidates
            else "no compilable extracted candidate"
        )
        return GeneratedCodeScore(
            outcome=GeneratedCodeOutcome.EXTRACTION_FAILED,
            score=0.0,
            error=extraction_error,
            raw_code=None,
            raw_compile_ok=raw_validation.compile_ok,
            raw_compile_error=raw_validation.compile_error,
            extraction_candidate_count=len(candidates),
            selected_candidate_index=None,
            extracted_compile_ok=False,
            extracted_compile_error=extracted_compile_error,
            extraction_error=extraction_error,
        )

    evaluation = evaluate_human_eval_code(
        task=task,
        candidate_code=selected_code,
        timeout_seconds=timeout,
    )
    outcome = (
        GeneratedCodeOutcome.PASSED
        if evaluation.passed
        else GeneratedCodeOutcome.TESTS_FAILED
    )
    error = None if evaluation.passed else "HumanEval tests failed"
    if not evaluation.function_names:
        outcome = GeneratedCodeOutcome.NO_TOP_LEVEL_FUNCTIONS
        error = "no top-level candidate functions"
    return GeneratedCodeScore(
        outcome=outcome,
        score=1.0 if evaluation.passed else 0.0,
        error=error,
        raw_code=selected_code,
        raw_compile_ok=raw_validation.compile_ok,
        raw_compile_error=raw_validation.compile_error,
        extraction_candidate_count=len(candidates),
        selected_candidate_index=selected_index,
        extracted_compile_ok=True,
        extracted_compile_error=None,
        extraction_error=None,
        evaluation=evaluation,
    )


def score_humaneval_prediction(
    *,
    prediction_id: str,
    raw_generation: str,
    task: HumanEvalTask,
    compression_input: str,
    ground_truth_code: str,
    timeout: float,
) -> HumanEvalScoreResult:
    generated_score = score_generated_code_for_humaneval(
        raw_generation=raw_generation,
        task=task,
        timeout=timeout,
    )
    metrics = compression_metrics(
        ground_truth_code=ground_truth_code,
        representation_text=compression_input,
    )
    best = best_compression_metric(metrics)
    raw_compression_ratio: float | None = None
    if metrics:
        any_metric = next(iter(metrics.values()))
        if any_metric.ground_truth_bytes:
            raw_compression_ratio = (
                any_metric.representation_bytes / any_metric.ground_truth_bytes
            )
    evaluation = generated_score.evaluation
    return HumanEvalScoreResult(
        prediction_id=prediction_id,
        score=generated_score.score,
        error=generated_score.error,
        generated_code_outcome=generated_score.outcome,
        raw_code=generated_score.raw_code,
        raw_compile_ok=generated_score.raw_compile_ok,
        raw_compile_error=generated_score.raw_compile_error,
        extraction_candidate_count=generated_score.extraction_candidate_count,
        selected_candidate_index=generated_score.selected_candidate_index,
        extracted_compile_ok=generated_score.extracted_compile_ok,
        extracted_compile_error=generated_score.extracted_compile_error,
        extraction_error=generated_score.extraction_error,
        evaluation_function_names=evaluation.function_names
        if evaluation
        else [],
        evaluation_total_cases=evaluation.total_cases if evaluation else None,
        evaluation_failure_count=len(evaluation.failures)
        if evaluation
        else None,
        evaluation_status_counts=evaluation.status_counts
        if evaluation
        else {},
        compression_metrics=metrics,
        raw_compression_ratio=raw_compression_ratio,
        best_compression_ratio=best.ratio_to_ground_truth if best else None,
        best_compression_percent_reduction=(
            best.percent_reduction_vs_ground_truth if best else None
        ),
    )
