"""Pure HumanEval scoring primitives.

`GeneratedCodeOutcome` is part of the primitive score contract so later
append-only score attempts can persist why a generation scored zero without
parsing error text. The current v0 experiment writers still persist their
legacy scoring columns only; wiring this outcome into durable score-attempt
records belongs to the schema/scoring-profile stage.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
)

from dr_dspy.humaneval.code_extraction import validate_python_source
from dr_dspy.humaneval.code_parsing import (
    CodeExtractionResult,
    CodeParserProfile,
    extract_best_effort_code,
    extract_code_with_profile,
)
from dr_dspy.humaneval.compression import (
    CompressionMetric,
    CompressionMetrics,
    compression_metrics,
)
from dr_dspy.humaneval.task import (
    EvaluationCaseStatus,
    EvaluationTaskResult,
    EvaluationTaskSummary,
    HumanEvalTask,
    evaluate_human_eval_code,
)

HUMANEVAL_EVALUATION_INCOMPLETE_ERROR = "HumanEval evaluation incomplete"
HUMANEVAL_TESTS_FAILED_ERROR = "HumanEval tests failed"


class GeneratedCodeOutcome(StrEnum):
    PASSED = "passed"
    TESTS_FAILED = "tests_failed"
    EVALUATION_INCOMPLETE = "evaluation_incomplete"
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
    evaluation_summary: EvaluationTaskSummary | None = None
    compression_metrics: CompressionMetrics = Field(default_factory=dict)
    raw_compression_ratio: float | None = None
    best_compression_ratio: float | None = None
    best_compression_percent_reduction: float | None = None


class HumanEvalGenerationScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_generation: str
    extraction: CodeExtractionResult
    outcome: GeneratedCodeOutcome
    score: float
    evaluation: EvaluationTaskResult | None = None


class EvaluationAggregateMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function_names: tuple[StrictStr, ...]
    total_cases: StrictInt
    result_count: StrictInt
    passed_count: StrictInt
    failed_count: StrictInt
    error_count: StrictInt
    timeout_count: StrictInt
    failure_count: StrictInt
    passed: StrictBool
    status_counts: dict[StrictStr, StrictInt]


def score_humaneval_generation(
    *,
    raw_generation: Any,
    task: HumanEvalTask,
    parser_profile: CodeParserProfile,
    timeout_seconds: float,
) -> HumanEvalGenerationScore:
    extraction = extract_code_with_profile(
        raw_generation,
        profile=parser_profile,
    )
    raw_generation_text = extraction.raw_generation or ""
    if extraction.extracted_code is None:
        outcome = extraction_failure_outcome(extraction)
        return HumanEvalGenerationScore(
            raw_generation=raw_generation_text,
            extraction=extraction,
            outcome=outcome,
            score=0.0,
            evaluation=None,
        )

    evaluation = evaluate_human_eval_code(
        task=task,
        candidate_code=extraction.extracted_code,
        timeout_seconds=timeout_seconds,
    )
    outcome = evaluation_outcome(evaluation)
    return HumanEvalGenerationScore(
        raw_generation=raw_generation_text,
        extraction=extraction,
        outcome=outcome,
        score=1.0 if outcome is GeneratedCodeOutcome.PASSED else 0.0,
        evaluation=evaluation,
    )


def extraction_failure_outcome(
    extraction: CodeExtractionResult,
) -> GeneratedCodeOutcome:
    if extraction.extraction_error == "empty raw generation":
        return GeneratedCodeOutcome.EMPTY_GENERATION
    return GeneratedCodeOutcome.EXTRACTION_FAILED


def evaluation_outcome(
    evaluation: EvaluationTaskResult,
) -> GeneratedCodeOutcome:
    if not evaluation.function_names:
        return GeneratedCodeOutcome.NO_TOP_LEVEL_FUNCTIONS
    if evaluation.passed:
        return GeneratedCodeOutcome.PASSED
    return GeneratedCodeOutcome.TESTS_FAILED


def evaluation_aggregate_metrics(
    evaluation: EvaluationTaskResult,
) -> EvaluationAggregateMetrics:
    status_counts = evaluation.status_counts
    return EvaluationAggregateMetrics(
        function_names=tuple(evaluation.function_names),
        total_cases=evaluation.total_cases,
        result_count=len(evaluation.results),
        passed_count=status_counts.get(EvaluationCaseStatus.PASSED.value, 0),
        failed_count=status_counts.get(EvaluationCaseStatus.FAILED.value, 0),
        error_count=status_counts.get(EvaluationCaseStatus.ERROR.value, 0),
        timeout_count=status_counts.get(EvaluationCaseStatus.TIMEOUT.value, 0),
        failure_count=len(evaluation.failures),
        passed=evaluation.passed,
        status_counts=status_counts,
    )


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
    extraction = extract_best_effort_code(raw_generation)
    selected_code = extraction.extracted_code
    if selected_code is None:
        extraction_error = extraction.extraction_error or (
            "no compilable extracted candidate"
        )
        return GeneratedCodeScore(
            outcome=GeneratedCodeOutcome.EXTRACTION_FAILED,
            score=0.0,
            error=extraction_error,
            raw_code=None,
            raw_compile_ok=raw_validation.compile_ok,
            raw_compile_error=raw_validation.compile_error,
            extraction_candidate_count=extraction.candidate_count,
            selected_candidate_index=None,
            extracted_compile_ok=False,
            extracted_compile_error=extraction.compile_error,
            extraction_error=extraction_error,
        )

    evaluation = evaluate_human_eval_code(
        task=task,
        candidate_code=selected_code,
        timeout_seconds=timeout,
    )
    if not evaluation.function_names:
        outcome = GeneratedCodeOutcome.NO_TOP_LEVEL_FUNCTIONS
        error = "no top-level candidate functions"
    elif evaluation.passed:
        outcome = GeneratedCodeOutcome.PASSED
        error = None
    elif not evaluation.coverage_complete and not evaluation.failures:
        outcome = GeneratedCodeOutcome.EVALUATION_INCOMPLETE
        error = HUMANEVAL_EVALUATION_INCOMPLETE_ERROR
    else:
        outcome = GeneratedCodeOutcome.TESTS_FAILED
        error = HUMANEVAL_TESTS_FAILED_ERROR
    return GeneratedCodeScore(
        outcome=outcome,
        score=1.0 if evaluation.passed else 0.0,
        error=error,
        raw_code=selected_code,
        raw_compile_ok=raw_validation.compile_ok,
        raw_compile_error=raw_validation.compile_error,
        extraction_candidate_count=extraction.candidate_count,
        selected_candidate_index=extraction.selected_candidate_index,
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
        evaluation_summary=evaluation.to_summary() if evaluation else None,
        compression_metrics=metrics,
        raw_compression_ratio=raw_compression_ratio,
        best_compression_ratio=best.ratio_to_ground_truth if best else None,
        best_compression_percent_reduction=(
            best.percent_reduction_vs_ground_truth if best else None
        ),
    )
