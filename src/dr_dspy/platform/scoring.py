from __future__ import annotations

from datetime import datetime
from typing import Any

from dr_dspy.eval_failures import (
    classify_exception,
    exception_type_name,
    failure_metadata_from_exception,
)
from dr_dspy.humaneval.scoring import GeneratedCodeOutcome
from dr_dspy.humaneval.task import (
    EvaluationTaskResult,
    HumanEvalTask,
    evaluate_human_eval_code,
)
from dr_dspy.platform.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    PARSER_PROFILE_VERSION,
    CodeExtractionResult,
    CodeParserProfile,
    extract_code_with_profile,
)
from dr_dspy.platform.metrics import build_metrics_payload
from dr_dspy.records import (
    ExtractedCodePayload,
    FailureMetadataPayload,
    GenerationRunRecord,
    GenerationRunStatus,
    NodeAttemptRecord,
    PerTestResultPayload,
    PredictionSpecRecord,
    ScoreAttemptRecord,
    ScoreAttemptStatus,
    stable_score_attempt_id,
)

HUMANEVAL_SCORING_PROFILE_ID = "humaneval"
HUMANEVAL_SCORING_PROFILE_VERSION = "v1"
DEFAULT_HUMANEVAL_TIMEOUT_SECONDS = 2.0


def score_generation_run(
    *,
    spec: PredictionSpecRecord,
    generation_run: GenerationRunRecord,
    node_attempts: tuple[NodeAttemptRecord, ...],
    task: HumanEvalTask,
    parser_profile: CodeParserProfile,
    scoring_profile_id: str = HUMANEVAL_SCORING_PROFILE_ID,
    scoring_profile_version: str = HUMANEVAL_SCORING_PROFILE_VERSION,
    score_attempt_index: int = 0,
    timeout_seconds: float = DEFAULT_HUMANEVAL_TIMEOUT_SECONDS,
    started_at: datetime,
    completed_at: datetime,
) -> ScoreAttemptRecord:
    try:
        validate_generation_run_for_scoring(spec=spec, run=generation_run)
        raw_generation = terminal_generation_text(generation_run)
        extraction = extract_code_with_profile(
            raw_generation,
            profile=parser_profile,
        )
        return score_attempt_from_extraction(
            spec=spec,
            generation_run=generation_run,
            node_attempts=node_attempts,
            task=task,
            parser_profile=parser_profile,
            scoring_profile_id=scoring_profile_id,
            scoring_profile_version=scoring_profile_version,
            score_attempt_index=score_attempt_index,
            timeout_seconds=timeout_seconds,
            extraction=extraction,
            started_at=started_at,
            completed_at=completed_at,
        )
    except Exception as error:
        return error_score_attempt(
            spec=spec,
            generation_run=generation_run,
            parser_profile=parser_profile,
            scoring_profile_id=scoring_profile_id,
            scoring_profile_version=scoring_profile_version,
            score_attempt_index=score_attempt_index,
            error=error,
            started_at=started_at,
            completed_at=completed_at,
        )


def score_attempt_from_extraction(
    *,
    spec: PredictionSpecRecord,
    generation_run: GenerationRunRecord,
    node_attempts: tuple[NodeAttemptRecord, ...],
    task: HumanEvalTask,
    parser_profile: CodeParserProfile,
    scoring_profile_id: str,
    scoring_profile_version: str,
    score_attempt_index: int,
    timeout_seconds: float,
    extraction: CodeExtractionResult,
    started_at: datetime,
    completed_at: datetime,
) -> ScoreAttemptRecord:
    if extraction.extracted_code is None:
        outcome = extraction_failure_outcome(extraction)
        raw_generation = extraction.raw_generation or ""
        return successful_score_attempt(
            spec=spec,
            generation_run=generation_run,
            node_attempts=node_attempts,
            task=task,
            parser_profile=parser_profile,
            scoring_profile_id=scoring_profile_id,
            scoring_profile_version=scoring_profile_version,
            score_attempt_index=score_attempt_index,
            outcome=outcome,
            score=0.0,
            extraction=extraction,
            evaluation=None,
            raw_generation=raw_generation,
            started_at=started_at,
            completed_at=completed_at,
        )

    evaluation = evaluate_human_eval_code(
        task=task,
        candidate_code=extraction.extracted_code,
        timeout_seconds=timeout_seconds,
    )
    outcome = evaluation_outcome(evaluation)
    return successful_score_attempt(
        spec=spec,
        generation_run=generation_run,
        node_attempts=node_attempts,
        task=task,
        parser_profile=parser_profile,
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
        score_attempt_index=score_attempt_index,
        outcome=outcome,
        score=1.0 if outcome is GeneratedCodeOutcome.PASSED else 0.0,
        extraction=extraction,
        evaluation=evaluation,
        raw_generation=extraction.raw_generation or "",
        started_at=started_at,
        completed_at=completed_at,
    )


def successful_score_attempt(
    *,
    spec: PredictionSpecRecord,
    generation_run: GenerationRunRecord,
    node_attempts: tuple[NodeAttemptRecord, ...],
    task: HumanEvalTask,
    parser_profile: CodeParserProfile,
    scoring_profile_id: str,
    scoring_profile_version: str,
    score_attempt_index: int,
    outcome: GeneratedCodeOutcome,
    score: float,
    extraction: CodeExtractionResult,
    evaluation: EvaluationTaskResult | None,
    raw_generation: str,
    started_at: datetime,
    completed_at: datetime,
) -> ScoreAttemptRecord:
    extracted_payload = ExtractedCodePayload(
        raw_generation=extraction.raw_generation,
        extracted_code=extraction.extracted_code,
        extraction_method=(
            extraction.extraction_method.value
            if extraction.extraction_method is not None
            else None
        ),
        parser_profile_id=parser_profile.profile_id,
        parser_version=parser_profile.version,
        metadata={
            **extraction.metadata,
            "compile_ok": extraction.compile_ok,
            "compile_error": extraction.compile_error,
            "extraction_error": extraction.extraction_error,
        },
    )
    per_test_results = ()
    if evaluation is not None:
        per_test_results = tuple(
            PerTestResultPayload.from_evaluation_case(result.to_summary())
            for result in evaluation.results
        )
    return ScoreAttemptRecord(
        score_attempt_id=stable_score_attempt_id(
            generation_run_id=generation_run.generation_run_id,
            scoring_profile_id=scoring_profile_id,
            scoring_profile_version=scoring_profile_version,
            parser_profile_id=parser_profile.profile_id,
            parser_version=parser_profile.version,
            attempt_index=score_attempt_index,
        ),
        prediction_id=spec.prediction_id,
        generation_run_id=generation_run.generation_run_id,
        attempt_index=score_attempt_index,
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
        parser_profile_id=parser_profile.profile_id,
        parser_version=parser_profile.version,
        status=ScoreAttemptStatus.SUCCESS,
        generated_code_outcome=outcome,
        score=score,
        extracted_code=extracted_payload,
        metrics=build_metrics_payload(
            raw_generation=raw_generation,
            extracted_code=extraction.extracted_code,
            task=task,
            node_attempts=node_attempts,
        ),
        per_test_results=per_test_results,
        started_at=started_at,
        completed_at=completed_at,
    )


def error_score_attempt(
    *,
    spec: PredictionSpecRecord,
    generation_run: GenerationRunRecord,
    parser_profile: CodeParserProfile,
    scoring_profile_id: str,
    scoring_profile_version: str,
    score_attempt_index: int,
    error: BaseException,
    started_at: datetime,
    completed_at: datetime,
) -> ScoreAttemptRecord:
    return ScoreAttemptRecord(
        score_attempt_id=stable_score_attempt_id(
            generation_run_id=generation_run.generation_run_id,
            scoring_profile_id=scoring_profile_id,
            scoring_profile_version=scoring_profile_version,
            parser_profile_id=parser_profile.profile_id,
            parser_version=parser_profile.version,
            attempt_index=score_attempt_index,
        ),
        prediction_id=spec.prediction_id,
        generation_run_id=generation_run.generation_run_id,
        attempt_index=score_attempt_index,
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
        parser_profile_id=parser_profile.profile_id,
        parser_version=parser_profile.version,
        status=ScoreAttemptStatus.ERROR,
        failure=failure_payload(
            error,
            metadata={
                "prediction_id": spec.prediction_id,
                "generation_run_id": generation_run.generation_run_id,
                "task_id": spec.task_id,
                "parser_profile_id": parser_profile.profile_id,
                "parser_version": parser_profile.version,
                "scoring_profile_id": scoring_profile_id,
                "scoring_profile_version": scoring_profile_version,
            },
        ),
        started_at=started_at,
        completed_at=completed_at,
    )


def validate_generation_run_for_scoring(
    *,
    spec: PredictionSpecRecord,
    run: GenerationRunRecord,
) -> None:
    if run.prediction_id != spec.prediction_id:
        raise ValueError("generation run prediction_id does not match spec")
    if run.status is not GenerationRunStatus.SUCCESS:
        raise ValueError(
            f"generation run is not terminal success: {run.status.value}"
        )


def terminal_generation_text(run: GenerationRunRecord) -> Any:
    output = run.summary.terminal_output
    if isinstance(output, str):
        return output
    if isinstance(output, dict) or getattr(output, "code", None) is not None:
        return output  # type: ignore[return-value]
    raise TypeError(
        "generation run terminal output is not string or code-bearing payload"
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


def failure_payload(
    error: BaseException,
    *,
    metadata: dict[str, Any],
) -> FailureMetadataPayload:
    return FailureMetadataPayload(
        failure_class=classify_exception(error),
        error_type=exception_type_name(error),
        message=str(error),
        metadata={**metadata, **failure_metadata_from_exception(error)},
    )


def default_parser_profile() -> CodeParserProfile:
    return CodeParserProfile(
        profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        version=PARSER_PROFILE_VERSION,
    )
