from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from dr_dspy.eval_failures import (
    classify_exception,
    exception_type_name,
    failure_metadata_from_exception,
)
from dr_dspy.humaneval.metrics import (
    NodeOutputMetricsSource,
    build_metrics_payload,
)
from dr_dspy.humaneval.profiles import (
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
    HumanEvalScoringProfile,
    resolve_humaneval_scoring_profile,
)
from dr_dspy.humaneval.scoring import (
    HumanEvalGenerationScore,
    score_humaneval_generation,
)
from dr_dspy.humaneval.task import HumanEvalTask
from dr_dspy.records import (
    ExtractedCodePayload,
    FailureMetadataPayload,
    GenerationRunRecord,
    GenerationRunStatus,
    MetricsPayload,
    NodeAttemptRecord,
    PerTestResultPayload,
    PredictionSpecRecord,
    ScoreAttemptRecord,
    ScoreAttemptStatus,
    stable_score_attempt_id,
)


def score_generation_run(
    *,
    spec: PredictionSpecRecord,
    generation_run: GenerationRunRecord,
    node_attempts: tuple[NodeAttemptRecord, ...],
    task: HumanEvalTask,
    scoring_profile: HumanEvalScoringProfile | None = None,
    scoring_profile_id: str = HUMANEVAL_SCORING_PROFILE_ID,
    scoring_profile_version: str = HUMANEVAL_SCORING_PROFILE_VERSION,
    score_attempt_index: int = 0,
    started_at: datetime,
    completed_at: datetime | None = None,
) -> ScoreAttemptRecord:
    scoring_profile = scoring_profile or resolve_humaneval_scoring_profile(
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
    )
    try:
        validate_generation_run_for_scoring(spec=spec, run=generation_run)
        raw_generation = terminal_generation_text(generation_run)
        domain_score = score_humaneval_generation(
            raw_generation=raw_generation,
            task=task,
            parser_profile=scoring_profile.parser_profile,
            timeout_seconds=scoring_profile.timeout_seconds,
        )
        return score_attempt_from_domain_score(
            spec=spec,
            generation_run=generation_run,
            node_attempts=node_attempts,
            task=task,
            scoring_profile=scoring_profile,
            score_attempt_index=score_attempt_index,
            domain_score=domain_score,
            started_at=started_at,
            completed_at=completed_at,
        )
    except Exception as error:
        return error_score_attempt(
            spec=spec,
            generation_run=generation_run,
            scoring_profile=scoring_profile,
            score_attempt_index=score_attempt_index,
            error=error,
            started_at=started_at,
            completed_at=resolve_completed_at(completed_at),
        )


def score_attempt_from_domain_score(
    *,
    spec: PredictionSpecRecord,
    generation_run: GenerationRunRecord,
    node_attempts: tuple[NodeAttemptRecord, ...],
    task: HumanEvalTask,
    scoring_profile: HumanEvalScoringProfile,
    score_attempt_index: int,
    domain_score: HumanEvalGenerationScore,
    started_at: datetime,
    completed_at: datetime | None,
) -> ScoreAttemptRecord:
    return successful_score_attempt(
        spec=spec,
        generation_run=generation_run,
        node_attempts=node_attempts,
        task=task,
        scoring_profile=scoring_profile,
        score_attempt_index=score_attempt_index,
        domain_score=domain_score,
        started_at=started_at,
        completed_at=resolve_completed_at(completed_at),
    )


def successful_score_attempt(
    *,
    spec: PredictionSpecRecord,
    generation_run: GenerationRunRecord,
    node_attempts: tuple[NodeAttemptRecord, ...],
    task: HumanEvalTask,
    scoring_profile: HumanEvalScoringProfile,
    score_attempt_index: int,
    domain_score: HumanEvalGenerationScore,
    started_at: datetime,
    completed_at: datetime,
) -> ScoreAttemptRecord:
    extraction = domain_score.extraction
    parser_profile = scoring_profile.parser_profile
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
    if domain_score.evaluation is not None:
        per_test_results = tuple(
            PerTestResultPayload.from_evaluation_case(result.to_summary())
            for result in domain_score.evaluation.results
        )
    return ScoreAttemptRecord(
        score_attempt_id=stable_score_attempt_id(
            generation_run_id=generation_run.generation_run_id,
            scoring_profile_id=scoring_profile.profile_id,
            scoring_profile_version=scoring_profile.version,
            parser_profile_id=parser_profile.profile_id,
            parser_version=parser_profile.version,
            attempt_index=score_attempt_index,
        ),
        prediction_id=spec.prediction_id,
        generation_run_id=generation_run.generation_run_id,
        attempt_index=score_attempt_index,
        scoring_profile_id=scoring_profile.profile_id,
        scoring_profile_version=scoring_profile.version,
        parser_profile_id=parser_profile.profile_id,
        parser_version=parser_profile.version,
        status=ScoreAttemptStatus.SUCCESS,
        generated_code_outcome=domain_score.outcome,
        score=domain_score.score,
        extracted_code=extracted_payload,
        metrics=MetricsPayload.model_validate(
            build_metrics_payload(
                raw_generation=domain_score.raw_generation,
                extracted_code=extraction.extracted_code,
                task=task,
                node_output_sources=node_output_metrics_sources(
                    node_attempts
                ),
                profile_id=scoring_profile.metrics_profile_id,
                profile_version=scoring_profile.metrics_profile_version,
            ).model_dump(mode="json")
        ),
        per_test_results=per_test_results,
        started_at=started_at,
        completed_at=completed_at,
    )


def error_score_attempt(
    *,
    spec: PredictionSpecRecord,
    generation_run: GenerationRunRecord,
    scoring_profile: HumanEvalScoringProfile,
    score_attempt_index: int,
    error: BaseException,
    started_at: datetime,
    completed_at: datetime,
) -> ScoreAttemptRecord:
    return ScoreAttemptRecord(
        score_attempt_id=stable_score_attempt_id(
            generation_run_id=generation_run.generation_run_id,
            scoring_profile_id=scoring_profile.profile_id,
            scoring_profile_version=scoring_profile.version,
            parser_profile_id=scoring_profile.parser_profile.profile_id,
            parser_version=scoring_profile.parser_profile.version,
            attempt_index=score_attempt_index,
        ),
        prediction_id=spec.prediction_id,
        generation_run_id=generation_run.generation_run_id,
        attempt_index=score_attempt_index,
        scoring_profile_id=scoring_profile.profile_id,
        scoring_profile_version=scoring_profile.version,
        parser_profile_id=scoring_profile.parser_profile.profile_id,
        parser_version=scoring_profile.parser_profile.version,
        status=ScoreAttemptStatus.ERROR,
        failure=failure_payload(
            error,
            metadata={
                "prediction_id": spec.prediction_id,
                "generation_run_id": generation_run.generation_run_id,
                "task_id": spec.task_id,
                "parser_profile_id": scoring_profile.parser_profile.profile_id,
                "parser_version": scoring_profile.parser_profile.version,
                "scoring_profile_id": scoring_profile.profile_id,
                "scoring_profile_version": scoring_profile.version,
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


def node_output_metrics_sources(
    node_attempts: tuple[NodeAttemptRecord, ...],
) -> tuple[NodeOutputMetricsSource, ...]:
    sources: list[NodeOutputMetricsSource] = []
    for attempt in node_attempts:
        if attempt.output is None:
            continue
        for field_name, value in sorted(attempt.output.values.items()):
            if isinstance(value, str):
                sources.append(
                    NodeOutputMetricsSource(
                        node_id=attempt.node_id,
                        field_name=field_name,
                        text=value,
                    )
                )
    return tuple(sources)


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


def resolve_completed_at(completed_at: datetime | None) -> datetime:
    return completed_at if completed_at is not None else datetime.now(UTC)
