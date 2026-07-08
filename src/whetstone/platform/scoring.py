from __future__ import annotations

from datetime import UTC, datetime

from dr_code.humaneval import (
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
    CompletedScore,
    HarnessFailure,
    HumanEvalScoringProfile,
    HumanEvalTask,
    NodeOutputMetricsSource,
    build_metrics_payload,
    evaluation_aggregate_metrics,
    resolve_humaneval_scoring_profile,
    score_humaneval_submission,
)
from dr_providers import FailureClass

from whetstone.eval_failures.recording import recordable_text
from whetstone.records import (
    DEFAULT_SCORE_DATASET_NAME,
    DEFAULT_SCORE_DATASET_SPLIT,
    ExtractedSubmissionPayload,
    GenerationRunRecord,
    GenerationRunStatus,
    HarnessFailureCausePayload,
    MetricsPayload,
    NodeAttemptRecord,
    PerTestResultPayload,
    PredictionSpecRecord,
    ScoreAttemptRecord,
    ScoreAttemptStatus,
    ScoreHarnessFailureRecord,
    stable_score_attempt_id,
)
from whetstone.records.limits import METRICS_STAGES_MAX_COUNT

type ScoreSubmissionRunRecord = ScoreAttemptRecord | ScoreHarnessFailureRecord


def score_submission_run(
    *,
    spec: PredictionSpecRecord,
    generation_run: GenerationRunRecord,
    node_attempts: tuple[NodeAttemptRecord, ...],
    task: HumanEvalTask,
    scoring_profile: HumanEvalScoringProfile | None = None,
    scoring_profile_id: str = HUMANEVAL_SCORING_PROFILE_ID,
    scoring_profile_version: str = HUMANEVAL_SCORING_PROFILE_VERSION,
    score_attempt_index: int = 0,
    dataset_name: str = DEFAULT_SCORE_DATASET_NAME,
    dataset_split: str = DEFAULT_SCORE_DATASET_SPLIT,
    started_at: datetime,
    completed_at: datetime | None = None,
) -> ScoreSubmissionRunRecord:
    scoring_profile = scoring_profile or resolve_humaneval_scoring_profile(
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
    )
    validate_generation_run_for_scoring(
        spec=spec,
        run=generation_run,
        task=task,
    )
    domain_score = score_humaneval_submission(
        raw_submission=generation_run.summary.terminal_submission_text,
        task=task,
        parser_profile=scoring_profile.parser_profile,
        timeout_seconds=scoring_profile.timeout_seconds,
    )
    resolved_completed_at = resolve_completed_at(completed_at)
    if isinstance(domain_score, CompletedScore):
        return score_attempt_from_completed_score(
            spec=spec,
            generation_run=generation_run,
            node_attempts=node_attempts,
            task=task,
            scoring_profile=scoring_profile,
            score_attempt_index=score_attempt_index,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            completed_score=domain_score,
            started_at=started_at,
            completed_at=resolved_completed_at,
        )
    return score_harness_failure_from_domain_failure(
        spec=spec,
        generation_run=generation_run,
        scoring_profile=scoring_profile,
        score_attempt_index=score_attempt_index,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        harness_failure=domain_score,
        started_at=started_at,
        completed_at=resolved_completed_at,
    )


def score_attempt_from_completed_score(
    *,
    spec: PredictionSpecRecord,
    generation_run: GenerationRunRecord,
    node_attempts: tuple[NodeAttemptRecord, ...],
    task: HumanEvalTask,
    scoring_profile: HumanEvalScoringProfile,
    score_attempt_index: int,
    dataset_name: str,
    dataset_split: str,
    completed_score: CompletedScore,
    started_at: datetime,
    completed_at: datetime,
) -> ScoreAttemptRecord:
    extraction = completed_score.extraction
    parser_profile = scoring_profile.parser_profile
    extracted_payload = ExtractedSubmissionPayload(
        raw_submission=completed_score.raw_submission,
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
    if completed_score.evaluation is not None:
        per_test_results = tuple(
            PerTestResultPayload.from_evaluation_case(result.to_summary())
            for result in completed_score.evaluation.results
        )
    return ScoreAttemptRecord(
        score_attempt_id=stable_score_attempt_id(
            generation_run_id=generation_run.generation_run_id,
            scoring_profile_id=scoring_profile.profile_id,
            scoring_profile_version=scoring_profile.version,
            parser_profile_id=parser_profile.profile_id,
            parser_version=parser_profile.version,
            attempt_index=score_attempt_index,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
        ),
        prediction_id=spec.prediction_id,
        generation_run_id=generation_run.generation_run_id,
        attempt_index=score_attempt_index,
        scoring_profile_id=scoring_profile.profile_id,
        scoring_profile_version=scoring_profile.version,
        parser_profile_id=parser_profile.profile_id,
        parser_version=parser_profile.version,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        status=ScoreAttemptStatus.SUCCESS,
        submission_outcome=completed_score.outcome,
        score=completed_score.score,
        extracted_submission=extracted_payload,
        metrics=score_metrics_payload(
            task=task,
            node_attempts=node_attempts,
            scoring_profile=scoring_profile,
            completed_score=completed_score,
        ),
        per_test_results=per_test_results,
        started_at=started_at,
        completed_at=completed_at,
    )


def score_harness_failure_from_domain_failure(
    *,
    spec: PredictionSpecRecord,
    generation_run: GenerationRunRecord,
    scoring_profile: HumanEvalScoringProfile,
    score_attempt_index: int,
    dataset_name: str,
    dataset_split: str,
    harness_failure: HarnessFailure,
    started_at: datetime,
    completed_at: datetime,
) -> ScoreHarnessFailureRecord:
    parser_profile = scoring_profile.parser_profile
    extracted_payload = None
    if harness_failure.extraction is not None:
        extraction = harness_failure.extraction
        extracted_payload = ExtractedSubmissionPayload(
            raw_submission=harness_failure.raw_submission,
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
    return ScoreHarnessFailureRecord(
        score_attempt_id=stable_score_attempt_id(
            generation_run_id=generation_run.generation_run_id,
            scoring_profile_id=scoring_profile.profile_id,
            scoring_profile_version=scoring_profile.version,
            parser_profile_id=parser_profile.profile_id,
            parser_version=parser_profile.version,
            attempt_index=score_attempt_index,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
        ),
        prediction_id=spec.prediction_id,
        generation_run_id=generation_run.generation_run_id,
        attempt_index=score_attempt_index,
        scoring_profile_id=scoring_profile.profile_id,
        scoring_profile_version=scoring_profile.version,
        parser_profile_id=parser_profile.profile_id,
        parser_version=parser_profile.version,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        kind=harness_failure.kind,
        raw_submission=harness_failure.raw_submission,
        extracted_submission=extracted_payload,
        cause=HarnessFailureCausePayload.model_validate(
            harness_failure.cause.model_dump(mode="json")
        ),
        failure_class=harness_failure_failure_class(harness_failure),
        started_at=started_at,
        completed_at=completed_at,
    )


def harness_failure_failure_class(
    harness_failure: HarnessFailure,
) -> FailureClass:
    try:
        return FailureClass(harness_failure.failure_class)
    except ValueError:
        return FailureClass.UNKNOWN


def validate_generation_run_for_scoring(
    *,
    spec: PredictionSpecRecord,
    run: GenerationRunRecord,
    task: HumanEvalTask,
) -> None:
    if run.prediction_id != spec.prediction_id:
        raise ValueError("generation run prediction_id does not match spec")
    if task.task_id != spec.task_id:
        raise ValueError(
            "HumanEval task_id does not match spec: "
            f"{task.task_id!r} != {spec.task_id!r}"
        )
    if run.status is GenerationRunStatus.SUCCESS:
        return
    if run.status is GenerationRunStatus.PARTIAL:
        if not run.summary.terminal_submission_text.strip():
            raise ValueError(
                "partial generation run missing terminal_submission_text"
            )
        return
    raise ValueError(
        f"generation run is not scoreable: {run.status.value}"
    )


def score_metrics_payload(
    *,
    task: HumanEvalTask,
    node_attempts: tuple[NodeAttemptRecord, ...],
    scoring_profile: HumanEvalScoringProfile,
    completed_score: CompletedScore,
) -> MetricsPayload:
    node_output_sources = node_output_metrics_sources(node_attempts)
    max_node_sources = METRICS_STAGES_MAX_COUNT - 1
    if len(node_output_sources) > max_node_sources:
        raise ValueError(
            f"node output metrics sources cannot exceed {max_node_sources} "
            f"entries (metrics.stages cap is {METRICS_STAGES_MAX_COUNT})"
        )
    metrics_payload = build_metrics_payload(
        raw_submission=completed_score.raw_submission,
        extracted_code=completed_score.extraction.extracted_code,
        task=task,
        node_output_sources=node_output_sources,
        profile_id=scoring_profile.metrics_profile_id,
        profile_version=scoring_profile.metrics_profile_version,
    ).model_dump(mode="json")
    if completed_score.evaluation is not None:
        metrics_payload["custom"] = {
            **metrics_payload["custom"],
            "evaluation": evaluation_aggregate_metrics(
                completed_score.evaluation
            ).model_dump(mode="json"),
        }
    return MetricsPayload.model_validate(metrics_payload)


def node_output_metrics_sources(
    node_attempts: tuple[NodeAttemptRecord, ...],
) -> tuple[NodeOutputMetricsSource, ...]:
    sources: list[NodeOutputMetricsSource] = []
    for attempt in node_attempts:
        if attempt.output is None:
            continue
        for field_name, value in sorted(attempt.output.values.items()):
            sources.append(
                NodeOutputMetricsSource(
                    node_id=attempt.node_id,
                    field_name=field_name,
                    text=recordable_text(value),
                )
            )
    return tuple(sources)


def resolve_completed_at(completed_at: datetime | None) -> datetime:
    return completed_at if completed_at is not None else datetime.now(UTC)
