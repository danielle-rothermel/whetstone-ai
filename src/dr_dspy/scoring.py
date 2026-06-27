from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from dr_dspy.code_eval import DEFAULT_CAPTURE_LIMIT_BYTES
from dr_dspy.code_extraction import apply_cleaning, validate_python_source
from dr_dspy.human_eval import (
    EvaluationTaskResult,
    HumanEvalTask,
    evaluate_human_eval_code,
)


class GeneratedCodeScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False


def score_generated_code_for_humaneval(
    *,
    raw_generation: str,
    task: HumanEvalTask,
    timeout: float,
    capture_limit_bytes: int = DEFAULT_CAPTURE_LIMIT_BYTES,
) -> GeneratedCodeScore:
    _ = capture_limit_bytes
    if not raw_generation.strip():
        extraction_error = "empty raw generation"
        return GeneratedCodeScore(
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
    error = None if evaluation.passed else "HumanEval tests failed"
    if not evaluation.function_names:
        error = "no top-level candidate functions"
    return GeneratedCodeScore(
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
