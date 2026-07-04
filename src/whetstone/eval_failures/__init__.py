"""Eval worker step failure taxonomy, policy, and recording boundary.

This package handles eval workflow failures: classify, retry, summarize, and
persist failure records. It is not a global exception registry.

Encoding errors live in ``dr_serialize`` and are bridged at
``eval_failures.recording``. Third-party exceptions are classified by
heuristics in ``eval_failures.policy`` without requiring custom types.
"""

from importlib import import_module
from typing import Any

from whetstone.eval_failures.exceptions import (
    EmptyGenerationError,
    EvalFailureError,
    PermanentFailureError,
    PredictionParseError,
    ProviderResponseParseError,
    RateLimitedFailureError,
    RecordingFailureError,
    ResourceExhaustionFailureError,
    StrandedGenerationError,
    StrandedScoringError,
    TransientFailureError,
    UnknownFailureError,
)
from whetstone.eval_failures.generation import (
    require_generation_text,
    validate_direct_generation,
    validate_encdec_generation,
)
from whetstone.eval_failures.types import (
    RECOVERABLE_FAILURE_CLASSES,
    RETRYABLE_STEP_FAILURE_CLASSES,
    FailureClass,
)

__all__ = [
    "RECOVERABLE_FAILURE_CLASSES",
    "RETRYABLE_STEP_FAILURE_CLASSES",
    "EmptyGenerationError",
    "EvalFailureError",
    "FailureClass",
    "FailureSummary",
    "PermanentFailureError",
    "PredictionParseError",
    "ProviderResponseParseError",
    "RateLimitedFailureError",
    "RecordingFailureError",
    "ResourceExhaustionFailureError",
    "StrandedGenerationError",
    "StrandedScoringError",
    "TransientFailureError",
    "UnknownFailureError",
    "classify_exception",
    "ensure_recordable",
    "error_text",
    "exception_type_name",
    "failure_metadata_dict_from_exception",
    "failure_metadata_from_exception",
    "failure_summary_payload",
    "find_classified_exception",
    "recordable_jsonb",
    "recordable_text",
    "require_generation_text",
    "should_retry_step",
    "summarize_exception",
    "unwrap_exception",
    "validate_direct_generation",
    "validate_encdec_generation",
]

_LAZY_EXPORTS = {
    "FailureSummary": ("whetstone.eval_failures.policy", "FailureSummary"),
    "classify_exception": (
        "whetstone.eval_failures.policy",
        "classify_exception",
    ),
    "error_text": ("whetstone.eval_failures.policy", "error_text"),
    "exception_type_name": (
        "whetstone.eval_failures.policy",
        "exception_type_name",
    ),
    "failure_summary_payload": (
        "whetstone.eval_failures.policy",
        "failure_summary_payload",
    ),
    "find_classified_exception": (
        "whetstone.eval_failures.policy",
        "find_classified_exception",
    ),
    "should_retry_step": (
        "whetstone.eval_failures.policy",
        "should_retry_step",
    ),
    "summarize_exception": (
        "whetstone.eval_failures.policy",
        "summarize_exception",
    ),
    "unwrap_exception": ("whetstone.eval_failures.policy", "unwrap_exception"),
    "ensure_recordable": (
        "whetstone.eval_failures.recording",
        "ensure_recordable",
    ),
    "failure_metadata_dict_from_exception": (
        "whetstone.eval_failures.recording",
        "failure_metadata_dict_from_exception",
    ),
    "failure_metadata_from_exception": (
        "whetstone.eval_failures.recording",
        "failure_metadata_from_exception",
    ),
    "recordable_jsonb": (
        "whetstone.eval_failures.recording",
        "recordable_jsonb",
    ),
    "recordable_text": (
        "whetstone.eval_failures.recording",
        "recordable_text",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
