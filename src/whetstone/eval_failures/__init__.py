"""Eval worker step failure taxonomy, policy, and recording boundary.

This package handles eval workflow failures: classify, retry, summarize, and
persist failure records. It is not a global exception registry.

Encoding errors live in ``dr_serialize`` and are bridged at
``eval_failures.recording``. Failures must use an explicit Whetstone or
provider exception type before they can be classified and persisted.
"""

from dr_providers import (
    RECOVERABLE_FAILURE_CLASSES,
    RETRYABLE_FAILURE_CLASSES,
    FailureClass,
)

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
from whetstone.eval_failures.policy import (
    FailureSummary,
    classify_exception,
    error_text,
    exception_type_name,
    failure_summary_payload,
    find_classified_exception,
    should_retry_step,
    summarize_exception,
    unwrap_exception,
)
from whetstone.eval_failures.recording import (
    ensure_recordable,
    failure_metadata_dict_from_exception,
    failure_metadata_from_exception,
    recordable_jsonb,
    recordable_text,
)

__all__ = [
    "RECOVERABLE_FAILURE_CLASSES",
    "RETRYABLE_FAILURE_CLASSES",
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
