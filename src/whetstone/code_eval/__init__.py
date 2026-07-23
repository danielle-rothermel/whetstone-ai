"""Whetstone dr-code adapters and experiment derivations.

This package is the Whetstone side of the dr-code ownership move: it consumes
the released dr-code evaluation kernel (typed candidates, compile-valid Code
Artifacts, generic compression references, pure aggregation) and adds only the
Whetstone-owned experiment policy and boundary roles. It introduces **no**
duplicate dr-code type, artifact, schema, or identity.

Deliverables:

* **Code Generation / Submission Text** boundary (``submission``) — projects an
  exact decoder Generation into native ``TextArtifact.text``.
* **Candidate Correctness** (``correctness``) — the all-candidates any-passing
  policy with distinguishable outcomes; infrastructure-unknown fails the
  rollout, never scores 0.
* **Score / Metric Fact derivations** (``scoring``) — Binary Test Pass Score,
  Compressed Description Length (zstd-19), Compression Ratio.
* **Compression Reference Selection** (``compression_selection``) — the
  experiment rule selecting ``task.gt_code_wo_comments`` bytes onto a generic
  dr-code Compression Reference Key.
* **Rollout Aggregate** (``aggregate``) — provenance-bearing binding of pure
  dr-code aggregation output; Average Binary Test Pass Rate and Mean
  Compression Ratio with explicit missing/failed-row policy.
"""

from whetstone.code_eval.aggregate import (
    CompletenessPolicy,
    RolloutAggregate,
    RowPolicy,
    RowValue,
    TaskRows,
    aggregation_definition,
    as_completeness_policy,
    average_binary_test_pass_rate,
    enforce_skip_tolerance,
    mean_compression_ratio,
)
from whetstone.code_eval.compression_selection import (
    COMPRESSION_REFERENCE_NAMESPACE,
    SELECTED_FIELD,
    ExperimentTaskView,
    build_resolver,
    compression_reference_binding,
    compression_reference_key,
    select_compression_reference,
)
from whetstone.code_eval.correctness import (
    CandidateEvaluation,
    CandidateRunner,
    CandidateVerdict,
    CorrectnessOutcome,
    CorrectnessResult,
    correctness_absent,
    evaluate_candidate_correctness,
)
from whetstone.code_eval.scoring import (
    BINARY_TEST_PASS_SCORE_NAME,
    COMPRESSED_DESCRIPTION_LENGTH_NAME,
    COMPRESSION_RATIO_NAME,
    ZSTD_LEVEL,
    InfrastructureUnknownScoreError,
    binary_test_pass_score,
    compressed_description_length_bytes,
    compressed_description_length_fact,
    compression_ratio_score,
    compression_ratio_value,
)
from whetstone.code_eval.submission import (
    submission_text,
    submission_text_artifact,
)

__all__ = [
    "BINARY_TEST_PASS_SCORE_NAME",
    "COMPRESSED_DESCRIPTION_LENGTH_NAME",
    "COMPRESSION_RATIO_NAME",
    "COMPRESSION_REFERENCE_NAMESPACE",
    "SELECTED_FIELD",
    "ZSTD_LEVEL",
    "CandidateEvaluation",
    "CandidateRunner",
    "CandidateVerdict",
    "CompletenessPolicy",
    "CorrectnessOutcome",
    "CorrectnessResult",
    "ExperimentTaskView",
    "InfrastructureUnknownScoreError",
    "RolloutAggregate",
    "RowPolicy",
    "RowValue",
    "TaskRows",
    "aggregation_definition",
    "as_completeness_policy",
    "average_binary_test_pass_rate",
    "binary_test_pass_score",
    "build_resolver",
    "compressed_description_length_bytes",
    "compressed_description_length_fact",
    "compression_ratio_score",
    "compression_ratio_value",
    "compression_reference_binding",
    "compression_reference_key",
    "correctness_absent",
    "enforce_skip_tolerance",
    "evaluate_candidate_correctness",
    "mean_compression_ratio",
    "select_compression_reference",
    "submission_text",
    "submission_text_artifact",
]
