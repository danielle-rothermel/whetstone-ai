"""Whetstone dr-code adapters and experiment derivations.

This package consumes the released dr-code evaluation kernel's generic
compression references and pure aggregation primitives, and adds only
Whetstone-owned experiment policy and boundary roles. It introduces **no**
duplicate dr-code type, artifact, schema, or identity.

Deliverables:

* **Code Generation / Submission Text** boundary (``submission``) — projects an
  exact decoder Generation into native ``TextArtifact.text``.
* **Score / Metric Fact derivations** (``scoring``) — Compressed Description
  Length (zstd-19) and Compression Ratio.
* **Compression Reference Selection** (``compression_selection``) — the
  experiment rule selecting ``task.gt_code_wo_comments`` bytes onto a generic
  dr-code Compression Reference Key.
* **Rollout Aggregate** (``aggregate``) — provenance-bearing binding of pure
  dr-code aggregation output; one caller-named Unweighted Task Mean over a
  validated evaluation matrix with explicit missing/failed-row policy.
* **Bootstrap Statistics** (``statistics``) — reproducible percentile
  confidence intervals with tasks as the resampling unit.
* **Power Analysis** (``power``) — deterministic paired sample-size
  recommendations over task-count and repeat-count grids.
"""

from whetstone.code_eval.aggregate import (
    CompletenessPolicy,
    EvaluationMatrixPlan,
    RolloutAggregate,
    RowPolicy,
    RowValue,
    TaskRows,
    aggregation_definition,
    enforce_skip_tolerance,
    unweighted_task_mean,
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
from whetstone.code_eval.power import (
    DEFAULT_ALPHA,
    DEFAULT_MDD_PLATEAU_EPSILON,
    DEFAULT_REPEAT_CAP,
    DEFAULT_TARGET_PROB,
    PowerConfig,
    PowerRecommendation,
    PowerResult,
    PowerSurfacePoint,
    VarianceDecomposition,
    analyze_power,
)
from whetstone.code_eval.scoring import (
    COMPRESSED_DESCRIPTION_LENGTH_NAME,
    COMPRESSION_RATIO_NAME,
    ZSTD_LEVEL,
    compressed_description_length_bytes,
    compressed_description_length_fact,
    compression_ratio_score,
    compression_ratio_value,
)
from whetstone.code_eval.statistics import (
    DEFAULT_RESAMPLES,
    BootstrapCI,
    bootstrap_delta_ci,
    bootstrap_mean_ci,
    bootstrap_paired_delta_ci,
    mean,
    resample_indices,
)
from whetstone.code_eval.submission import (
    submission_text,
    submission_text_artifact,
)

__all__ = [
    "COMPRESSED_DESCRIPTION_LENGTH_NAME",
    "COMPRESSION_RATIO_NAME",
    "COMPRESSION_REFERENCE_NAMESPACE",
    "DEFAULT_ALPHA",
    "DEFAULT_MDD_PLATEAU_EPSILON",
    "DEFAULT_REPEAT_CAP",
    "DEFAULT_RESAMPLES",
    "DEFAULT_TARGET_PROB",
    "SELECTED_FIELD",
    "ZSTD_LEVEL",
    "BootstrapCI",
    "CompletenessPolicy",
    "EvaluationMatrixPlan",
    "ExperimentTaskView",
    "PowerConfig",
    "PowerRecommendation",
    "PowerResult",
    "PowerSurfacePoint",
    "RolloutAggregate",
    "RowPolicy",
    "RowValue",
    "TaskRows",
    "VarianceDecomposition",
    "aggregation_definition",
    "analyze_power",
    "bootstrap_delta_ci",
    "bootstrap_mean_ci",
    "bootstrap_paired_delta_ci",
    "build_resolver",
    "compressed_description_length_bytes",
    "compressed_description_length_fact",
    "compression_ratio_score",
    "compression_ratio_value",
    "compression_reference_binding",
    "compression_reference_key",
    "enforce_skip_tolerance",
    "mean",
    "resample_indices",
    "select_compression_reference",
    "submission_text",
    "submission_text_artifact",
    "unweighted_task_mean",
]
