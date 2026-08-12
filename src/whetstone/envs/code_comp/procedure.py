from __future__ import annotations

from dr_code.humaneval.plus_dataset import HF_REVISION

from whetstone.envs.code_comp.constants import (
    CODE_COMP_DATASET_ID,
    CODE_COMP_DATASET_REVISION,
    CODE_COMP_ENV_NAME,
    CODE_COMP_SUBMISSION_SCORE_NAME,
    DEFINITION_VERSION,
)
from whetstone.envs.code_comp.scoring import (
    CODE_COMP_SCORING_PROFILE_ID,
    CODE_COMP_SCORING_PROFILE_VERSION,
)
from whetstone.evaluation import (
    EvaluationProcedureConfig,
    EvaluationProcedureDefinition,
    MetricExtractionConfig,
    MetricExtractionDefinition,
    MetricQuestionBinding,
    PreprocessingDefinition,
)


def build_code_eval_procedure_config(
    *,
    env_name: str,
    primary_metric_name: str,
    primary_metric_settings: tuple[tuple[str, str], ...],
    zero_denominator: str = "not_applicable",
    compression_algorithm: str = "zstd",
    compression_level: int = 19,
    compression_reference: str = "task.gt_code_wo_comments",
) -> EvaluationProcedureConfig:
    """Build one enc-dec code-eval procedure with a concrete primary metric."""
    if compression_algorithm != "zstd":
        raise ValueError(
            f"unsupported compression algorithm {compression_algorithm!r}"
        )
    definition = MetricExtractionDefinition(
        definition_id=f"whetstone.{env_name}.code_eval",
        version=DEFINITION_VERSION,
        questions=(
            MetricQuestionBinding(
                metric=primary_metric_name,
                on="submission",
                settings=primary_metric_settings,
            ),
            MetricQuestionBinding(
                metric="whetstone.code_comp.compression_ratio",
                on="description",
                settings=(
                    ("zstd_level", str(compression_level)),
                    ("reference", compression_reference),
                ),
            ),
        ),
    )
    metric_extraction = MetricExtractionConfig._create(
        definition=definition,
        assignment={},
        resolved_operators=(
            (f"whetstone.{env_name}.code_eval_operator", "1"),
        ),
    )
    preprocessing = PreprocessingDefinition(
        definition_id=f"whetstone.{env_name}.preprocess",
        version=DEFINITION_VERSION,
        steps=(),
    ).materialize()
    return EvaluationProcedureDefinition(
        definition_id=f"whetstone.{env_name}.procedure",
        version=DEFINITION_VERSION,
    ).materialize(
        preprocessing=preprocessing,
        metric_extraction=metric_extraction,
        assignment={"zero_denominator": zero_denominator},
    )


def build_encdec_procedure_config(
    *,
    zero_denominator: str = "not_applicable",
    compression_level: int = 19,
    compression_reference: str = "task.gt_code_wo_comments",
) -> EvaluationProcedureConfig:
    """The canonical ED1 HumanEval-submission evaluation procedure."""
    return build_code_eval_procedure_config(
        env_name=CODE_COMP_ENV_NAME,
        primary_metric_name=CODE_COMP_SUBMISSION_SCORE_NAME,
        primary_metric_settings=(
            ("dataset", CODE_COMP_DATASET_ID),
            ("dataset_coordinate", CODE_COMP_DATASET_REVISION),
            ("upstream_revision", HF_REVISION),
            ("scorer", "dr_code.humaneval.score_humaneval_submission"),
            ("scoring_profile_id", CODE_COMP_SCORING_PROFILE_ID),
            ("scoring_profile_version", CODE_COMP_SCORING_PROFILE_VERSION),
            ("completed_outcome_projection", "definitive_score"),
        ),
        zero_denominator=zero_denominator,
        compression_level=compression_level,
        compression_reference=compression_reference,
    )


__all__ = [
    "build_code_eval_procedure_config",
    "build_encdec_procedure_config",
]
