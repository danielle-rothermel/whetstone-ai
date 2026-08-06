from __future__ import annotations

from dr_code.eval import (
    EvaluationProcedureConfig,
    EvaluationProcedureDefinition,
    MetricExtractionConfig,
    MetricExtractionDefinition,
    MetricQuestionBinding,
    PreprocessingDefinition,
)

from whetstone.envs.oracle_operator import (
    ENV_EXACT_MATCH_NAME,
    ENV_ORACLE_OPERATOR_NAME,
    ENV_ORACLE_OPERATOR_VERSION,
)
from whetstone.envs.registry import EnvSpec

#: The dr-code Evaluation Procedure Config schema name, referenced by the
#: Eval Node's static Variable typed reference.
EVALUATION_PROCEDURE_CONFIG_SCHEMA = "dr_code.evaluation_procedure.config"

_DEFINITION_VERSION = "1"
#: The Metric Question keys onto the LLM Call Node's generation output.
_METRIC_ON = "generation"


def env_metric_extraction_config(env: EnvSpec) -> MetricExtractionConfig:
    """The env's Metric Extraction Config (folds in the oracle wiring).

    One Metric Question -- ``env_exact_match`` on the generation -- whose
    settings name the env and its oracle entry point, built via dr-code's
    sole-owner constructor with the whetstone operator's explicit resolved
    version (the operator is whetstone-owned, not registry-resolved).
    """
    definition = MetricExtractionDefinition(
        definition_id=f"whetstone.env_oracle.{env.name}",
        version=_DEFINITION_VERSION,
        questions=(
            MetricQuestionBinding(
                metric=ENV_ORACLE_OPERATOR_NAME,
                on=_METRIC_ON,
                settings=(
                    ("env", env.name),
                    ("metric_name", ENV_EXACT_MATCH_NAME),
                    ("oracle", env.oracle_qualname),
                ),
            ),
        ),
    )
    return MetricExtractionConfig._create(
        definition=definition,
        assignment={},
        resolved_operators=(
            (ENV_ORACLE_OPERATOR_NAME, ENV_ORACLE_OPERATOR_VERSION),
        ),
    )


def env_procedure_config(
    env: EnvSpec,
    *,
    zero_denominator: str = "not_applicable",
) -> EvaluationProcedureConfig:
    """Build the env's Evaluation Procedure Config.

    Composes an empty Preprocessing Config (the oracle owns normalization)
    with the env-specific Metric Extraction Config. The returned Config's
    ``config_identity_hash`` is the Evaluation Procedure Config identity the
    Eval Node references and the composite Eval Config folds in.
    """
    preprocessing = PreprocessingDefinition(
        definition_id="whetstone.env_procedure.preprocess",
        version=_DEFINITION_VERSION,
        steps=(),
    ).materialize()
    metric_extraction = env_metric_extraction_config(env)
    return EvaluationProcedureDefinition(
        definition_id=f"whetstone.env_procedure.{env.name}",
        version=_DEFINITION_VERSION,
    ).materialize(
        preprocessing=preprocessing,
        metric_extraction=metric_extraction,
        assignment={"zero_denominator": zero_denominator},
    )


__all__ = [
    "EVALUATION_PROCEDURE_CONFIG_SCHEMA",
    "env_metric_extraction_config",
    "env_procedure_config",
]
