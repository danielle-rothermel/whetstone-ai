from __future__ import annotations

from whetstone.core.roles import EvaluationRole
from whetstone.evaluation.engine import EvaluationEngine
from whetstone.experiment.binding import (
    EVALUATION_BINDING_SCHEMA_VERSION,
    EvaluationBinding,
    ExecutionEnvironmentFingerprint,
)

__all__ = ["preview_evaluation_binding"]


def preview_evaluation_binding(
    engine: EvaluationEngine,
    *,
    campaign: str,
    provenance_note: str,
    environment_fingerprint: ExecutionEnvironmentFingerprint,
    role: EvaluationRole = EvaluationRole.INTERNAL,
) -> EvaluationBinding:
    return EvaluationBinding(
        schema_version=EVALUATION_BINDING_SCHEMA_VERSION,
        eval_config=engine.eval_config_ref,
        role=role,
        campaign=campaign,
        provider_execution_policy_ref=engine.provider_execution_policy_ref,
        environment_fingerprint=environment_fingerprint,
        provenance_note=provenance_note,
    )
