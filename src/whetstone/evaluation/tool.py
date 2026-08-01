"""Tool evaluator projection onto the canonical evaluation engine."""

from __future__ import annotations

from whetstone.evaluation.engine import EvaluationEngine, EvaluationRequest
from whetstone.evaluation_role import EvaluationRole
from whetstone.optimization.identity import ImmutableJsonObject, TypedRef
from whetstone.optimization.schema import (
    EVALUATION_BINDING_SCHEMA_VERSION,
    Candidate,
    EvaluationBinding,
)
from whetstone.optimization.tool_eval import (
    ToolEvaluation,
    ToolValidationError,
)
from whetstone.optimization.tools import ToolCall, ToolConfig


class EngineToolEvaluator:
    def __init__(self, engine: EvaluationEngine) -> None:
        self._engine = engine

    def evaluate(self, call: ToolCall, config: ToolConfig) -> ToolEvaluation:
        if config.eval_config_identity_hash != (
            self._engine.eval_config_ref.identity_hash
        ):
            raise ToolValidationError(
                "tool config is not bound to the engine's exact Eval Config"
            )
        model_route = call.args.get("model_route")
        provider_config = (
            self._engine.experiment.rollout_definition.provider_call_config
        )
        expected_model_route = provider_config.definition.route.model
        if model_route != expected_model_route:
            raise ToolValidationError(
                "tool call model_route must match the engine's exact "
                "Provider Call Config route"
            )
        candidate = Candidate(
            candidate_id=call.call_id,
            base_ref=TypedRef.model_validate(call.args["base_ref"]),
            payload={
                "user_prompt_template": call.args.get("template"),
            },
        )
        evaluated = self._engine.evaluate(
            EvaluationRequest(
                candidate=candidate,
                evaluation_binding=EvaluationBinding(
                    schema_version=EVALUATION_BINDING_SCHEMA_VERSION,
                    eval_config=self._engine.eval_config_ref,
                    role=EvaluationRole.INTERNAL,
                    campaign=config.store_namespace_key,
                    provider_execution_policy_ref=(
                        self._engine.provider_execution_policy_ref
                    ),
                ),
                purpose=config.tool_name,
            )
        )
        evidence = evaluated.evidence
        available_output = {
            "evaluation_evidence_ref": evaluated.evidence_ref.model_dump(
                mode="json"
            ),
            "output_artifact_ref": evidence.outputs_ref.model_dump(
                mode="json"
            ),
            "per_task_values": list(evidence.per_task_values),
            "per_task_counts": list(evidence.per_task_counts),
            "row_accounting": evidence.row_accounting.model_dump(mode="json"),
        }
        unsupported = tuple(
            field
            for field in config.definition.record.output_fields
            if field not in available_output
        )
        if unsupported:
            raise ToolValidationError(
                "tool definition declares unsupported engine output fields: "
                + ", ".join(unsupported)
            )
        return ToolEvaluation(
            output=ImmutableJsonObject(
                {
                    field: available_output[field]
                    for field in config.definition.record.output_fields
                }
            ),
            rollout_refs=(evaluated.evidence_ref,),
            aggregates={evidence.aggregate_name: evidence.aggregate_value},
            eval_config_hash=evidence.evaluation_binding.eval_config.identity_hash,
        )


__all__ = ["EngineToolEvaluator"]
