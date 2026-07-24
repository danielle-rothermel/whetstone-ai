"""PR4 ToolEvaluator projection onto the canonical evaluation engine."""

from __future__ import annotations

from whetstone.evaluation.engine import EvaluationEngine, EvaluationRequest
from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization.schema import Candidate
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
        candidate = Candidate(
            candidate_id=call.call_id,
            base_ref=str(call.args.get("base_ref", call.call_id)),
            payload={
                "user_prompt_template": call.args.get("template"),
                **(
                    {"model_route": call.args["model_route"]}
                    if "model_route" in call.args
                    else {}
                ),
            },
        )
        evaluated = self._engine.evaluate(
            EvaluationRequest(
                candidate=candidate,
                evaluation_role=EvaluationRole.INTERNAL,
                evaluation_context_id=call.call_id,
                purpose=config.tool_name,
            )
        )
        evidence = evaluated.evidence
        template = call.args.get("template")
        objectives = {
            "correctness": evidence.aggregate_value or 0.0,
            "compression": float(len(template))
            if isinstance(template, str)
            else 0.0,
        }
        return ToolEvaluation(
            rollout_refs=(evaluated.evidence_ref,),
            aggregates={evidence.aggregate_name: evidence.aggregate_value},
            eval_config_hash=evidence.eval_config.identity_hash,
            objective_values=objectives,
            extra_output={
                "evaluation_evidence_ref": (
                    evaluated.evidence_ref.model_dump(mode="json")
                ),
                "output_artifact_ref": evidence.outputs_ref.model_dump(
                    mode="json"
                ),
                "per_task_values": list(evidence.per_task_values),
                "per_task_counts": list(evidence.per_task_counts),
                "row_accounting": evidence.row_accounting.model_dump(
                    mode="json"
                ),
            },
        )


__all__ = ["EngineToolEvaluator"]
