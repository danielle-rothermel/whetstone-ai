from __future__ import annotations

from collections.abc import Sequence

from whetstone.core.identity import ImmutableJsonObject, TypedRef
from whetstone.core.roles import EvaluationRole
from whetstone.envs.code_comp.constants import (
    CODE_COMP_ENV_NAME,
    MUTATION_FIELD,
)
from whetstone.evaluation.engine import EvaluationEngine, EvaluationRequest
from whetstone.experiment.binding import (
    EVALUATION_BINDING_SCHEMA_VERSION,
    EvaluationBinding,
)
from whetstone.experiment.candidate import Candidate
from whetstone.optimization.tools.contracts import ToolCall, ToolConfig
from whetstone.optimization.tools.execution import (
    ToolEvaluation,
    ToolValidationError,
)


class EngineToolEvaluator:
    def __init__(self, engine: EvaluationEngine) -> None:
        self._engine = engine

    def validate(self, call: ToolCall, config: ToolConfig) -> None:
        """Refuse an unevaluatable Call without consuming Tool Capacity."""
        self._resolve_engine(call, config)

    def _resolve_engine(
        self, call: ToolCall, config: ToolConfig
    ) -> EvaluationEngine:
        if config.eval_config_hash != (
            self._engine.eval_config_ref.config_hash
        ):
            raise ToolValidationError(
                "tool config is not bound to the engine's exact Eval Config"
            )
        model_route = call.args.get("model_route")
        provider_config = (
            self._engine.experiment.generation_graph.provider_call_config
        )
        expected_model_route = provider_config.definition.route.model
        if model_route != expected_model_route:
            raise ToolValidationError(
                "tool call model_route must match the engine's exact "
                "Provider Call Config route"
            )
        engine = self._engine
        task_ids = call.args.get("task_ids")
        if task_ids is not None:
            # Frozen JSON args render arrays as tuples, so accept any
            # non-string ordered sequence of strings.
            if isinstance(task_ids, (str, bytes)) or not isinstance(
                task_ids, Sequence
            ):
                raise ToolValidationError(
                    "tool task_ids must be an ordered list of strings"
                )
            resolved: list[str] = []
            for task_id in task_ids:
                if not isinstance(task_id, str):
                    raise ToolValidationError(
                        "tool task_ids must be an ordered list of strings"
                    )
                resolved.append(task_id)
            try:
                engine = self._engine.for_task_ids(tuple(resolved))
            except ValueError as exc:
                raise ToolValidationError(str(exc)) from exc
        return engine

    def evaluate(self, call: ToolCall, config: ToolConfig) -> ToolEvaluation:
        engine = self._resolve_engine(call, config)
        template = call.args.get("template")
        if engine.experiment.env_name == CODE_COMP_ENV_NAME:
            payload = {MUTATION_FIELD: template}
        else:
            payload = {"user_prompt_template": template}
        candidate = Candidate(
            candidate_id=call.call_id,
            base_ref=TypedRef.model_validate(call.args["base_ref"]),
            payload=payload,
        )
        evaluated = engine.evaluate(
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
            generation_refs=(evaluated.evidence_ref,),
            aggregates={evidence.aggregate_name: evidence.aggregate_value},
            eval_config_hash=(
                evidence.evaluation_binding.eval_config.config_hash
            ),
        )


__all__ = ["EngineToolEvaluator"]
