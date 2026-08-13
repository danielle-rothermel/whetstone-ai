from __future__ import annotations

from collections.abc import Sequence

from whetstone.core.identity import ImmutableJsonObject, TypedRef
from whetstone.eval.metadata import metadata_with_purpose
from whetstone.eval.protocol import (
    EvalEvidenceWithRef,
    EvalRequest,
    EvalEngine,
    eval_is_rejected,
    eval_is_success,
)
from whetstone.optim.tools.contracts import ToolCall, ToolConfig
from whetstone.optim.tools.execution import (
    ToolEvaluation,
    ToolEvaluationError,
    ToolValidationError,
)


class EngineToolEvaluator:
    def __init__(self, engine: EvalEngine) -> None:
        self._engine = engine

    def validate(self, call: ToolCall, config: ToolConfig) -> None:
        self._resolve_engine(call, config)

    def _resolve_engine(
        self, call: ToolCall, config: ToolConfig
    ) -> EvalEngine:
        if config.eval_config_hash != (
            self._engine.eval_config_ref.config_hash
        ):
            raise ToolValidationError(
                "tool config is not bound to the engine's exact Eval Config"
            )
        model_route = call.args.get("model_route")
        expected_model_route = self._engine.expected_model_route()
        if model_route != expected_model_route:
            raise ToolValidationError(
                "tool call model_route must match the engine's exact "
                "Provider Call Config route"
            )
        engine = self._engine
        task_ids = call.args.get("task_ids")
        if task_ids is not None:
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
        payload = {config.candidate_template_field: template}
        candidate = Candidate(
            candidate_id=call.call_id,
            base_ref=TypedRef.model_validate(call.args["base_ref"]),
            payload=payload,
        )
        result = engine.evaluate(
            EvalRequest(
                request_id=f"tool:{call.call_id}",
                candidate=candidate,
                metadata=metadata_with_purpose(config.tool_name),
            )
        )
        if eval_is_rejected(result):
            raise ToolValidationError(result.detail.message)
        if not isinstance(result, EvalEvidenceWithRef):
            raise ToolEvaluationError(
                f"unexpected evaluation result: {result!r}"
            )
        if isinstance(result.evidence, EvalFailureEvidence):
            raise ToolEvaluationError(result.evidence.message)
        if not eval_is_success(result):
            raise ToolEvaluationError(
                f"unexpected evaluation result: {result!r}"
            )
        evidence = result.evidence
        assert isinstance(evidence, EvalEvidence)
        available_output = {
            "evaluation_evidence_ref": result.evidence_ref.model_dump(
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
            generation_refs=(result.evidence_ref,),
            aggregates={evidence.aggregate_name: evidence.aggregate_value},
            eval_config_hash=evidence.eval_config_ref.config_hash,
        )


__all__ = ["EngineToolEvaluator"]
