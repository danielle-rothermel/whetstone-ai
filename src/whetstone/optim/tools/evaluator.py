from __future__ import annotations

from whetstone.core.identity import (
    ImmutableJsonObject,
    TerminalFailure,
    TypedRef,
)
from whetstone.eval.metadata import metadata_with_purpose
from whetstone.eval.protocol import (
    EvalEvidenceWithRef,
    EvalRequest,
    EvalEngine,
    eval_is_rejected,
    eval_is_success,
)
from whetstone.eval.schema import EvalEvidence, EvalFailureEvidence
from whetstone.experiment.candidate import Candidate
from whetstone.optim.tools.contracts import ToolCall, ToolConfig
from whetstone.optim.tools.execution import (
    ToolEvaluation,
    ToolEvaluationError,
    ToolValidationError,
)

#: Terminal failure codes this evaluator owns. They are persisted on the
#: Tool Result and read back by the adapter, so they are named constants
#: rather than call-site literals.
TOOL_EVAL_FAILURE_EVIDENCE_CODE = "tool_eval_failure_evidence"
TOOL_EVAL_UNEXPECTED_RESULT_CODE = "tool_eval_unexpected_result"

#: The ``TerminalFailure.details`` key carrying the typed ref of the
#: persisted ``EvalFailureEvidence`` behind a failed tool evaluation. It
#: is the only citation of that evidence on a failed Tool Result -- the
#: executor leaves ``evaluation_evidence_refs`` empty there -- so run
#: cost reads the rows already paid for through this exact key.
TOOL_EVAL_FAILURE_EVIDENCE_REF_KEY = "evidence_ref"


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
        return self._engine

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
                TerminalFailure(
                    code=TOOL_EVAL_UNEXPECTED_RESULT_CODE,
                    message=(
                        "Tool evaluation produced an unrecognized Eval Result"
                    ),
                    details={"result_type": type(result).__name__},
                )
            )
        if isinstance(result.evidence, EvalFailureEvidence):
            raise ToolEvaluationError(
                TerminalFailure(
                    code=TOOL_EVAL_FAILURE_EVIDENCE_CODE,
                    message=(
                        "Tool evaluation started but produced terminal "
                        "failure evidence"
                    ),
                    details={
                        "exception_type": result.evidence.exception_type,
                        "message": result.evidence.message,
                        TOOL_EVAL_FAILURE_EVIDENCE_REF_KEY: (
                            result.evidence_ref.model_dump(mode="json")
                        ),
                    },
                )
            )
        if not eval_is_success(result):
            raise ToolEvaluationError(
                TerminalFailure(
                    code=TOOL_EVAL_UNEXPECTED_RESULT_CODE,
                    message=(
                        "Tool evaluation produced an unrecognized Eval Result"
                    ),
                    details={
                        "evidence_type": type(result.evidence).__name__,
                    },
                )
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
            # An unobserved task crosses the wire as JSON ``null``, not 0.0:
            # the reading agent must be able to tell a task that scored zero
            # from one that was never measured. ``per_task_counts`` reads 0
            # for exactly those tasks.
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


__all__ = [
    "TOOL_EVAL_FAILURE_EVIDENCE_CODE",
    "TOOL_EVAL_FAILURE_EVIDENCE_REF_KEY",
    "TOOL_EVAL_UNEXPECTED_RESULT_CODE",
    "EngineToolEvaluator",
]
