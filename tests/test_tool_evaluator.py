"""``EngineToolEvaluator``'s success and failure paths.

Every one of its failure paths was previously dead: the module referenced
three names it never imported, and it raised ``ToolEvaluationError`` with
a plain string where the constructor takes a ``TerminalFailure``. These
tests exercise each path, so a repeat of either defect is a red test
rather than a ``NameError`` inside a leased effect.
"""

from __future__ import annotations

import pytest
from dr_store.sync import open_sqlite

from tests.codex_support import (
    toy_capacity_binding,
    toy_codex_control,
    toy_codex_run,
    toy_tool_args,
)
from whetstone.core.identity import ImmutableJsonObject, TypedRef
from whetstone.eval.protocol import EvalEvidenceWithRef, EvalRejected
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.eval.schema import EvalFailureEvidence
from whetstone.optim.contracts import ResolutionClass, ResolutionDetail
from whetstone.optim.tools.contracts import ToolCall, tool_config_reference
from whetstone.optim.tools.evaluator import (
    TOOL_EVAL_FAILURE_EVIDENCE_CODE,
    TOOL_EVAL_UNEXPECTED_RESULT_CODE,
    EngineToolEvaluator,
)
from whetstone.optim.tools.execution import (
    ToolEvaluationError,
    ToolValidationError,
)


class _ScriptedEngine:
    """An engine that returns one scripted Eval Result.

    It delegates every identity accessor to a real reference engine, so
    the evaluator's config and route checks pass for the right reasons.
    """

    def __init__(self, inner, result) -> None:
        self._inner = inner
        self._result = result

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def evaluate(self, request):
        return self._result

    def for_task_ids(self, task_ids):
        return _ScriptedEngine(self._inner.for_task_ids(task_ids), self._result)


@pytest.fixture
def codex_call_and_config(tmp_path):
    """One valid Tool Call plus the engine and config it is bound to."""
    with open_sqlite(str(tmp_path / "evaluator.sqlite")) as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(store)
        control = toy_codex_control(engine=engine)
        run, config, candidate = toy_codex_run(
            control=control, engine=engine
        )
        call = ToolCall(
            call_id="c1",
            tool_config=tool_config_reference(config),
            capacity_binding=toy_capacity_binding(run),
            args=ImmutableJsonObject(
                toy_tool_args(
                    candidate=candidate,
                    engine=engine,
                    template="Answer {prompt} briefly.",
                )
            ),
        )
        yield call, config, engine


def _failure_evidence(candidate_ref, engine) -> EvalFailureEvidence:
    from whetstone.core.roles import EvalRole

    return EvalFailureEvidence(
        candidate=candidate_ref,
        eval_config_ref=engine.eval_config_ref,
        eval_role=EvalRole.INTERNAL,
        exception_type="RuntimeError",
        message="the provider refused the request",
    )


def test_a_successful_engine_result_becomes_a_tool_evaluation(
    codex_call_and_config,
) -> None:
    call, config, engine = codex_call_and_config

    evaluation = EngineToolEvaluator(engine).evaluate(call, config)

    assert set(evaluation.output) == set(
        config.definition.record.output_fields
    )
    assert evaluation.eval_config_hash == config.eval_config_hash
    assert len(evaluation.generation_refs) == 1
    assert evaluation.aggregates


def test_a_rejected_result_is_a_pre_execution_validation_refusal(
    codex_call_and_config,
) -> None:
    call, config, engine = codex_call_and_config
    scripted = _ScriptedEngine(
        engine,
        EvalRejected(
            detail=ResolutionDetail(
                classification=ResolutionClass.VALIDATION,
                message="the candidate does not bind this task set",
            )
        ),
    )

    with pytest.raises(ToolValidationError, match="does not bind this task"):
        EngineToolEvaluator(scripted).evaluate(call, config)


def test_failure_evidence_raises_a_terminal_failure(
    codex_call_and_config,
) -> None:
    call, config, engine = codex_call_and_config
    from whetstone.experiment.candidate import Candidate, candidate_reference

    candidate_ref = candidate_reference(
        Candidate(
            candidate_id="c1",
            base_ref=TypedRef.model_validate(call.args["base_ref"]),
            payload={config.candidate_template_field: "Answer {prompt}."},
        )
    )
    scripted = _ScriptedEngine(
        engine,
        EvalEvidenceWithRef(
            evidence=_failure_evidence(candidate_ref, engine),
            evidence_ref=TypedRef(
                schema_name="whetstone.eval_failure",
                content_hash="a" * 64,
            ),
        ),
    )

    with pytest.raises(ToolEvaluationError) as caught:
        EngineToolEvaluator(scripted).evaluate(call, config)

    assert caught.value.failure.code == TOOL_EVAL_FAILURE_EVIDENCE_CODE
    assert caught.value.failure.details["exception_type"] == "RuntimeError"


def test_an_unrecognized_result_raises_a_terminal_failure(
    codex_call_and_config,
) -> None:
    call, config, engine = codex_call_and_config
    scripted = _ScriptedEngine(engine, object())

    with pytest.raises(ToolEvaluationError) as caught:
        EngineToolEvaluator(scripted).evaluate(call, config)

    assert caught.value.failure.code == TOOL_EVAL_UNEXPECTED_RESULT_CODE
    assert caught.value.failure.details["result_type"] == "object"


def test_a_mismatched_model_route_refuses_before_execution(
    codex_call_and_config,
) -> None:
    call, config, engine = codex_call_and_config
    diverted = call.model_copy(
        update={
            "args": ImmutableJsonObject(
                {**call.args.to_json(), "model_route": "some/other/route"}
            )
        }
    )

    with pytest.raises(ToolValidationError, match="model_route must match"):
        EngineToolEvaluator(engine).validate(diverted, config)
