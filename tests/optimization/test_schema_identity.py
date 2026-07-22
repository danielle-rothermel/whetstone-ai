"""Schema and identity coverage for the Optimization Step protocol.

Covers every Step Request field group, the rejection of Runtime Tool Handles
and mutable process objects, the Evaluation Intent's no-score contract, the
exactly-one-status rule, and the immutable-prior-reference resolution.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization import (
    STEP_REQUEST_SCHEMA,
    BudgetState,
    EvaluationIntent,
    OptimizationRun,
    OptimizationStepRequest,
    RuntimeToolHandle,
    StepKind,
    StepMode,
    ToolConfig,
    TypedRef,
    step_request_reference,
)

from .support import (
    FULL_A,
    FULL_B,
    candidate,
    make_intent,
    make_tool_definition_config,
    output_contract,
    proposal_request,
    pure_request,
)


def test_step_request_carries_every_field_group() -> None:
    cfg = make_tool_definition_config()
    prior = TypedRef(schema_name="whetstone.optimization_step_result",
                     content_hash=FULL_B)
    req = OptimizationStepRequest(
        run_id="run-1",
        step_id="step-1",
        optimizer_config_hash=FULL_A,
        mode=StepMode.TOOL_USING,
        kind=StepKind.TOOL,
        kind_label="gepa_step",
        step_index=1,
        prior_step_result_ref=prior,
        prior_state_ref=TypedRef(schema_name="s", content_hash=FULL_A),
        prior_history_ref=TypedRef(schema_name="h", content_hash=FULL_A),
        candidates=(candidate("A"),),
        pools={"instruction_pool": ["i0", "i1"]},
        hyperparameters={"breadth": 4, "depth": 2},
        budget=BudgetState(
            consumed={"rollouts": 10}, remaining={"rollouts": 90}
        ),
        output_contract=output_contract(3),
        tool_configs=(cfg,),
    )
    # Every group is present and self-describing.
    assert req.run_id and req.step_id and req.optimizer_config_hash
    assert req.kind is StepKind.TOOL and req.step_index == 1
    assert req.prior_step_result_ref is prior
    assert req.pools["instruction_pool"] == ["i0", "i1"]
    assert req.hyperparameters["breadth"] == 4
    assert req.budget.remaining["rollouts"] == 90
    assert req.output_contract.returned_proposal_count == 3
    assert req.tool_configs[0].tool_name == "evaluate_candidate"


def test_step_request_rejects_runtime_tool_handle() -> None:
    handle = RuntimeToolHandle(
        make_tool_definition_config(),
        lambda call: None,  # type: ignore
    )
    with pytest.raises(ValidationError):
        OptimizationStepRequest(
            run_id="run-1",
            step_id="step-1",
            optimizer_config_hash=FULL_A,
            mode=StepMode.TOOL_USING,
            kind=StepKind.TOOL,
            step_index=0,
            output_contract=output_contract(1),
            # A Runtime Tool Handle is a plain non-JSON object; the Tool Config
            # slot only accepts serialized ToolConfigs.
            tool_configs=(handle,),
        )


def test_step_request_rejects_mutable_process_object() -> None:
    class _LiveClient:
        pass

    with pytest.raises(ValidationError):
        OptimizationStepRequest(
            run_id="run-1",
            step_id="step-1",
            optimizer_config_hash=FULL_A,
            mode=StepMode.PROPOSAL_ONLY,
            kind=StepKind.PROPOSAL,
            step_index=0,
            output_contract=output_contract(1),
            # A live client is not strict-JSON; the request refuses it.
            hyperparameters={"client": _LiveClient()},
        )


def test_step_request_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        OptimizationStepRequest(
            run_id="run-1",
            step_id="step-1",
            optimizer_config_hash=FULL_A,
            mode=StepMode.PURE,
            kind=StepKind.IDENTITY,
            step_index=0,
            output_contract=output_contract(1),
            runtime_tool_handle="anything",  # type: ignore
        )


def test_initial_step_has_no_prior_result_ref() -> None:
    with pytest.raises(ValidationError):
        OptimizationStepRequest(
            run_id="run-1",
            step_id="step-1",
            optimizer_config_hash=FULL_A,
            mode=StepMode.PROPOSAL_ONLY,
            kind=StepKind.PROPOSAL,
            step_index=0,
            output_contract=output_contract(1),
            prior_step_result_ref=TypedRef(
                schema_name="whetstone.optimization_step_result",
                content_hash=FULL_B,
            ),
        )


def test_noninitial_step_requires_prior_result_ref() -> None:
    with pytest.raises(ValidationError):
        OptimizationStepRequest(
            run_id="run-1",
            step_id="step-2",
            optimizer_config_hash=FULL_A,
            mode=StepMode.PROPOSAL_ONLY,
            kind=StepKind.PROPOSAL,
            step_index=1,
            output_contract=output_contract(1),
        )


def test_later_request_resolves_immutable_prior_reference() -> None:
    # A later request names the prior Step Result strictly by TypedRef; the
    # reference is content-addressed and immutable.
    first = proposal_request(step_index=0)
    first_ref = step_request_reference(first)
    later = proposal_request(
        step_index=1, prior_step_result_ref=TypedRef(
            schema_name="whetstone.optimization_step_result",
            content_hash=first_ref.content_hash,
        )
    )
    assert later.prior_step_result_ref is not None
    assert later.prior_step_result_ref.content_hash == first_ref.content_hash
    # The prior reference denotes a typed dr-store Object Reference.
    ref = later.prior_step_result_ref.reference
    assert ref.content_hash == first_ref.content_hash


def test_evaluation_intent_has_no_score_field() -> None:
    intent = make_intent()
    dumped = intent.model_dump()
    assert "score" not in dumped
    assert "result" not in dumped
    # Attempting to add a score is rejected by extra="forbid".
    with pytest.raises(ValidationError):
        EvaluationIntent(
            intent_id="i1",
            candidate_id="P1",
            target_eval_config_ref="evalcfg://x",
            target_eval_config_hash=FULL_B,
            context_role=EvaluationRole.INTERNAL,
            purpose="minibatch",
            run_id="run-1",
            step_index=0,
            score=0.9,  # type: ignore
        )


def test_evaluation_intent_requires_full_target_hash() -> None:
    with pytest.raises(ValidationError):
        make_intent(target_hash="tooshort")


def test_pure_mode_requires_identity_kind() -> None:
    with pytest.raises(ValidationError):
        OptimizationStepRequest(
            run_id="run-1",
            step_id="step-1",
            optimizer_config_hash=FULL_A,
            mode=StepMode.PURE,
            kind=StepKind.PROPOSAL,
            step_index=0,
            output_contract=output_contract(1),
        )


def test_tool_using_request_requires_a_tool_config() -> None:
    with pytest.raises(ValidationError):
        OptimizationStepRequest(
            run_id="run-1",
            step_id="step-1",
            optimizer_config_hash=FULL_A,
            mode=StepMode.TOOL_USING,
            kind=StepKind.TOOL,
            step_index=0,
            output_contract=output_contract(1),
        )


def test_proposal_only_request_rejects_tool_configs() -> None:
    with pytest.raises(ValidationError):
        OptimizationStepRequest(
            run_id="run-1",
            step_id="step-1",
            optimizer_config_hash=FULL_A,
            mode=StepMode.PROPOSAL_ONLY,
            kind=StepKind.PROPOSAL,
            step_index=0,
            output_contract=output_contract(1),
            tool_configs=(make_tool_definition_config(),),
        )


def test_optimization_run_identity_is_stable_and_covers_tools() -> None:
    cfg = make_tool_definition_config()
    run_a = OptimizationRun(
        run_id="r1",
        optimizer_config_hash=FULL_A,
        mode=StepMode.TOOL_USING,
        output_contract=output_contract(4),
        tool_configs=(cfg,),
    )
    run_b = OptimizationRun(
        run_id="r1",
        optimizer_config_hash=FULL_A,
        mode=StepMode.TOOL_USING,
        output_contract=output_contract(4),
        tool_configs=(cfg,),
    )
    assert run_a.identity_hash() == run_b.identity_hash()
    assert len(run_a.identity_hash()) == 64


def test_optimization_run_rejects_tools_when_not_tool_using() -> None:
    with pytest.raises(ValidationError):
        OptimizationRun(
            run_id="r1",
            optimizer_config_hash=FULL_A,
            mode=StepMode.PROPOSAL_ONLY,
            output_contract=output_contract(1),
            tool_configs=(make_tool_definition_config(),),
        )


def test_step_request_reference_is_content_addressed() -> None:
    req = pure_request()
    ref = step_request_reference(req)
    assert ref.schema_name == STEP_REQUEST_SCHEMA
    assert len(ref.content_hash) == 64
    # Same request -> same reference.
    again = step_request_reference(pure_request())
    assert again.content_hash == ref.content_hash


def test_candidate_and_toolconfig_are_frozen() -> None:
    cand = candidate("A")
    with pytest.raises(ValidationError):
        cand.candidate_id = "B"  # type: ignore
    cfg: ToolConfig = make_tool_definition_config()
    with pytest.raises(ValidationError):
        cfg.endpoint = "other"  # type: ignore
