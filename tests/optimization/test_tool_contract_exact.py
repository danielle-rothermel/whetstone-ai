"""Exact Tool and Reward serialized-contract regressions."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable

import pytest
from dr_code.eval import DefinitionRef, EvalConfig
from pydantic import ValidationError

from whetstone.evaluation_role import EvaluationRole
from whetstone.optimization.identity import (
    TerminalFailure,
    TypedRef,
    typed_ref_for_record,
)
from whetstone.optimization.reward import (
    REWARD_SCHEMA,
    Reward,
    RewardInputCitation,
    RewardPolicy,
    RewardRef,
    RewardTerm,
    reward_reference,
)
from whetstone.optimization.tools import (
    EVAL_CONFIG_SCHEMA,
    TOOL_CALL_SCHEMA,
    TOOL_CONFIG_SCHEMA,
    TOOL_RESULT_SCHEMA,
    RefusalClass,
    ToolCall,
    ToolCallRef,
    ToolCapacity,
    ToolCapacityBinding,
    ToolCapacityScope,
    ToolConfig,
    ToolConfigRef,
    ToolDefinition,
    ToolDefinitionRef,
    ToolRefusal,
    ToolResult,
    ToolResultRef,
    tool_call_reference,
    tool_capacity_binding,
    tool_config_reference,
    tool_definition_reference,
    tool_result_reference,
)


def _eval_config() -> EvalConfig:
    return EvalConfig(
        definition_ref=DefinitionRef(
            definition_id="eval",
            version="1",
            schema_name="dr_code.eval.definition",
            identity_hash="a" * 64,
        ),
        sampling_config_hash="b" * 64,
        evaluation_procedure_config_hash="c" * 64,
        aggregation_config_hash="d" * 64,
        config_identity_hash="e" * 64,
    )


def _definition() -> ToolDefinition:
    return ToolDefinition(
        tool_name="evaluate",
        input_fields=("candidate",),
        output_fields=("score",),
    )


def _config(*, reward_policy_hash: str = "f" * 64) -> ToolConfig:
    return ToolConfig(
        definition=tool_definition_reference(_definition()),
        endpoint_key="eval",
        eval_config=_eval_config(),
        reward_policy_hash=reward_policy_hash,
        capacity=ToolCapacity(
            max_accepted_calls=2,
            scope=ToolCapacityScope.RUN,
        ),
        store_namespace_key="tools",
    )


def _call(config: ToolConfig | None = None) -> ToolCall:
    exact_config = config or _config(
        reward_policy_hash=_reward_policy().identity_hash()
    )
    return ToolCall(
        call_id="call-1",
        tool_config=tool_config_reference(exact_config),
        capacity_binding=_run_binding(),
        args={"candidate": "candidate-1"},
    )


def _run_binding(label: str = "run-1") -> ToolCapacityBinding:
    return tool_capacity_binding(
        ToolCapacityScope.RUN,
        typed_ref_for_record(
            "whetstone.optimization_run",
            {"run_id": label},
        ),
    )


def _step_binding(label: str = "step-1") -> ToolCapacityBinding:
    return tool_capacity_binding(
        ToolCapacityScope.STEP,
        typed_ref_for_record(
            "whetstone.optimization_step_request",
            {"step_id": label},
        ),
    )


def _reward_policy(*, policy_name: str = "score/v1") -> RewardPolicy:
    return RewardPolicy(
        policy_name=policy_name,
        terms=(RewardTerm(name="score", weight=1.0),),
    )


def _evidence_ref() -> TypedRef:
    return typed_ref_for_record("whetstone.test.evidence", {"score": 0.75})


def _reward(policy: RewardPolicy | None = None) -> Reward:
    exact_policy = policy or _reward_policy()
    return Reward(
        reward_name="reward",
        value=0.75,
        reward_policy=exact_policy,
        evidence_role=EvaluationRole.INTERNAL,
        input_citations=(
            RewardInputCitation(
                name="score",
                value=0.75,
                contributed=0.75,
            ),
        ),
        evidence_refs=(_evidence_ref(),),
    )


def test_tool_definition_v1_payload_and_digest_are_exact() -> None:
    definition = _definition()
    assert definition.identity_payload() == {
        "tool_name": "evaluate",
        "version": 1,
        "input_fields": ["candidate"],
        "output_fields": ["score"],
        "refusal_classes": [
            "authorization",
            "capacity",
            "budget",
            "validation",
        ],
        "expansion_semantics": None,
    }
    assert (
        definition.identity_hash()
        == "9180692f2deaf763fea956554dd70f58a5efa7c8c4b8e492ae7e90abd759d135"
    )


def test_tool_config_v1_payload_and_digest_are_exact() -> None:
    config = _config()
    assert config.identity_payload() == {
        "definition": {
            "record": {
                "tool_name": "evaluate",
                "version": 1,
                "input_fields": ["candidate"],
                "output_fields": ["score"],
                "refusal_classes": [
                    "authorization",
                    "capacity",
                    "budget",
                    "validation",
                ],
                "expansion_semantics": None,
            },
            "record_ref": {
                "schema_name": "whetstone.tool_definition",
                "content_hash": (
                    "ced70298e924dd8875d12faff456aae4"
                    "cce3cf01e7d5f8aa6e248fdcf53253f3"
                ),
            },
            "identity_hash": (
                "9180692f2deaf763fea956554dd70f58a"
                "5efa7c8c4b8e492ae7e90abd759d135"
            ),
        },
        "endpoint_key": "eval",
        "eval_config": {
            "definition_ref": {
                "definition_id": "eval",
                "version": "1",
                "schema_name": "dr_code.eval.definition",
                "identity_hash": "a" * 64,
            },
            "sampling_config_hash": "b" * 64,
            "evaluation_procedure_config_hash": "c" * 64,
            "aggregation_config_hash": "d" * 64,
            "config_identity_hash": "e" * 64,
        },
        "reward_policy_hash": "f" * 64,
        "capacity": {
            "max_accepted_calls": 2,
            "scope": "run",
        },
        "timeout_policy_ref": None,
        "operational_policy_refs": [],
        "store_namespace_key": "tools",
        "idempotent_replay": True,
    }
    assert (
        config.identity_hash() == "b4fb938b3d60d50f86da6246316314f"
        "54271dcafc817ff9f800b59c96ccba6e8"
    )


def test_reward_policy_v1_payload_and_digest_are_exact() -> None:
    policy = RewardPolicy(
        policy_name="score/v1",
        terms=(RewardTerm(name="score", weight=1.0),),
    )
    assert policy.identity_payload() == {
        "policy_name": "score/v1",
        "reward_name": "reward",
        "terms": [
            {
                "name": "score",
                "weight": 1.0,
                "maximize": True,
                "worst_value": 0.0,
            }
        ],
        "missing_data": "fail",
    }
    assert (
        policy.identity_hash() == "d9c6fa2f12b83f2df1259086c84f4bf8"
        "9732627e416e00c9a0c8c82e07c62ed7"
    )


def test_tool_config_composes_exact_eval_config_and_derives_addresses() -> (
    None
):
    config = _config()
    assert config.eval_config == _eval_config()
    assert config.eval_config_identity_hash == "e" * 64
    assert config.eval_config_ref == typed_ref_for_record(
        EVAL_CONFIG_SCHEMA,
        _eval_config().model_dump(mode="json"),
    )
    forged = config.model_dump(mode="json")
    forged["eval_config_ref"] = config.eval_config_ref.model_dump(mode="json")
    forged["eval_config_identity_hash"] = config.eval_config_identity_hash
    with pytest.raises(ValidationError, match="extra"):
        ToolConfig.model_validate(forged)


def test_capacity_binding_global_run_and_step_contracts_are_exact() -> None:
    global_binding = tool_capacity_binding(ToolCapacityScope.GLOBAL)
    run_binding = _run_binding()
    step_binding = _step_binding()
    assert global_binding.subject_ref is None
    assert global_binding.capacity_scope_id == "global"
    assert run_binding.subject_ref is not None
    assert (
        run_binding.capacity_scope_id == run_binding.subject_ref.content_hash
    )
    assert step_binding.subject_ref is not None
    assert (
        step_binding.capacity_scope_id == step_binding.subject_ref.content_hash
    )

    with pytest.raises(ValidationError, match=r"GLOBAL.*no subject_ref"):
        tool_capacity_binding(
            ToolCapacityScope.GLOBAL,
            typed_ref_for_record("unexpected", {"id": "subject"}),
        )
    with pytest.raises(ValidationError, match=r"RUN.*requires subject_ref"):
        tool_capacity_binding(ToolCapacityScope.RUN)
    with pytest.raises(ValidationError, match=r"whetstone\.optimization_run"):
        tool_capacity_binding(
            ToolCapacityScope.RUN,
            typed_ref_for_record(
                "whetstone.optimization_step_request", {"step_id": "s"}
            ),
        )
    with pytest.raises(
        ValidationError, match=r"whetstone\.optimization_step_request"
    ):
        tool_capacity_binding(
            ToolCapacityScope.STEP,
            typed_ref_for_record(
                "whetstone.optimization_run", {"run_id": "r"}
            ),
        )


def test_tool_call_rejects_free_capacity_scope_id_and_scope_mismatch() -> None:
    call = _call()
    forged = call.model_dump(mode="json")
    forged["capacity_scope_id"] = "forged"
    with pytest.raises(ValidationError, match="extra"):
        ToolCall.model_validate(forged)

    with pytest.raises(ValidationError, match="binding scope must match"):
        ToolCall(
            call_id="call-1",
            tool_config=tool_config_reference(_config()),
            capacity_binding=_step_binding(),
            args={"candidate": "candidate-1"},
        )


@pytest.mark.parametrize(
    "other_binding",
    [_run_binding("other-run"), _step_binding("other-step")],
)
def test_runtime_handle_refuses_other_exact_capacity_subject(
    other_binding: ToolCapacityBinding,
) -> None:
    from whetstone.optimization.tools import RuntimeToolHandle

    config = _config()
    expected_binding = (
        _run_binding()
        if other_binding.scope is ToolCapacityScope.RUN
        else _step_binding()
    )
    if other_binding.scope is ToolCapacityScope.STEP:
        config_data = config.model_dump(mode="json")
        config_data["capacity"]["scope"] = "step"
        config = ToolConfig.model_validate(config_data)
    invocations = 0

    def execute(call: ToolCall) -> ToolResult:
        nonlocal invocations
        invocations += 1
        return ToolResult(
            call=tool_call_reference(call),
            output={"score": 1.0},
            provenance_ordinal=1,
        )

    handle = RuntimeToolHandle(config, expected_binding, execute)
    call = ToolCall(
        call_id="call-1",
        tool_config=tool_config_reference(config),
        capacity_binding=other_binding,
        args={"candidate": "candidate-1"},
    )
    with pytest.raises(ValueError, match="must match the Runtime Tool Handle"):
        handle(call)
    assert invocations == 0


def test_reward_ref_binds_exact_reward_record() -> None:
    reward = _reward()
    exact = reward_reference(reward)
    assert exact.record_ref.schema_name == REWARD_SCHEMA
    with pytest.raises(ValidationError, match="exact Reward record"):
        RewardRef(
            record=reward,
            record_ref=typed_ref_for_record(
                REWARD_SCHEMA, {"different": "record"}
            ),
        )


def test_tool_result_requires_exact_config_reward_policy() -> None:
    policy = _reward_policy()
    call = _call(_config(reward_policy_hash=policy.identity_hash()))
    result = ToolResult(
        call=tool_call_reference(call),
        output={"score": 0.75},
        evaluation_evidence_refs=(_evidence_ref(),),
        reward=reward_reference(_reward(policy)),
        provenance_ordinal=1,
    )
    assert result.reward is not None
    assert (
        result.reward.record.reward_policy_hash
        == call.tool_config.record.reward_policy_hash
    )

    with pytest.raises(ValidationError, match="policy must match"):
        ToolResult(
            call=tool_call_reference(call),
            output={"score": 0.75},
            evaluation_evidence_refs=(_evidence_ref(),),
            reward=reward_reference(
                _reward(_reward_policy(policy_name="other/v1"))
            ),
            provenance_ordinal=1,
        )


@pytest.mark.parametrize(
    ("terminal", "message"),
    [
        (
            {
                "refusal": ToolRefusal(
                    refusal_class=RefusalClass.CAPACITY,
                    reason="capacity exhausted",
                )
            },
            "no evaluation evidence or Reward",
        ),
        (
            {
                "terminal_failure": TerminalFailure(
                    code="failed",
                    message="evaluation failed",
                )
            },
            "failed Tool Result carries no Reward",
        ),
    ],
)
def test_non_success_result_forbids_reward(
    terminal: dict[str, object],
    message: str,
) -> None:
    value = {
        "call": tool_call_reference(_call()),
        "reward": reward_reference(_reward()),
        **terminal,
    }
    with pytest.raises(ValidationError, match=message):
        ToolResult.model_validate(value)


@pytest.mark.parametrize("terminal_variant", ["success", "failure"])
@pytest.mark.parametrize("hostile_ordinal", ["missing", 0])
def test_serialized_non_refused_result_requires_positive_provenance_ordinal(
    terminal_variant: str,
    hostile_ordinal: str | int,
) -> None:
    result = ToolResult(
        call=tool_call_reference(_call()),
        output={"score": 1.0} if terminal_variant == "success" else None,
        terminal_failure=(
            TerminalFailure(code="failed", message="evaluation failed")
            if terminal_variant == "failure"
            else None
        ),
        provenance_ordinal=1,
    )
    payload = result.model_dump(mode="json")
    if hostile_ordinal == "missing":
        payload.pop("provenance_ordinal")
    else:
        payload["provenance_ordinal"] = hostile_ordinal

    with pytest.raises(
        ValidationError,
        match="non-refused Tool Result requires a positive provenance ordinal",
    ):
        ToolResult.model_validate_json(json.dumps(payload))


def test_serialized_refusal_requires_absent_provenance_ordinal() -> None:
    refused = ToolResult(
        call=tool_call_reference(_call()),
        refusal=ToolRefusal(
            refusal_class=RefusalClass.CAPACITY,
            reason="capacity exhausted",
        ),
    )
    assert refused.provenance_ordinal is None
    payload = refused.model_dump(mode="json")
    payload["provenance_ordinal"] = 1

    with pytest.raises(
        ValidationError,
        match="pre-execution refusal has no provenance ordinal",
    ):
        ToolResult.model_validate_json(json.dumps(payload))


def test_exact_definition_config_call_result_chain_rejects_tampering() -> None:
    result = ToolResult(
        call=tool_call_reference(_call()),
        output={"score": 0.75},
        evaluation_evidence_refs=(_evidence_ref(),),
        reward=reward_reference(_reward()),
        provenance_ordinal=1,
    )
    exact = tool_result_reference(result)
    assert exact.record_ref.schema_name == TOOL_RESULT_SCHEMA

    dumped = exact.model_dump(mode="json")
    dumped["record"]["call"]["record"]["tool_config"]["record"][
        "endpoint_key"
    ] = "different"
    with pytest.raises(ValidationError, match="exact record"):
        ToolResultRef.model_validate(dumped)


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (
            lambda: ToolDefinition(
                tool_name="tool",
                input_fields={"a", "b"},
                output_fields=("out",),
            ),
            "input_fields",
        ),
        (
            lambda: ToolDefinition(
                tool_name="tool",
                input_fields=("in",),
                output_fields=frozenset({"a", "b"}),
            ),
            "output_fields",
        ),
        (
            lambda: ToolDefinition(
                tool_name="tool",
                input_fields=("in",),
                output_fields=("out",),
                refusal_classes={RefusalClass.CAPACITY},
            ),
            "refusal_classes",
        ),
        (
            lambda: ToolConfig(
                definition=tool_definition_reference(_definition()),
                endpoint_key="eval",
                eval_config=_eval_config(),
                reward_policy_hash="f" * 64,
                capacity=ToolCapacity(
                    max_accepted_calls=2,
                    scope=ToolCapacityScope.RUN,
                ),
                operational_policy_refs={
                    typed_ref_for_record("policy", {"name": "one"})
                },
                store_namespace_key="tools",
            ),
            "operational_policy_refs",
        ),
        (
            lambda: RewardPolicy(
                policy_name="policy",
                terms={RewardTerm(name="score", weight=1.0)},
            ),
            "terms",
        ),
        (
            lambda: Reward(
                reward_name="reward",
                value=1.0,
                reward_policy=_reward_policy(),
                evidence_role=EvaluationRole.INTERNAL,
                input_citations={
                    RewardInputCitation(
                        name="score",
                        value=1.0,
                        contributed=1.0,
                    )
                },
                evidence_refs=(_evidence_ref(),),
            ),
            "input_citations",
        ),
        (
            lambda: ToolResult(
                call=tool_call_reference(_call()),
                output={"score": 0.75},
                evaluation_evidence_refs={
                    typed_ref_for_record("evidence", {"value": 1})
                },
                provenance_ordinal=1,
            ),
            "evaluation_evidence_refs",
        ),
    ],
)
def test_ordered_contract_fields_reject_unordered_containers(
    factory: Callable[[], object],
    field: str,
) -> None:
    with pytest.raises(ValidationError, match=field):
        factory()


def test_unordered_input_is_rejected_for_every_python_hash_seed() -> None:
    script = """
from pydantic import ValidationError
from whetstone.optimization.tools import ToolDefinition
try:
    ToolDefinition(
        tool_name="tool",
        input_fields={"alpha", "beta", "gamma"},
        output_fields=("out",),
    )
except ValidationError:
    raise SystemExit(0)
raise SystemExit("unordered set was accepted")
"""
    for seed in ("1", "2", "17", "101"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr


def test_relevant_ordered_labels_and_refs_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="term names and order"):
        Reward(
            reward_name="reward",
            value=1.0,
            reward_policy=_reward_policy(),
            evidence_role=EvaluationRole.INTERNAL,
            input_citations=(
                RewardInputCitation(
                    name="score",
                    value=1.0,
                    contributed=1.0,
                ),
                RewardInputCitation(
                    name="score",
                    value=1.0,
                    contributed=1.0,
                ),
            ),
            evidence_refs=(_evidence_ref(),),
        )
    evidence = typed_ref_for_record("evidence", {"value": 1})
    with pytest.raises(ValidationError, match="evidence_refs must be unique"):
        ToolResult(
            call=tool_call_reference(_call()),
            output={"score": 0.75},
            evaluation_evidence_refs=(evidence, evidence),
            provenance_ordinal=1,
        )


def test_exact_ref_models_reject_wrong_schema_or_content() -> None:
    definition = _definition()
    with pytest.raises(ValidationError, match="exact record"):
        ToolDefinitionRef(
            record=definition,
            record_ref=typed_ref_for_record(
                "wrong", definition.record_content()
            ),
            identity_hash=definition.identity_hash(),
        )

    config = _config()
    with pytest.raises(ValidationError, match="exact record"):
        ToolConfigRef(
            record=config,
            record_ref=typed_ref_for_record(
                TOOL_CONFIG_SCHEMA, {"different": "config"}
            ),
            identity_hash=config.identity_hash(),
        )

    call = _call()
    with pytest.raises(ValidationError, match="exact call record"):
        ToolCallRef(
            record=call,
            record_ref=typed_ref_for_record(
                TOOL_CALL_SCHEMA, {"different": "call"}
            ),
        )

    result = ToolResult(
        call=tool_call_reference(call),
        output={"score": 1.0},
        provenance_ordinal=1,
    )
    with pytest.raises(ValidationError, match="exact result record"):
        ToolResultRef(
            record=result,
            record_ref=typed_ref_for_record(
                TOOL_RESULT_SCHEMA, {"different": "result"}
            ),
        )


def test_tool_definition_has_only_enforceable_provenance_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        ToolDefinition.model_validate(
            {
                "tool_name": "tool",
                "input_fields": ("in",),
                "output_fields": ("out",),
                "required_provenance_fields": ("source",),
            }
        )
