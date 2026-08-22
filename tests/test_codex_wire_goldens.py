"""Literal spellings of the Codex wire and environment formats.

These strings cross a process boundary into a foreign agent and into the
MCP server's own environment. Nothing derives them from Python field
names, so only a golden literal test catches silent drift.
"""

from __future__ import annotations

import json

from whetstone.core.identity import ImmutableJsonObject, TerminalFailure
from whetstone.optim.codex.adapter import (
    CODEX_ADAPTER_KEY,
    CODEX_LEASE_TOKEN_MISMATCH_CODE,
    CODEX_OUTPUT_ARTIFACT_SCHEMA,
    CODEX_SELECTION_CONTRACT_CODE,
    CODEX_SELECTION_UNEVALUATED_CODE,
    CODEX_SELECTION_UNSCORED_CODE,
    CODEX_UNREPORTED_EVALUATION_CODE,
    CODEX_WALL_BUDGET_EXCEEDED_CODE,
    CodexOutputArtifact,
)
from whetstone.optim.codex.mcp_bridge import (
    CODEX_EVAL_INPUT_FIELDS,
    CODEX_EVAL_OUTPUT_FIELDS,
    CODEX_EVAL_TOOL_NAME,
    McpResultKey,
)
from whetstone.optim.codex.mcp_environment import McpEnvironmentKey
from whetstone.optim.codex.step_contract import (
    CODEX_TOOL_CALLS_BUDGET_LABEL,
)
from whetstone.optim.tools.contracts import (
    RefusalClass,
    ToolCall,
    ToolCallRef,
    ToolRefusal,
    ToolResult,
    tool_call_reference,
)
from whetstone.optim.tools.evaluator import (
    TOOL_EVAL_FAILURE_EVIDENCE_CODE,
    TOOL_EVAL_UNEXPECTED_RESULT_CODE,
)


def test_the_mcp_environment_keys_are_pinned() -> None:
    assert {member.name: member.value for member in McpEnvironmentKey} == {
        "SQLITE_PATH": "WS_MCP_SQLITE_PATH",
        "TOOL_CONFIG": "WS_MCP_TOOL_CONFIG",
        "CAPACITY_BINDING": "WS_MCP_CAPACITY_BINDING",
        "RUNTIME_CONFIG": "WS_MCP_RUNTIME_CONFIG",
        "RUNTIME_CONFIG_CLASS": "WS_MCP_RUNTIME_CONFIG_CLASS",
        "REWARD_POLICY": "WS_MCP_REWARD_POLICY",
        "RUN_LEASE_TOKEN": "WS_MCP_RUN_LEASE_TOKEN",
    }


def test_the_mcp_result_payload_keys_are_pinned() -> None:
    assert {member.name: member.value for member in McpResultKey} == {
        "REFUSED": "refused",
        "CALL_ID": "call_id",
        "REFUSAL_CLASS": "refusal_class",
        "REASON": "reason",
        "TERMINAL_FAILURE": "terminal_failure",
        "OUTPUT": "output",
        "REWARD": "reward",
    }


def test_the_tool_name_and_field_tuples_are_pinned() -> None:
    assert CODEX_EVAL_TOOL_NAME == "evaluate_candidate"
    assert CODEX_EVAL_INPUT_FIELDS == (
        "base_ref",
        "model_route",
        "template",
    )
    assert CODEX_EVAL_OUTPUT_FIELDS == (
        "evaluation_evidence_ref",
        "output_artifact_ref",
        "per_task_values",
        "per_task_counts",
        "row_accounting",
    )


def test_the_persisted_schema_and_code_literals_are_pinned() -> None:
    assert CODEX_ADAPTER_KEY == "codex"
    assert CODEX_OUTPUT_ARTIFACT_SCHEMA == "whetstone.codex_output_artifact"
    assert CODEX_SELECTION_UNEVALUATED_CODE == "codex_selection_unevaluated"
    assert CODEX_LEASE_TOKEN_MISMATCH_CODE == "codex_lease_token_mismatch"
    assert CODEX_SELECTION_CONTRACT_CODE == "codex_selection_contract"
    assert CODEX_SELECTION_UNSCORED_CODE == "codex_selection_unscored"
    assert (
        CODEX_UNREPORTED_EVALUATION_CODE
        == "codex_unreported_evaluation"
    )
    assert (
        CODEX_WALL_BUDGET_EXCEEDED_CODE == "codex_wall_budget_exceeded"
    )
    assert TOOL_EVAL_FAILURE_EVIDENCE_CODE == "tool_eval_failure_evidence"
    assert TOOL_EVAL_UNEXPECTED_RESULT_CODE == "tool_eval_unexpected_result"


def test_the_budget_label_the_ledger_reads_is_pinned() -> None:
    # _IssuedToolCallLedger keys its hard limit and its budget_delta
    # injection on this exact literal; any other spelling silently
    # disables both.
    assert CODEX_TOOL_CALLS_BUDGET_LABEL == "tool_calls"


def test_the_output_artifact_field_set_is_pinned() -> None:
    assert set(CodexOutputArtifact.model_fields) == {
        "run_id",
        "evaluated_call_ids",
        "selected_call_id",
        "lease_token_hash",
        "conversation_evidence",
        "control_cost",
    }
    # The artifact carries no candidate body: a template that was never
    # evaluated through the tool cannot be returned.
    assert "proposals" not in CodexOutputArtifact.model_fields


def _refusal_result(call_ref: ToolCallRef) -> ToolResult:
    return ToolResult(
        call=call_ref,
        refusal=ToolRefusal(
            refusal_class=RefusalClass.CAPACITY,
            reason="Tool Capacity exhausted",
        ),
    )


def test_a_refusal_payload_uses_the_pinned_keys(codex_tool_config) -> None:
    from whetstone.optim.codex.mcp_bridge import tool_result_to_mcp_result
    from whetstone.optim.tools.contracts import (
        ToolCapacityScope,
        tool_capacity_binding,
        tool_config_reference,
    )

    config, subject_ref = codex_tool_config
    call = ToolCall(
        call_id="c1",
        tool_config=tool_config_reference(config),
        capacity_binding=tool_capacity_binding(
            ToolCapacityScope.RUN, subject_ref
        ),
        args=ImmutableJsonObject(
            {
                "base_ref": {
                    "schema_name": "whetstone.candidate",
                    "content_hash": "0" * 64,
                },
                "model_route": "route",
                "template": "hello {prompt}",
            }
        ),
    )
    result = _refusal_result(tool_call_reference(call))

    mcp_result = tool_result_to_mcp_result(result)

    assert mcp_result.is_error is True
    assert set(mcp_result.structured_content) == {
        "refused",
        "call_id",
        "refusal_class",
        "reason",
    }
    assert mcp_result.structured_content["refused"] is True
    assert mcp_result.structured_content["refusal_class"] == "capacity"
    # The text content is the same payload, canonically ordered.
    assert json.loads(mcp_result.content[0].text) == (
        mcp_result.structured_content
    )


def test_a_terminal_failure_payload_uses_the_pinned_keys(
    codex_tool_config,
) -> None:
    from whetstone.optim.codex.mcp_bridge import tool_result_to_mcp_result
    from whetstone.optim.tools.contracts import (
        ToolCapacityScope,
        tool_capacity_binding,
        tool_config_reference,
    )

    config, subject_ref = codex_tool_config
    call = ToolCall(
        call_id="c2",
        tool_config=tool_config_reference(config),
        capacity_binding=tool_capacity_binding(
            ToolCapacityScope.RUN, subject_ref
        ),
        args=ImmutableJsonObject(
            {
                "base_ref": {
                    "schema_name": "whetstone.candidate",
                    "content_hash": "0" * 64,
                },
                "model_route": "route",
                "template": "hello {prompt}",
            }
        ),
    )
    result = ToolResult(
        call=tool_call_reference(call),
        terminal_failure=TerminalFailure(
            code=TOOL_EVAL_UNEXPECTED_RESULT_CODE,
            message="Tool evaluation produced an unrecognized Eval Result",
        ),
        provenance_ordinal=1,
    )

    mcp_result = tool_result_to_mcp_result(result)

    assert mcp_result.is_error is True
    assert set(mcp_result.structured_content) == {
        "refused",
        "call_id",
        "terminal_failure",
    }
    assert mcp_result.structured_content["refused"] is False
    assert (
        mcp_result.structured_content["terminal_failure"]["code"]
        == TOOL_EVAL_UNEXPECTED_RESULT_CODE
    )
