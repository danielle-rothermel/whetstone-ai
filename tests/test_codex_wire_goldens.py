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
    CODEX_ARTIFACT_RUN_MISMATCH_CODE,
    CODEX_LEASE_TOKEN_MISMATCH_CODE,
    CODEX_MCP_HOST_FAILED_CODE,
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
from whetstone.optim.codex.control import CodexControl
from whetstone.optim.codex.mcp_environment import McpEnvironmentKey
from whetstone.optim.codex.containment import CODEX_DENIED_FEATURES
from whetstone.optim.codex.runner import build_codex_command
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
from whetstone.optim.tools.execution import TOOL_EVALUATION_REJECTED_CODE


def test_the_mcp_environment_keys_are_pinned() -> None:
    assert {member.name: member.value for member in McpEnvironmentKey} == {
        "SQLITE_PATH": "WS_MCP_SQLITE_PATH",
        "TOOL_CONFIG": "WS_MCP_TOOL_CONFIG",
        "CAPACITY_BINDING": "WS_MCP_CAPACITY_BINDING",
        "RUNTIME_CONFIG": "WS_MCP_RUNTIME_CONFIG",
        "RUNTIME_CONFIG_CLASS": "WS_MCP_RUNTIME_CONFIG_CLASS",
        "REWARD_POLICY": "WS_MCP_REWARD_POLICY",
        "RUN_LEASE_TOKEN": "WS_MCP_RUN_LEASE_TOKEN",
        "RUN_LEASE_BINDING": "WS_MCP_RUN_LEASE_BINDING",
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
    assert (
        CODEX_ARTIFACT_RUN_MISMATCH_CODE == "codex_artifact_run_mismatch"
    )
    assert CODEX_MCP_HOST_FAILED_CODE == "codex_mcp_host_failed"
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
    assert TOOL_EVALUATION_REJECTED_CODE == "tool_evaluation_rejected"


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


def test_the_reasoning_effort_reaches_the_cli_as_a_config_override() -> None:
    """Every identity-bearing control field must reach the invocation.

    The Codex CLI has no reasoning-effort flag; it is a config key, and
    the run passes ``--strict-config``, so a misspelling fails the launch
    rather than being silently dropped. The literal is pinned here
    because nothing derives it and it crosses into a foreign binary.
    """
    argv = build_codex_command(
        prompt="go",
        codex_binary="/usr/bin/codex",
        model="toy-model",
        reasoning_effort="high",
        mcp_endpoint=None,
        output_schema_path="/tmp/schema.json",
        output_artifact_path="/tmp/last.json",
        working_directory="/tmp/work",
    )

    assert "--strict-config" in argv
    override = 'model_reasoning_effort="high"'
    assert override in argv
    assert argv[argv.index(override) - 1] == "-c"


def test_an_empty_reasoning_effort_adds_no_override() -> None:
    argv = build_codex_command(
        prompt="go",
        codex_binary="/usr/bin/codex",
        model="toy-model",
        reasoning_effort="",
        mcp_endpoint=None,
        output_schema_path="/tmp/schema.json",
        output_artifact_path="/tmp/last.json",
        working_directory="/tmp/work",
    )

    assert not any("model_reasoning_effort" in entry for entry in argv)


def test_web_search_is_disabled_by_config_key_not_by_feature_flags() -> None:
    """Codex 0.148 enables web search by default; the flags no longer gate it.

    ``web_search_cached`` and ``web_search_request`` were in the deny list
    as ``--disable`` flags. Against the real 0.148 CLI they are
    *deprecated*: they disable nothing, and each one makes the CLI emit a
    deprecation ``error`` item into the JSONL transcript the adapter
    parses. Web search stayed on, so a contained agent could still reach
    the open web.

    The top-level ``web_search`` config key is what actually turns it off,
    and ``--strict-config`` makes a misspelling fatal rather than silent.
    Both halves are pinned: the key is present, and the deprecated flags
    are gone.
    """
    argv = build_codex_command(
        prompt="go",
        codex_binary="/usr/bin/codex",
        model="toy-model",
        reasoning_effort="",
        mcp_endpoint=None,
        output_schema_path="/tmp/schema.json",
        output_artifact_path="/tmp/last.json",
        working_directory="/tmp/work",
    )

    assert 'web_search="disabled"' in argv
    assert argv[argv.index('web_search="disabled"') - 1] == "-c"
    # The deprecated flags must not come back: passing them is what
    # produced the transcript noise and the false sense of containment.
    assert "web_search_cached" not in argv
    assert "web_search_request" not in argv
    assert "web_search_cached" not in CODEX_DENIED_FEATURES
    assert "web_search_request" not in CODEX_DENIED_FEATURES


def test_the_control_carries_no_field_the_cli_cannot_honor() -> None:
    """A turn cap and a sampling seed are not Codex CLI concepts.

    ``codex exec`` exposes neither, and ``--strict-config`` rejects
    ``max_turns`` and ``seed`` as unknown configuration fields. Carrying
    them would mean two runs with different identities and different
    recorded hyperparameters executing byte-identical invocations.
    """
    fields = set(CodexControl.model_fields)

    assert "max_turns" not in fields
    assert "seed" not in fields
    assert "reasoning_effort" in fields


def test_the_agent_is_given_an_endpoint_url_and_never_the_store() -> None:
    """The agent connects to a server; it never spawns one.

    Spawning the server as a child of the sandboxed agent gave the agent
    the server's profile, and the server must write the whetstone store
    -- the durable ledger, and the admission-capacity rows that cap paid
    evaluations. So the CLI is configured with a URL, and the store path
    and the server's own configuration never enter its argv at all.
    """
    from whetstone.optim.codex.mcp_host import CodexMcpEndpoint
    from whetstone.optim.codex.runner import CODEX_MCP_TOKEN_ENV

    argv = build_codex_command(
        prompt="go",
        codex_binary="/usr/bin/codex",
        model="toy-model",
        reasoning_effort="low",
        mcp_endpoint=CodexMcpEndpoint(
            url="http://127.0.0.1:4242/mcp", auth_token="run-token"
        ),
        output_schema_path="/tmp/schema.json",
        output_artifact_path="/tmp/last.json",
        working_directory="/tmp/work",
    )
    joined = " ".join(argv)

    assert 'mcp_servers.whetstone.url="http://127.0.0.1:4242/mcp"' in argv
    assert (
        f'mcp_servers.whetstone.bearer_token_env_var="{CODEX_MCP_TOKEN_ENV}"'
        in argv
    )
    # No command, no server module, and no environment block: the agent
    # is given nothing it could use to start a store-writing process.
    assert "mcp_servers.whetstone.command" not in joined
    assert "mcp_servers.whetstone.env." not in joined
    # The token travels in the environment, not in a world-readable argv.
    assert "run-token" not in joined


def test_the_bearer_token_variable_is_pinned() -> None:
    from whetstone.optim.codex.runner import CODEX_MCP_TOKEN_ENV

    assert CODEX_MCP_TOKEN_ENV == "WS_MCP_BEARER_TOKEN"


def test_the_mcp_endpoint_path_and_auth_scheme_are_pinned() -> None:
    from whetstone.optim.codex.mcp_host import (
        CODEX_MCP_AUTH_HEADER,
        CODEX_MCP_AUTH_SCHEME,
        CODEX_MCP_HTTP_PATH,
    )

    assert CODEX_MCP_HTTP_PATH == "/mcp"
    assert CODEX_MCP_AUTH_HEADER == "authorization"
    assert CODEX_MCP_AUTH_SCHEME == "Bearer"
