from __future__ import annotations

import json
from enum import UNIQUE, StrEnum, verify
from typing import Any

from mcp import types as mcp_types
from mcp.server.mcpserver import Context, MCPServer

from whetstone.core.identity import (
    ImmutableJsonObject,
    NonEmptyId,
    TypedRef,
)
from whetstone.optim.tools.contracts import (
    RuntimeToolHandle,
    ToolCall,
    ToolResult,
)

#: The one Tool the Codex optimizer is granted (D12).
CODEX_EVAL_TOOL_NAME = "evaluate_candidate"

#: The canonical ordered ``ToolDefinition.input_fields``. The wire schema
#: the MCP server advertises is these fields plus ``call_id``.
#:
#: The tool evaluates the whole internal split (D12). There is no
#: subset-narrowing variant: ``EvalEngine.for_task_ids`` mints a different
#: Eval Config identity, which ``EvaluatingToolExecutor`` rejects as
#: ``tool_eval_config_mismatch``, so a narrowed call could never complete.
CODEX_EVAL_INPUT_FIELDS: tuple[str, ...] = (
    "base_ref",
    "model_route",
    "template",
)
#: The canonical ordered ``ToolDefinition.output_fields``, chosen from what
#: ``EngineToolEvaluator`` can supply.
CODEX_EVAL_OUTPUT_FIELDS: tuple[str, ...] = (
    "evaluation_evidence_ref",
    "output_artifact_ref",
    "per_task_values",
    "per_task_counts",
    "row_accounting",
)

_BASE_INPUT_FIELDS = frozenset(CODEX_EVAL_INPUT_FIELDS)


@verify(UNIQUE)
class McpResultKey(StrEnum):
    """Keys of the MCP tool-result payload the Codex CLI reads.

    The payload crosses a process boundary into a foreign agent, so these
    spellings are a persisted wire format with golden literal tests.
    """

    REFUSED = "refused"
    CALL_ID = "call_id"
    REFUSAL_CLASS = "refusal_class"
    REASON = "reason"
    TERMINAL_FAILURE = "terminal_failure"
    OUTPUT = "output"
    REWARD = "reward"


def tool_result_to_mcp_result(
    result: ToolResult,
) -> mcp_types.CallToolResult:
    payload: dict[str, Any]
    if result.refusal is not None:
        payload = {
            McpResultKey.REFUSED.value: True,
            McpResultKey.CALL_ID.value: str(result.call_id),
            McpResultKey.REFUSAL_CLASS.value: (
                result.refusal.refusal_class.value
            ),
            McpResultKey.REASON.value: str(result.refusal.reason),
        }
        is_error = True
    elif result.terminal_failure is not None:
        payload = {
            McpResultKey.REFUSED.value: False,
            McpResultKey.CALL_ID.value: str(result.call_id),
            McpResultKey.TERMINAL_FAILURE.value: (
                result.terminal_failure.model_dump(mode="json")
            ),
        }
        is_error = True
    else:
        assert result.output is not None
        payload = {
            McpResultKey.REFUSED.value: False,
            McpResultKey.CALL_ID.value: str(result.call_id),
            McpResultKey.OUTPUT.value: result.output.to_json(),
            McpResultKey.REWARD.value: (
                None
                if result.reward is None
                else result.reward.model_dump(mode="json")
            ),
        }
        is_error = False
    return mcp_types.CallToolResult(
        content=[
            mcp_types.TextContent(
                type="text",
                text=json.dumps(payload, sort_keys=True),
            )
        ],
        structured_content=payload,
        is_error=is_error,
    )


class EvaluateCandidateServer(MCPServer[None]):
    def __init__(self, *, handle: RuntimeToolHandle) -> None:
        definition = handle.config.definition.record
        input_fields = frozenset(definition.input_fields)
        if input_fields != _BASE_INPUT_FIELDS:
            raise ValueError(
                "MCP evaluation Tool Definition must declare exactly "
                "base_ref, model_route, and template"
            )

        super().__init__(name="whetstone", version="1")
        self.tool_config = handle.config
        self._handle = handle
        self._input_names = frozenset({"call_id", *input_fields})
        #: The store path this server's persistent session was opened on,
        #: set by the factory that opened it. Whoever hosts the server is
        #: the only party that knows when the session may be released, so
        #: the path travels with the server rather than being rediscovered.
        self.sqlite_path: str | None = None

        def evaluate_candidate(
            call_id: NonEmptyId,
            base_ref: TypedRef,
            model_route: NonEmptyId,
            template: str,
        ) -> mcp_types.CallToolResult:
            return self._call(
                call_id=call_id,
                base_ref=base_ref,
                model_route=model_route,
                template=template,
            )

        self.add_tool(
            evaluate_candidate,
            name=str(definition.tool_name),
            description=(
                "Evaluate a candidate using Whetstone's canonical engine."
            ),
            structured_output=False,
        )

    async def list_tools(self) -> list[mcp_types.Tool]:
        tools = await super().list_tools()
        return [
            tool.model_copy(
                update={
                    "input_schema": {
                        **tool.input_schema,
                        "additionalProperties": False,
                    }
                }
            )
            for tool in tools
        ]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Context[None, Any] | None = None,
    ) -> mcp_types.CallToolResult | mcp_types.InputRequiredResult:
        if name == self.tool_config.tool_name and set(arguments) != (
            self._input_names
        ):
            raise ValueError(
                "MCP evaluation arguments must exactly match the advertised "
                "tool input schema"
            )
        return await super().call_tool(name, arguments, context)

    @property
    def handle(self) -> RuntimeToolHandle:
        return self._handle

    def _call(
        self,
        *,
        call_id: NonEmptyId,
        base_ref: TypedRef,
        model_route: NonEmptyId,
        template: str,
    ) -> mcp_types.CallToolResult:
        raw_args: dict[str, object] = {
            "base_ref": base_ref.model_dump(mode="json"),
            "model_route": str(model_route),
            "template": template,
        }
        call = ToolCall(
            call_id=call_id,
            tool_config=self._handle.tool_config_ref,
            capacity_binding=self._handle.binding,
            args=ImmutableJsonObject(raw_args),
        )
        return tool_result_to_mcp_result(self._handle(call))


__all__ = [
    "CODEX_EVAL_INPUT_FIELDS",
    "CODEX_EVAL_OUTPUT_FIELDS",
    "CODEX_EVAL_TOOL_NAME",
    "EvaluateCandidateServer",
    "McpResultKey",
    "tool_result_to_mcp_result",
]
