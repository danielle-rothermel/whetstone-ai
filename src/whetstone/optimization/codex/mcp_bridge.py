from __future__ import annotations

import json
from typing import Any

from mcp import types as mcp_types
from mcp.server.mcpserver import Context, MCPServer

from whetstone.core.identity import (
    ImmutableJsonObject,
    NonEmptyId,
    TypedRef,
)
from whetstone.optimization.tools.contracts import (
    RuntimeToolHandle,
    ToolCall,
    ToolResult,
)

_BASE_INPUT_FIELDS = frozenset({"base_ref", "model_route", "template"})
_TASK_SUBSET_INPUT_FIELDS = _BASE_INPUT_FIELDS | {"task_ids"}


def tool_result_to_mcp_result(
    result: ToolResult,
) -> mcp_types.CallToolResult:
    if result.refusal is not None:
        payload = {
            "refused": True,
            "call_id": str(result.call_id),
            "refusal_class": result.refusal.refusal_class.value,
            "reason": str(result.refusal.reason),
        }
        is_error = True
    elif result.terminal_failure is not None:
        payload = {
            "refused": False,
            "call_id": str(result.call_id),
            "terminal_failure": result.terminal_failure.model_dump(
                mode="json"
            ),
        }
        is_error = True
    else:
        assert result.output is not None
        payload = {
            "refused": False,
            "call_id": str(result.call_id),
            "output": result.output.to_json(),
            "reward": (
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
        if input_fields not in (
            _BASE_INPUT_FIELDS,
            _TASK_SUBSET_INPUT_FIELDS,
        ):
            raise ValueError(
                "MCP evaluation Tool Definition must declare exactly "
                "base_ref, model_route, template, and optionally task_ids"
            )

        super().__init__(name="whetstone", version="1")
        self.tool_config = handle.config
        self._handle = handle
        self._input_names = frozenset({"call_id", *input_fields})

        if input_fields == _TASK_SUBSET_INPUT_FIELDS:

            def evaluate_candidate(
                call_id: NonEmptyId,
                base_ref: TypedRef,
                model_route: NonEmptyId,
                template: str,
                task_ids: list[str],
            ) -> mcp_types.CallToolResult:
                return self._call(
                    call_id=call_id,
                    base_ref=base_ref,
                    model_route=model_route,
                    template=template,
                    task_ids=task_ids,
                )

        else:

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
        task_ids: list[str] | None = None,
    ) -> mcp_types.CallToolResult:
        raw_args: dict[str, object] = {
            "base_ref": base_ref.model_dump(mode="json"),
            "model_route": str(model_route),
            "template": template,
        }
        if task_ids is not None:
            raw_args["task_ids"] = task_ids
        call = ToolCall(
            call_id=call_id,
            tool_config=self._handle.tool_config_ref,
            capacity_binding=self._handle.binding,
            args=ImmutableJsonObject(raw_args),
        )
        return tool_result_to_mcp_result(self._handle(call))


__all__ = ["EvaluateCandidateServer", "tool_result_to_mcp_result"]
