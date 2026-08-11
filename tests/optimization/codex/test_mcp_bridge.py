from __future__ import annotations

import anyio
import pytest
from mcp.client import Client
from mcp.types import CallToolResult, Tool

from tests.optimization.codex.support import (
    MODEL_ROUTE,
    CodexStack,
    binding,
    request,
    stack,
)
from whetstone.core.identity import TerminalFailure
from whetstone.envs.code_comp.constants import MUTATION_FIELD
from whetstone.envs.factory import EnvExperiment
from whetstone.optimization.codex.mcp_bridge import EvaluateCandidateServer
from whetstone.optimization.tools.contracts import (
    RefusalClass,
    RuntimeToolHandle,
    ToolCall,
    ToolResult,
    tool_call_reference,
    tool_definition_reference,
)


def _arguments(
    codex: CodexStack,
    *,
    call_id: str,
    model_route: str = MODEL_ROUTE,
    task_ids: list[str] | None = None,
) -> dict[str, object]:
    template = codex.base.payload[MUTATION_FIELD]
    assert isinstance(template, str)
    arguments: dict[str, object] = {
        "call_id": call_id,
        "base_ref": codex.base.base_ref.model_dump(mode="json"),
        "model_route": model_route,
        "template": template,
    }
    if task_ids is not None:
        arguments["task_ids"] = task_ids
    return arguments


async def _list_and_call(
    server: EvaluateCandidateServer,
    tool_name: str,
    arguments: dict[str, object],
) -> tuple[tuple[Tool, ...], CallToolResult]:
    async with Client(server) as client:
        listed = await client.list_tools()
        result = await client.call_tool(tool_name, arguments)
    return tuple(listed.tools), result


def _call(
    server: EvaluateCandidateServer,
    *,
    tool_name: str,
    arguments: dict[str, object],
) -> tuple[tuple[Tool, ...], CallToolResult]:
    return anyio.run(
        _list_and_call,
        server,
        tool_name,
        arguments,
    )


def _payload(result: CallToolResult) -> dict[str, object]:
    assert isinstance(result.structured_content, dict)
    return result.structured_content


def test_sdk_client_lists_and_calls_exactly_the_configured_tool(
    tmp_path, codex_experiment: EnvExperiment
) -> None:
    codex = stack(
        tmp_path,
        codex_experiment,
        namespace="codex-renamed-tool",
        tool_name="score_candidate_draft",
    )
    step_request = request(
        codex.base,
        codex.config,
        run_id="codex-run-renamed-tool",
    )
    handle = codex.executor.runtime_handle(
        codex.config,
        codex.tool_store,
        binding(step_request),
    )

    listed, result = _call(
        EvaluateCandidateServer(handle=handle),
        tool_name="score_candidate_draft",
        arguments=_arguments(codex, call_id="renamed-tool-call"),
    )

    assert tuple(tool.name for tool in listed) == ("score_candidate_draft",)
    assert listed[0].input_schema["additionalProperties"] is False
    assert result.is_error is False
    assert _payload(result)["output"] is not None
    assert codex.tool_store.accepted_count(codex.config, handle.binding) == 1


def test_sdk_rejects_malformed_typed_arguments_before_the_handle(
    tmp_path, codex_experiment: EnvExperiment
) -> None:
    codex = stack(
        tmp_path,
        codex_experiment,
        namespace="codex-malformed-mcp-arguments",
    )
    step_request = request(codex.base, codex.config)
    handle = codex.executor.runtime_handle(
        codex.config,
        codex.tool_store,
        binding(step_request),
    )
    arguments = _arguments(codex, call_id="malformed-mcp-call")
    arguments["base_ref"] = "not-a-typed-reference"

    _, result = _call(
        EvaluateCandidateServer(handle=handle),
        tool_name=str(codex.config.tool_name),
        arguments=arguments,
    )

    assert result.is_error is True
    assert result.structured_content is None
    assert codex.tool_store.accepted_count(codex.config, handle.binding) == 0


def test_sdk_forwards_declared_task_ids_to_the_evaluator(
    tmp_path, codex_experiment: EnvExperiment
) -> None:
    codex = stack(
        tmp_path,
        codex_experiment,
        namespace="codex-task-subset",
        supports_task_ids=True,
    )
    step_request = request(codex.base, codex.config)
    handle = codex.executor.runtime_handle(
        codex.config,
        codex.tool_store,
        binding(step_request),
    )
    task_id = codex_experiment.eval_configs.internal.task_set.task_hashes[0]

    _, result = _call(
        EvaluateCandidateServer(handle=handle),
        tool_name=str(codex.config.tool_name),
        arguments=_arguments(
            codex,
            call_id="task-subset-call",
            task_ids=[task_id],
        ),
    )

    assert result.is_error is False
    assert _payload(result)["output"] is not None
    assert codex.tool_store.accepted_count(codex.config, handle.binding) == 1


def test_sdk_rejects_task_ids_the_tool_does_not_declare(
    tmp_path, codex_experiment: EnvExperiment
) -> None:
    codex = stack(
        tmp_path,
        codex_experiment,
        namespace="codex-undeclared-task-subset",
    )
    step_request = request(codex.base, codex.config)
    handle = codex.executor.runtime_handle(
        codex.config,
        codex.tool_store,
        binding(step_request),
    )

    _, result = _call(
        EvaluateCandidateServer(handle=handle),
        tool_name=str(codex.config.tool_name),
        arguments=_arguments(
            codex,
            call_id="undeclared-task-subset-call",
            task_ids=["not-declared-by-this-tool"],
        ),
    )

    assert result.is_error is True
    assert codex.tool_store.accepted_count(codex.config, handle.binding) == 0


def test_server_rejects_an_incompatible_tool_definition(
    tmp_path, codex_experiment: EnvExperiment
) -> None:
    codex = stack(
        tmp_path,
        codex_experiment,
        namespace="codex-incompatible-definition",
    )
    definition = codex.config.definition.record.model_copy(
        update={
            "input_fields": (
                *codex.config.definition.record.input_fields,
                "unsupported_input",
            )
        }
    )
    config = codex.config.model_copy(
        update={"definition": tool_definition_reference(definition)}
    )

    def unexpected_call(_call: ToolCall) -> ToolResult:
        raise AssertionError("incompatible definitions must fail at ingress")

    handle = RuntimeToolHandle(
        config,
        binding(request(codex.base, config)),
        unexpected_call,
    )

    with pytest.raises(ValueError):
        EvaluateCandidateServer(handle=handle)


def test_refusal_is_model_visible_without_consuming_capacity(
    tmp_path, codex_experiment: EnvExperiment
) -> None:
    codex = stack(
        tmp_path,
        codex_experiment,
        namespace="codex-mcp-refusal",
    )
    step_request = request(codex.base, codex.config)
    handle = codex.executor.runtime_handle(
        codex.config,
        codex.tool_store,
        binding(step_request),
    )

    _, result = _call(
        EvaluateCandidateServer(handle=handle),
        tool_name=str(codex.config.tool_name),
        arguments=_arguments(
            codex,
            call_id="refused-mcp-call",
            model_route="openai/foreign",
        ),
    )

    assert result.is_error is True
    assert _payload(result)["refusal_class"] == RefusalClass.VALIDATION.value
    assert codex.tool_store.accepted_count(codex.config, handle.binding) == 0


def test_terminal_failure_is_model_visible_as_an_mcp_error(
    tmp_path, codex_experiment: EnvExperiment
) -> None:
    codex = stack(
        tmp_path,
        codex_experiment,
        namespace="codex-mcp-terminal-failure",
    )
    step_request = request(codex.base, codex.config)
    capacity_binding = binding(step_request)
    failure = TerminalFailure(
        code="evaluator_exhausted",
        message="all evaluator attempts failed",
        details={"attempts": 3},
    )

    def fail(call: ToolCall) -> ToolResult:
        return ToolResult(
            call=tool_call_reference(call),
            terminal_failure=failure,
            provenance_ordinal=1,
        )

    handle = RuntimeToolHandle(codex.config, capacity_binding, fail)

    _, result = _call(
        EvaluateCandidateServer(handle=handle),
        tool_name=str(codex.config.tool_name),
        arguments=_arguments(codex, call_id="terminal-mcp-call"),
    )

    assert result.is_error is True
    assert _payload(result)["terminal_failure"] == failure.model_dump(
        mode="json"
    )
