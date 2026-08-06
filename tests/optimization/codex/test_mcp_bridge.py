from __future__ import annotations

import json
from io import StringIO

import pytest

from tests.optimization.codex.mcp_client import (
    InProcessMcpProcess,
    JsonRpcClient,
)
from tests.optimization.codex.support import (
    MODEL_ROUTE,
    binding,
    fake_runner,
    request,
    stack,
)
from whetstone.envs.factory import EnvExperiment
from whetstone.optimization.codex.adapter import CodexAdapter
from whetstone.optimization.codex.mcp_bridge import (
    EvaluateCandidateServer,
    McpError,
    serve_stdio,
)
from whetstone.optimization.contracts import StepStatus


def test_client_calls_the_handle_configured_tool_name(
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
    runner = fake_runner(codex.base, call_id="renamed-tool-call")

    output = CodexAdapter(
        runner,
        store=codex.store,
        tool_store=codex.tool_store,
    ).invoke(step_request, (handle,))

    assert output.proposed_status is StepStatus.COMPLETE
    assert runner.observed_payloads[0]["refused"] is False
    assert codex.tool_store.accepted_count(codex.config, handle.binding) == 1


def test_client_rejects_a_tool_name_the_server_does_not_serve(
    tmp_path, codex_experiment: EnvExperiment
) -> None:
    codex = stack(
        tmp_path,
        codex_experiment,
        namespace="codex-mismatched-tool",
    )
    step_request = request(
        codex.base,
        codex.config,
        run_id="codex-run-mismatched-tool",
    )
    handle = codex.executor.runtime_handle(
        codex.config,
        codex.tool_store,
        binding(step_request),
    )
    client = JsonRpcClient(
        InProcessMcpProcess(EvaluateCandidateServer(handle=handle)).exchange,
        tool_name="not_the_served_tool",
    )
    client.initialize()

    with pytest.raises(McpError):
        client.evaluate(
            call_id="mismatched-tool-call",
            base_ref=codex.base.base_ref.model_dump(mode="json"),
            model_route=MODEL_ROUTE,
            template="{question}\n{query}\nRespond True or False.",
        )


@pytest.mark.parametrize(
    ("malformed", "code"),
    [("not-json", -32700), ("[]", -32600)],
)
def test_serve_stdio_answers_malformed_lines_and_keeps_serving(
    tmp_path,
    codex_experiment: EnvExperiment,
    malformed: str,
    code: int,
) -> None:
    codex = stack(
        tmp_path,
        codex_experiment,
        namespace=f"codex-serve-stdio-{code}",
    )
    step_request = request(
        codex.base,
        codex.config,
        run_id=f"codex-run-serve-stdio-{code}",
    )
    handle = codex.executor.runtime_handle(
        codex.config,
        codex.tool_store,
        binding(step_request),
    )
    stdin = StringIO(
        f"{malformed}\n"
        + json.dumps({"jsonrpc": "2.0", "id": 7, "method": "ping"})
        + "\n"
    )
    stdout = StringIO()

    serve_stdio(
        EvaluateCandidateServer(handle=handle),
        stdin=stdin,
        stdout=stdout,
    )

    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert len(responses) == 2
    assert responses[0]["error"]["code"] == code
    assert responses[1]["result"] == {}
