"""Typer-backed fake executable for Codex subprocess-boundary tests."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer

from whetstone.optimization.mcp_bridge import (
    InProcessMcpProcess,
    JsonRpcClient,
)
from whetstone.optimization.mcp_server import build_server_from_env
from whetstone.optimization.schema import OptimizationStepRequest

app = typer.Typer(
    add_completion=False,
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)


def _option(args: list[str], name: str) -> str:
    try:
        return args[args.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"fake Codex missing {name}") from exc


def _mcp_environment(args: list[str]) -> dict[str, str]:
    prefix = "mcp_servers.whetstone.env."
    environment: dict[str, str] = {}
    for index, item in enumerate(args):
        if item != "-c" or index + 1 >= len(args):
            continue
        assignment = args[index + 1]
        if not assignment.startswith(prefix):
            continue
        key, raw = assignment[len(prefix) :].split("=", 1)
        environment[key] = str(json.loads(raw))
    return environment


def _request_context(prompt: str) -> OptimizationStepRequest:
    marker = "OPTIMIZATION_STEP_REQUEST_JSON="
    return OptimizationStepRequest.model_validate_json(
        prompt.split(marker, 1)[1]
    )


@app.command(
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    }
)
def main(context: typer.Context) -> None:
    args = list(context.args)
    schema_path = Path(_option(args, "--output-schema"))
    artifact_path = Path(_option(args, "--output-last-message"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    prompt = args[-1]
    request = _request_context(prompt)
    outside_path = request.hyperparameters.get("adversarial_outside_path")
    outside_read = "not_requested"
    if isinstance(outside_path, str):
        try:
            Path(outside_path).read_text(encoding="utf-8")
        except OSError:
            outside_read = "denied"
        else:
            outside_read = "readable"
    event = {
        "type": "turn.started",
        "argv": args,
        "cwd": str(Path.cwd()),
        "env_keys": sorted(os.environ),
        "schema_run_id": schema["properties"]["run_id"]["const"],
        "schema_exists": schema_path.is_file(),
        "outside_read": outside_read,
    }
    typer.echo(json.dumps(event, sort_keys=True))
    typer.echo(
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "text": json.dumps({"proposals": []}),
                },
            },
            sort_keys=True,
        )
    )
    mode = Path(sys.argv[0]).name.rsplit("-", 1)[-1]
    if mode == "missing":
        return
    if mode == "malformed":
        artifact_path.write_text("{malformed", encoding="utf-8")
        return
    proposals: list[dict[str, object]] = []
    conversation_evidence: dict[str, object] = {"agent": "final"}
    output_count = request.output_contract.returned_proposal_count
    if output_count:
        base = request.candidates[0]
        template = base.payload["user_prompt_template"]
        assert isinstance(template, str)
        proposed_template = f"{template}\nAnswer carefully."
        client = JsonRpcClient(
            InProcessMcpProcess(
                build_server_from_env(_mcp_environment(args))
            ).exchange
        )
        client.initialize()
        tools = client.list_tools()
        assert [tool["name"] for tool in tools] == ["evaluate_candidate"]
        result = client.evaluate(
            call_id="subprocess-proposal-evaluation",
            base_ref=base.base_ref,
            model_route=base.base_ref,
            template=proposed_template,
        )
        proposals.append(
            {
                "candidate_id": "subprocess-proposal",
                "base_ref": base.base_ref,
                "payload": {"user_prompt_template": proposed_template},
            }
        )
        conversation_evidence["mcp_result"] = result
    artifact_path.write_text(
        json.dumps(
            {
                "run_id": schema["properties"]["run_id"]["const"],
                "proposals": proposals,
                "conversation_evidence": conversation_evidence,
                "control_cost": {"agent_tokens": 7},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    app()
