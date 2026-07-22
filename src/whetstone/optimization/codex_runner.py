"""Runners for the opaque Codex Step: the real subprocess and the fake double.

The Codex adapter (:mod:`whetstone.optimization.codex`) drives a
:class:`~whetstone.optimization.codex.CodexRunner`. Two are shipped:

* :class:`FakeCodexRunner` -- the deterministic test double used by ALL
  deterministic tests. It is a scripted MCP *client* (via
  :class:`~whetstone.optimization.mcp_bridge.ScriptedMcpClient`) driving an
  in-process
  :class:`~whetstone.optimization.mcp_bridge.EvaluateCandidateServer` over the
  same JSON-RPC protocol a real MCP client (Codex) speaks. It performs the
  ``initialize`` / ``tools/list`` / ``tools/call`` handshake, makes a scripted
  sequence of ``evaluate_candidate`` calls (including replays and over-cap
  calls), then emits the final proposals -- with no real CLI and no network.

* :class:`SubprocessCodexRunner` -- launches the real ``codex exec``
  non-interactively, configured to use the whetstone MCP server subprocess
  (``python -m whetstone.optimization.mcp_server``) via a
  ``-c mcp_servers.whetstone.command=...`` override, passes the task prompt,
  and
  treats the process output as the opaque execution. The codex binary and model
  are configurable. It shares the durable Tool Call Store with the whetstone
  process through a common SqliteBackend file, so capacity and completion are
  the same durable state across the process boundary.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from typing import Any

from whetstone.optimization.codex import (
    CodexRunResult,
    CodexToolCallLog,
    OpaqueStepError,
)
from whetstone.optimization.mcp_bridge import (
    EvaluateCandidateServer,
    ScriptedMcpClient,
)
from whetstone.optimization.schema import Candidate, OptimizationStepRequest
from whetstone.optimization.tools import ToolConfig, ToolResult

__all__ = [
    "FakeCodexRunner",
    "ScriptedAgentCall",
    "SubprocessCodexRunner",
    "build_codex_command",
]


class ScriptedAgentCall:
    """One scripted ``evaluate_candidate`` call the fake agent makes."""

    __slots__ = ("call_id", "model_route", "template")

    def __init__(
        self, *, call_id: str, model_route: str, template: str
    ) -> None:
        self.call_id = call_id
        self.model_route = model_route
        self.template = template


class FakeCodexRunner:
    """A scripted MCP client standing in for the opaque Codex agent.

    Deterministic and offline. Given the in-process
    :class:`EvaluateCandidateServer`, a script of tool calls, and the final
    proposals the "agent" decides to return, it drives the real MCP protocol
    against the server and reads back the durable tool evidence. It records
    each
    parsed tool payload so a test can assert refusals never carried a
    measurement.
    """

    def __init__(
        self,
        *,
        server: EvaluateCandidateServer,
        scripted_calls: Sequence[ScriptedAgentCall],
        final_proposals: Sequence[Candidate],
        control_cost: dict[str, Any] | None = None,
    ) -> None:
        self._server = server
        self._scripted_calls = list(scripted_calls)
        self._final_proposals = tuple(final_proposals)
        self._control_cost = control_cost or {
            "agent_tokens": 0,
            "wall_clock_s": 0.0,
        }
        # Parsed tool-call payloads (evidence the agent saw), in order.
        self.observed_payloads: list[dict[str, Any]] = []

    def run(
        self,
        request: OptimizationStepRequest,
        tool_config: ToolConfig,
    ) -> CodexRunResult:
        client = ScriptedMcpClient.for_server(self._server)
        client.initialize()
        tools = client.list_tools()
        if not any(t["name"] == tool_config.tool_name for t in tools):
            raise OpaqueStepError(
                "the MCP bridge did not expose evaluate_candidate"
            )
        logs: list[CodexToolCallLog] = []
        seen: dict[str, ToolResult] = {}
        for scripted in self._scripted_calls:
            payload = client.call_evaluate(
                call_id=scripted.call_id,
                model_route=scripted.model_route,
                template=scripted.template,
            )
            self.observed_payloads.append(payload)
            result = self._server.produced_results[-1]
            # Identical-call_id replay returns the completed Result and spends
            # no new slot; keep one log per distinct call_id (agent order).
            if scripted.call_id not in seen:
                logs.append(
                    CodexToolCallLog(call_id=scripted.call_id, result=result)
                )
                seen[scripted.call_id] = result
        return CodexRunResult(
            proposals=self._final_proposals,
            tool_calls=tuple(logs),
            conversation_evidence={
                "transcript_summary": "fake-codex scripted session",
                "scripted_call_count": len(self._scripted_calls),
            },
            control_cost=self._control_cost,
        )


def build_codex_command(
    *,
    prompt: str,
    codex_binary: str,
    model: str,
    mcp_server_command: Sequence[str],
    mcp_env: dict[str, str],
    server_name: str = "whetstone",
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Assemble the ``codex exec`` argv that wires in the whetstone MCP server.

    Uses ``-c mcp_servers.<name>.command`` / ``.args`` / ``.env.<KEY>``
    overrides so ``codex exec`` launches the whetstone stdio MCP server as a
    subprocess and can call ``evaluate_candidate``. Each override value is
    TOML-encoded (a JSON string is a valid TOML basic string); the ``env`` map
    is passed one key at a time so a complex value (the serialized Tool Config)
    is a single TOML string rather than an inline table that must round-trip.
    The prompt is passed as the exec argument.
    """
    command_toml = json.dumps(mcp_server_command[0])
    args_toml = json.dumps(list(mcp_server_command[1:]))
    argv = [codex_binary, "exec"]
    # An empty model defers to the codex account's default model (the design's
    # illustrative gpt-5.6 is not a real account model); a set model is pinned.
    if model:
        argv.extend(["--model", model])
    argv.extend(
        [
            "--json",
            "--skip-git-repo-check",
            "-c",
            f"mcp_servers.{server_name}.command={command_toml}",
            "-c",
            f"mcp_servers.{server_name}.args={args_toml}",
        ]
    )
    for key, value in mcp_env.items():
        argv.extend(
            ["-c", f"mcp_servers.{server_name}.env.{key}={json.dumps(value)}"]
        )
    argv.extend([*extra_args, prompt])
    return argv


class SubprocessCodexRunner:
    """Launches the real ``codex exec`` with the whetstone MCP server.

    The codex binary and model are configurable. The MCP server is launched by
    codex as ``python -m whetstone.optimization.mcp_server`` (a fixed argv)
    with
    its configuration passed through the environment (shared SqliteBackend
    path,
    serialized Tool Config + Reward Policy, and the evaluator factory spec), so
    the bridge reconstructs the identical Tool Config / Tool Call Store across
    the process boundary. After codex exits, the durable tool evidence is read
    back from the shared Tool Call Store by the adapter; this runner returns
    the
    proposals the agent wrote plus control cost.
    """

    def __init__(
        self,
        *,
        sqlite_path: str,
        reward_policy_json: str,
        evaluator_spec: str,
        codex_binary: str = "codex",
        model: str = "gpt-5.6",
        prompt_builder: (
            Callable[[OptimizationStepRequest], str] | None
        ) = None,
        timeout_s: float = 600.0,
        proposals_parser: (
            Callable[[str], tuple[Candidate, ...]] | None
        ) = None,
        tool_result_reader: (
            Callable[[str, ToolConfig], tuple[CodexToolCallLog, ...]] | None
        ) = None,
    ) -> None:
        self._sqlite_path = sqlite_path
        self._reward_policy_json = reward_policy_json
        self._evaluator_spec = evaluator_spec
        self._codex_binary = codex_binary
        self._model = model
        self._prompt_builder = prompt_builder or _default_prompt
        self._timeout_s = timeout_s
        self._proposals_parser = proposals_parser
        self._tool_result_reader = tool_result_reader

    def run(
        self,
        request: OptimizationStepRequest,
        tool_config: ToolConfig,
    ) -> CodexRunResult:
        if shutil.which(self._codex_binary) is None:
            raise OpaqueStepError(
                f"codex binary {self._codex_binary!r} not found on PATH"
            )
        mcp_env = {
            "WS_MCP_SQLITE_PATH": self._sqlite_path,
            "WS_MCP_TOOL_CONFIG": tool_config.model_dump_json(),
            "WS_MCP_REWARD_POLICY": self._reward_policy_json,
            "WS_MCP_EVALUATOR": self._evaluator_spec,
        }
        argv = build_codex_command(
            prompt=self._prompt_builder(request),
            codex_binary=self._codex_binary,
            model=self._model,
            mcp_server_command=(
                sys.executable,
                "-m",
                "whetstone.optimization.mcp_server",
            ),
            mcp_env=mcp_env,
        )
        completed = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=self._timeout_s,
            env={**os.environ},
            check=False,
        )
        if completed.returncode != 0:
            raise OpaqueStepError(
                "codex exec exited non-zero "
                f"({completed.returncode}): {completed.stderr[:2000]}"
            )
        stdout = completed.stdout
        proposals = (
            self._proposals_parser(stdout)
            if self._proposals_parser is not None
            else _parse_proposals_from_jsonl(stdout, request)
        )
        tool_calls = (
            self._tool_result_reader(stdout, tool_config)
            if self._tool_result_reader is not None
            else ()
        )
        return CodexRunResult(
            proposals=proposals,
            tool_calls=tool_calls,
            conversation_evidence={"codex_stdout_tail": stdout[-2000:]},
            control_cost={"wall_clock_source": "subprocess"},
        )


def _default_prompt(request: OptimizationStepRequest) -> str:
    bases = [c.base_ref for c in request.candidates]
    target = request.output_contract.returned_proposal_count
    return (
        "You are optimizing an encoder user_prompt_template. Call the "
        "evaluate_candidate MCP tool to measure candidate templates, then "
        f"return exactly {target} proposals, one per Model Route "
        f"({', '.join(sorted(set(bases)))}). Change only the "
        "user_prompt_template; keep each proposal bound to its base."
    )


def _parse_proposals_from_jsonl(
    stdout: str, request: OptimizationStepRequest
) -> tuple[Candidate, ...]:  # pragma: no cover - real-CLI path
    """Best-effort parse of proposals from codex --json event output.

    The live smoke test proves connectivity + one tool call, not proposal
    fidelity, so this is deliberately permissive: if no proposals are found the
    caller records the empty tuple and the adapter fails the contract cleanly.
    """
    for raw in reversed(stdout.splitlines()):
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        proposals = _extract_proposals(event)
        if proposals:
            return proposals
    return ()


def _extract_proposals(
    event: dict[str, Any],
) -> tuple[Candidate, ...]:  # pragma: no cover - real-CLI path
    candidates_raw: Any = None
    if isinstance(event, dict):
        candidates_raw = event.get("proposals")
    if not isinstance(candidates_raw, list):
        return ()
    out: list[Candidate] = []
    for item in candidates_raw:
        if not isinstance(item, dict):
            return ()
        out.append(
            Candidate(
                candidate_id=str(item.get("candidate_id", "")),
                base_ref=str(item.get("base_ref", "")),
                payload={
                    "user_prompt_template": str(item.get("template", ""))
                },
            )
        )
    return tuple(out)
