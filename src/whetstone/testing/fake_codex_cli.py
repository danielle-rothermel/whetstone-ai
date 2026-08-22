"""A scripted stand-in for ``codex exec``.

The real Codex CLI is a paid, non-deterministic agent, so an end-to-end
test of the Codex optimizer drives this instead. It is not a mock of the
tool path: it spawns the real MCP evaluation server over stdio and calls
the real tool, so the admission, lease, evaluation, and ledger path under
test is the production one. Only the agent's decisions are scripted.

It ships in :mod:`whetstone.testing` rather than under ``tests/`` because
``whetstone-envs`` runs its own end-to-end test against it and cannot
import from another distribution's test tree.

Run it as ``python -m whetstone.testing.fake_codex_cli exec ...``. The
transcript path comes from :data:`FAKE_CODEX_TRANSCRIPT_ENV`.

Transcript format -- a JSON list of steps::

    [
      {"tool": "evaluate_candidate",
       "args": {"call_id": "c1", "base_ref": {...},
                "model_route": "...", "template": "..."}},
      {"final": {"selected_call_id": "c1"}},
      {"exit": 3}
    ]

A ``final`` step supplies the fields the harness does not already know;
``run_id``, ``lease_token_hash``, and ``evaluated_call_ids`` are filled in
from the CLI arguments and the calls actually made unless the transcript
overrides them. An ``exit`` step sets a non-zero exit code.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

FAKE_CODEX_TRANSCRIPT_ENV = "WS_FAKE_CODEX_TRANSCRIPT"
FAKE_CODEX_MCP_MODULE = "whetstone.optim.codex.mcp_server"
FAKE_CODEX_CLI_MODULE = "whetstone.testing.fake_codex_cli"

#: Keys of the JSONL events this CLI writes to stdout. The runner parses
#: them with the same strict decoder it uses for the real CLI.
_EVENT_TYPE_KEY = "type"
_EVENT_TOOL_CALL = "tool_call"
_EVENT_FINAL = "final"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parsed = _parse_args(args)
    transcript = _load_transcript()

    mcp_env = _mcp_env_from_args(args)
    evaluated: list[str] = []
    final_overrides: dict[str, Any] = {}
    exit_code = 0

    for step in transcript:
        if "tool" in step:
            call_args = dict(step["args"])
            result = asyncio.run(
                _call_tool(
                    tool_name=str(step["tool"]),
                    arguments=call_args,
                    env=mcp_env,
                )
            )
            evaluated.append(str(call_args["call_id"]))
            _emit(
                {
                    _EVENT_TYPE_KEY: _EVENT_TOOL_CALL,
                    "call_id": call_args["call_id"],
                    "is_error": result.get("is_error", False),
                }
            )
        elif "final" in step:
            final_overrides = dict(step["final"])
        elif "exit" in step:
            exit_code = int(step["exit"])
        else:
            raise ValueError(f"unrecognized fake Codex transcript step: {step}")

    artifact: dict[str, Any] = {
        "run_id": parsed.run_id,
        "evaluated_call_ids": evaluated,
        "selected_call_id": None,
        "lease_token_hash": parsed.lease_token_hash,
        "conversation_evidence": {},
        "control_cost": {},
    }
    artifact.update(final_overrides)
    _emit({_EVENT_TYPE_KEY: _EVENT_FINAL, "run_id": artifact["run_id"]})
    Path(parsed.output_artifact_path).write_text(
        json.dumps(artifact, sort_keys=True), encoding="utf-8"
    )
    return exit_code


class _ParsedArgs:
    __slots__ = ("lease_token_hash", "output_artifact_path", "run_id")

    def __init__(
        self,
        *,
        run_id: str,
        lease_token_hash: str,
        output_artifact_path: str,
    ) -> None:
        self.run_id = run_id
        self.lease_token_hash = lease_token_hash
        self.output_artifact_path = output_artifact_path


def _parse_args(args: list[str]) -> _ParsedArgs:
    """Read only the arguments the real command receives that matter here.

    ``--output-schema`` pins ``run_id`` and ``lease_token_hash`` as JSON
    Schema constants, which is exactly how the real CLI learns them, so
    the fake reads them from there rather than being told separately.
    """
    schema_path = _option(args, "--output-schema")
    artifact_path = _option(args, "--output-last-message")
    if schema_path is None or artifact_path is None:
        raise ValueError(
            "fake Codex requires --output-schema and --output-last-message"
        )
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    properties = schema["properties"]
    return _ParsedArgs(
        run_id=str(properties["run_id"]["const"]),
        lease_token_hash=str(properties["lease_token_hash"]["const"]),
        output_artifact_path=artifact_path,
    )


def _option(args: list[str], name: str) -> str | None:
    for index, value in enumerate(args):
        if value == name and index + 1 < len(args):
            return args[index + 1]
    return None


def _mcp_env_from_args(args: list[str]) -> dict[str, str]:
    """Rebuild the MCP server environment from the ``-c`` overrides.

    The real CLI receives it as ``mcp_servers.whetstone.env.KEY=<json>``
    and passes it to the server process; the fake does the same so the
    server sees exactly the environment the runner granted.
    """
    prefix = "mcp_servers.whetstone.env."
    env: dict[str, str] = {}
    for index, value in enumerate(args):
        if value != "-c" or index + 1 >= len(args):
            continue
        override = args[index + 1]
        if not override.startswith(prefix):
            continue
        key, _, raw = override[len(prefix) :].partition("=")
        env[key] = json.loads(raw)
    return env


async def _call_tool(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    env: dict[str, str],
) -> dict[str, Any]:
    from mcp import ClientSession, StdioServerParameters, stdio_client

    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", FAKE_CODEX_MCP_MODULE],
        env={**_inherited_env(), **env},
    )
    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
    structured = getattr(result, "structured_content", None)
    return {
        "is_error": bool(getattr(result, "is_error", False)),
        "structured_content": structured,
    }


def _inherited_env() -> dict[str, str]:
    """The minimum the spawned server needs to import whetstone."""
    keys = ("PATH", "PYTHONPATH", "HOME", "VIRTUAL_ENV")
    return {key: os.environ[key] for key in keys if key in os.environ}


def _load_transcript() -> list[dict[str, Any]]:
    raw = os.environ.get(FAKE_CODEX_TRANSCRIPT_ENV)
    if not raw:
        raise ValueError(
            f"fake Codex requires {FAKE_CODEX_TRANSCRIPT_ENV} to name a "
            "transcript file"
        )
    loaded = json.loads(Path(raw).read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError("fake Codex transcript must be a JSON list")
    return loaded


def _emit(event: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(event, sort_keys=True) + "\n")
    sys.stdout.flush()


def install_fake_codex_binary(directory: Path) -> Path:
    """Write an executable ``codex`` shim into ``directory`` and return it.

    :class:`~whetstone.optim.codex.runner.SubprocessCodexRunner` resolves
    its binary with ``shutil.which`` against the run PATH, so the fake has
    to be a real executable file rather than a Python entry point. Put
    ``directory`` first on the PATH the runner is given.
    """
    directory.mkdir(parents=True, exist_ok=True)
    shim = directory / "codex"
    shim.write_text(
        "#!/bin/sh\n"
        f"exec {shlex.quote(sys.executable)} -m "
        f'{FAKE_CODEX_CLI_MODULE} "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return shim


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FAKE_CODEX_CLI_MODULE",
    "FAKE_CODEX_MCP_MODULE",
    "FAKE_CODEX_TRANSCRIPT_ENV",
    "install_fake_codex_binary",
    "main",
]
