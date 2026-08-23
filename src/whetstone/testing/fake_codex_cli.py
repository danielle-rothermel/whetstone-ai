"""A scripted stand-in for ``codex exec``.

The real Codex CLI is a paid, non-deterministic agent, so an end-to-end
test of the Codex optimizer drives this instead. It is not a mock of the
tool path: it connects to the real MCP evaluation endpoint whetstone
hosts and calls the real tool over authenticated HTTP, so the admission,
lease, evaluation, and ledger path under test is the production one.
Only the agent's decisions are scripted.

It ships in :mod:`whetstone.testing` rather than under ``tests/`` because
``whetstone-envs`` runs its own end-to-end test against it and cannot
import from another distribution's test tree.

Run it as ``python -m whetstone.testing.fake_codex_cli exec ...``. The
transcript JSON comes from :data:`FAKE_CODEX_TRANSCRIPT_ENV` -- inline,
not as a path, because the sandbox grants no read access to a caller's
temporary directory. Like the real CLI, it learns the endpoint from the
``mcp_servers.whetstone.url`` override and its bearer token from the
environment variable that override names.

Transcript format -- a JSON list of steps::

    [
      {"tool": "evaluate_candidate",
       "args": {"call_id": "c1", "base_ref": {...},
                "model_route": "...", "template": "..."}},
      {"final": {"selected_call_id": "c1"}},
      {"exit": 3}
    ]

A ``hang`` step blocks forever, so the caller's wall budget is what ends
the process -- the way a real agent that overruns its budget ends.

An ``error`` step writes an ``{"type": "error", "message": ...}`` item to
stdout, and a ``stderr`` step writes a line to stderr, which together
reproduce a real failing launch: the cause on stdout, advisories on
stderr.

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
FAKE_CODEX_CLI_MODULE = "whetstone.testing.fake_codex_cli"

#: Where this CLI echoes the prompt it was given, inside the artifact's
#: ``conversation_evidence``. The runner preserves that field under
#: ``conversation_evidence["agent"]``, so a test reads the emitted prompt
#: back from the persisted artifact without a production seam.
FAKE_CODEX_PROMPT_EVIDENCE_KEY = "fake_codex_prompt"

#: Keys of the JSONL events this CLI writes to stdout. The runner parses
#: them with the same strict decoder it uses for the real CLI.
_EVENT_TYPE_KEY = "type"
_EVENT_TOOL_CALL = "tool_call"
_EVENT_FINAL = "final"
#: The real CLI reports an unrecoverable turn failure as an ``error``
#: item on *stdout* under ``--json``, not on stderr. A transcript emits
#: one so the runner's failure path can be tested against the shape the
#: real binary produces.
_EVENT_ERROR = "error"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parsed = _parse_args(args)
    prompt = _prompt_from_args(args)
    endpoint = _endpoint_from_args(args)
    transcript = _load_transcript()
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
                    endpoint=endpoint,
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
        elif "error" in step:
            # An ``error`` item on stdout, the way the real CLI reports a
            # failed turn under ``--json``.
            _emit(
                {_EVENT_TYPE_KEY: _EVENT_ERROR, "message": str(step["error"])}
            )
        elif "stderr" in step:
            # Startup advisories the real CLI writes to stderr. These are
            # the noise a failure message must not mistake for its cause.
            sys.stderr.write(str(step["stderr"]) + "\n")
            sys.stderr.flush()
        elif "exit" in step:
            exit_code = int(step["exit"])
        elif "hang" in step:
            # Never returns. The caller's wall budget is what stops this
            # process, which is how a real agent that overruns its budget
            # ends, so the runner's timeout path is exercised for real.
            _hang_until_killed()
        else:
            raise ValueError(f"unrecognized fake Codex transcript step: {step}")

    artifact: dict[str, Any] = {
        "run_id": parsed.run_id,
        "evaluated_call_ids": evaluated,
        "selected_call_id": None,
        "lease_token_hash": parsed.lease_token_hash,
        # The prompt the runner actually emitted, echoed back through the
        # field the agent's own evidence already travels on. A real agent
        # reports what it was told; this one reports it verbatim, which is
        # what lets a test assert on the prompt the runner built rather
        # than on one the test rebuilt for itself.
        "conversation_evidence": {FAKE_CODEX_PROMPT_EVIDENCE_KEY: prompt},
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


def _prompt_from_args(args: list[str]) -> str:
    """The prompt, which the real command takes as its last positional.

    ``build_codex_command`` appends it after every flag, so the trailing
    argument is it.
    """
    if not args:
        raise ValueError("fake Codex requires a prompt argument")
    return args[-1]


def _option(args: list[str], name: str) -> str | None:
    for index, value in enumerate(args):
        if value == name and index + 1 < len(args):
            return args[index + 1]
    return None


class _Endpoint:
    """Where the evaluation server is, and the token to reach it."""

    __slots__ = ("auth_token", "url")

    def __init__(self, *, url: str, auth_token: str) -> None:
        self.url = url
        self.auth_token = auth_token


def _endpoint_from_args(args: list[str]) -> _Endpoint:
    """Read the endpoint exactly as the real CLI is configured with it.

    ``mcp_servers.whetstone.url`` names a server whetstone already runs,
    and ``mcp_servers.whetstone.bearer_token_env_var`` names the variable
    holding this run's token, so the token never appears in argv.
    """
    overrides: dict[str, str] = {}
    prefix = "mcp_servers.whetstone."
    for index, value in enumerate(args):
        if value != "-c" or index + 1 >= len(args):
            continue
        override = args[index + 1]
        if not override.startswith(prefix):
            continue
        key, _, raw = override[len(prefix) :].partition("=")
        overrides[key] = json.loads(raw)
    url = overrides.get("url")
    token_env = overrides.get("bearer_token_env_var")
    if not url or not token_env:
        raise ValueError(
            "fake Codex requires mcp_servers.whetstone.url and "
            "mcp_servers.whetstone.bearer_token_env_var"
        )
    token = os.environ.get(token_env)
    if not token:
        raise ValueError(
            f"fake Codex found no bearer token in {token_env!r}"
        )
    return _Endpoint(url=url, auth_token=token)


async def _call_tool(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    endpoint: _Endpoint,
) -> dict[str, Any]:
    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    headers = {"authorization": f"Bearer {endpoint.auth_token}"}
    async with httpx2.AsyncClient(headers=headers) as http_client:
        async with streamable_http_client(
            endpoint.url, http_client=http_client
        ) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
    structured = getattr(result, "structured_content", None)
    return {
        "is_error": bool(getattr(result, "is_error", False)),
        "structured_content": structured,
    }


def _load_transcript() -> list[dict[str, Any]]:
    """Read the transcript the runner granted this process.

    The value is the transcript JSON itself, not a path. The Codex
    process runs under a sandbox profile whose only readable roots are
    the scratch directory and the run's own state paths, so a path into
    a test's temporary directory would be denied; passing the document
    inline keeps the containment profile untouched.
    """
    raw = os.environ.get(FAKE_CODEX_TRANSCRIPT_ENV)
    if not raw:
        raise ValueError(
            f"fake Codex requires {FAKE_CODEX_TRANSCRIPT_ENV} to carry its "
            "transcript JSON"
        )
    loaded = json.loads(raw)
    if not isinstance(loaded, list):
        raise ValueError("fake Codex transcript must be a JSON list")
    return loaded


def _hang_until_killed() -> None:
    """Block until the wall budget stops the process group."""
    import time

    while True:
        time.sleep(3600)


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
    "FAKE_CODEX_PROMPT_EVIDENCE_KEY",
    "FAKE_CODEX_TRANSCRIPT_ENV",
    "install_fake_codex_binary",
    "main",
]
