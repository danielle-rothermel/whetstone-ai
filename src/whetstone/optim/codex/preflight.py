"""Authentication preflight for the Codex-direct optimizer.

A Codex run commits real eval capacity the moment its first Tool Call is
admitted. A run that starts without a working Codex session burns wall
time and produces nothing, so the launch path proves the session first:
the binary resolves, an auth source exists, and one cheap structured probe
returns a schema-conforming artifact.

The probe runs through the same :class:`~dr_exec.Executor` and the same
sandbox as the real run, so it exercises the containment path too.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from dr_serialize import StrictJsonDecodeError, decode_strict_json_bytes

from whetstone.optim.codex.adapter import OpaqueStepError

if TYPE_CHECKING:
    from dr_exec import Executor

#: Wall budget for the probe. It is deliberately far below a real run's:
#: the probe asks for one token of structured output, not a search.
CODEX_PREFLIGHT_WALL_SECONDS = 60.0
CODEX_PREFLIGHT_PROMPT = (
    "Reply with the JSON object {\"ready\": true} and nothing else."
)
CODEX_PREFLIGHT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ready"],
    "properties": {"ready": {"type": "boolean", "const": True}},
}
#: The auth material the Codex CLI accepts, in the order it is looked for.
CODEX_AUTH_FILENAMES = ("auth.json", ".credentials.json")
CODEX_AUTH_ENV_KEY = "OPENAI_API_KEY"


class CodexPreflightError(RuntimeError):
    """The Codex CLI cannot run a budgeted optimization step."""


def codex_auth_preflight(
    *,
    executor: Executor,
    codex_binary: str,
    environment: Mapping[str, str],
    model: str = "",
    wall_seconds: float = CODEX_PREFLIGHT_WALL_SECONDS,
) -> None:
    """Prove a usable Codex session, or raise :class:`CodexPreflightError`.

    ``environment`` is the allowed environment the run will pass through,
    not the caller's whole environment: the preflight must look at exactly
    what the run will see.
    """
    from whetstone.optim.codex.runner import (
        CodexStructuredExecutionFailure,
        SubprocessCodexRunner,
    )

    resolved = shutil.which(codex_binary, path=environment.get("PATH"))
    if resolved is None:
        raise CodexPreflightError(
            f"Codex binary {codex_binary!r} was not found on the run PATH"
        )
    _require_auth_source(environment)

    runner = SubprocessCodexRunner(
        executor=executor,
        codex_binary=codex_binary,
        model=model,
        timeout_seconds=wall_seconds,
        environment=environment,
    )
    try:
        execution = runner.run_structured_prompt(
            prompt=CODEX_PREFLIGHT_PROMPT,
            output_schema=CODEX_PREFLIGHT_SCHEMA,
        )
    except CodexStructuredExecutionFailure as exc:
        raise CodexPreflightError(
            f"Codex preflight failed: {exc}; "
            f"stderr tail: {_stderr_tail(exc.stderr)}"
        ) from exc
    except OpaqueStepError as exc:
        raise CodexPreflightError(f"Codex preflight failed: {exc}") from exc

    raw = execution.artifact_bytes
    try:
        probe = decode_strict_json_bytes(
            raw,
            max_bytes=len(raw),
            max_depth=len(raw),
        )
    except StrictJsonDecodeError as exc:
        raise CodexPreflightError(
            "Codex preflight artifact is not strict JSON; stderr tail: "
            f"{_stderr_tail(execution.stderr.encode())}"
        ) from exc
    if not isinstance(probe, dict) or probe.get("ready") is not True:
        raise CodexPreflightError(
            "Codex preflight artifact does not conform to the probe schema"
        )


def _require_auth_source(environment: Mapping[str, str]) -> None:
    if environment.get(CODEX_AUTH_ENV_KEY):
        return
    configured_home = environment.get("CODEX_HOME")
    home = (
        Path(configured_home)
        if configured_home
        else Path.home() / ".codex"
    )
    if any((home / name).is_file() for name in CODEX_AUTH_FILENAMES):
        return
    raise CodexPreflightError(
        "Codex has no usable auth source: neither "
        f"{CODEX_AUTH_ENV_KEY} nor {home}/{{"
        f"{','.join(CODEX_AUTH_FILENAMES)}}} is present"
    )


def _stderr_tail(stderr: bytes) -> str:
    return stderr.decode("utf-8", errors="replace")[-2000:]


__all__ = [
    "CODEX_AUTH_ENV_KEY",
    "CODEX_AUTH_FILENAMES",
    "CODEX_PREFLIGHT_PROMPT",
    "CODEX_PREFLIGHT_SCHEMA",
    "CODEX_PREFLIGHT_WALL_SECONDS",
    "CodexPreflightError",
    "codex_auth_preflight",
]
