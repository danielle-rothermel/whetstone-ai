"""Launchable entrypoint for the whetstone evaluate_candidate MCP server.

``codex exec`` launches this module as a stdio MCP server subprocess:

    python -m whetstone.optimization.mcp_server

Configuration crosses the process boundary through the environment (so the
Codex ``mcp_servers`` command config stays a fixed argv):

* ``WS_MCP_SQLITE_PATH`` -- the shared dr-store SqliteBackend file the
  whetstone process also opens, so Tool Call Store acceptance/refusal/
  completion and the ``max_evaluation_calls`` capacity accounting are the SAME
  durable state across the process boundary and survive a bridge restart.
* ``WS_MCP_TOOL_CONFIG`` — the serialized (JSON) :class:`ToolConfig` the
  whetstone process pinned; the server reconstructs the identical Tool Config
  (and therefore ``tool_config_hash``) from it.
* ``WS_MCP_REWARD_POLICY`` — the serialized (JSON) :class:`RewardPolicy` whose
  Identity Hash the Tool Config's ``reward_policy_ref`` names.
* ``WS_MCP_EVALUATOR`` — a dotted ``module:callable`` factory returning a
  :class:`~whetstone.optimization.tool_eval.ToolEvaluator`; it is the seam that
  lets the live smoke test inject a deterministic stub evaluator (so the real
  CLI proves connectivity without a real graph run).

Nothing here manufactures a score: the evaluator produces internal-role
evaluation evidence and the Reward Policy scalarizes it, exactly as the
in-process tool-using path does.
"""

from __future__ import annotations

import importlib
import os
import sys

from dr_store import ObjectStore, SqliteBackend

from whetstone.optimization.mcp_bridge import (
    EvaluateCandidateServer,
    serve_stdio,
)
from whetstone.optimization.reward import RewardPolicy
from whetstone.optimization.tool_eval import (
    EvaluatingToolExecutor,
    ToolEvaluator,
)
from whetstone.optimization.tool_store import ToolCallStore
from whetstone.optimization.tools import ToolConfig

__all__ = ["build_server_from_env", "main"]


def _load_evaluator(spec: str) -> ToolEvaluator:
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError(
            f"WS_MCP_EVALUATOR must be 'module:callable', got {spec!r}"
        )
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    return factory()


def build_server_from_env(
    environ: dict[str, str] | None = None,
) -> EvaluateCandidateServer:
    """Reconstruct the MCP server from the launch environment."""
    env = environ if environ is not None else dict(os.environ)
    sqlite_path = env["WS_MCP_SQLITE_PATH"]
    tool_config = ToolConfig.model_validate_json(env["WS_MCP_TOOL_CONFIG"])
    reward_policy = RewardPolicy.model_validate_json(
        env["WS_MCP_REWARD_POLICY"]
    )
    evaluator = _load_evaluator(env["WS_MCP_EVALUATOR"])

    store = ObjectStore(SqliteBackend(sqlite_path))
    tool_store = ToolCallStore(store)
    executor = EvaluatingToolExecutor(evaluator, reward_policy)
    return EvaluateCandidateServer(
        tool_config=tool_config, store=tool_store, executor=executor
    )


def main() -> None:
    server = build_server_from_env()
    serve_stdio(server, stdin=sys.stdin, stdout=sys.stdout)


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    main()
