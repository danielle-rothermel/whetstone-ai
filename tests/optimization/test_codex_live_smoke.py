"""ONE live Codex smoke test: the real CLI connects to the MCP bridge.

Env-gated by ``WS_LIVE_CODEX=1`` (the sole permitted skip: a live smoke test
that reaches the real ``codex`` CLI). It is also skipped when ``codex`` is not
on PATH or ``codex login status`` does not report authenticated, both checked
at test time. Every other test in this suite runs unconditionally.

It proves the real ``codex exec`` can launch the whetstone MCP server
subprocess
and call ``evaluate_candidate`` at least once against the shipped stub
evaluator: after the run, the shared durable Tool Call Store shows at least one
accepted call under the pinned Tool Config.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import pytest
from dr_store import ObjectStore, SqliteBackend

from tests.optimization.tool_support import (
    candidate,
    codex_request,
    evaluate_candidate_config,
    reward_policy,
)
from whetstone.optimization import ToolCallStore
from whetstone.optimization.codex import OpaqueStepError
from whetstone.optimization.codex_runner import SubprocessCodexRunner

_ROUTES = ("route-0", "route-1", "route-2", "route-3")


def _codex_authenticated() -> bool:
    if shutil.which("codex") is None:
        return False
    try:
        result = subprocess.run(
            ["codex", "login", "status"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        return False
    # `codex login status` prints its status to stdout or stderr depending on
    # the build; inspect both.
    combined = f"{result.stdout}\n{result.stderr}".lower()
    return "logged in" in combined


_LIVE = os.environ.get("WS_LIVE_CODEX") == "1"


@pytest.mark.skipif(
    not _LIVE,
    reason="live Codex smoke is env-gated: set WS_LIVE_CODEX=1 to run",
)
@pytest.mark.skipif(
    not _codex_authenticated(),
    reason="codex not on PATH or not authenticated (codex login status)",
)
def test_live_codex_connects_to_bridge_and_calls_evaluate_candidate():
    # Default to the codex account's own default model; the design's
    # illustrative gpt-5.6 is not a real ChatGPT-account model.
    model = os.environ.get("WS_LIVE_CODEX_MODEL", "")
    with tempfile.TemporaryDirectory() as tmp:
        sqlite_path = os.path.join(tmp, "tool_calls.sqlite")
        # Materialize the shared backend the MCP subprocess will reopen.
        store = ObjectStore(SqliteBackend(sqlite_path))
        tool_store = ToolCallStore(store)
        config = evaluate_candidate_config(capacity=20, namespace="live-ns")

        prompt = (
            "Call the evaluate_candidate tool EXACTLY ONCE with call_id "
            "'live-smoke-1', model_route 'route-0', and template 'terse "
            "spec: state signature and edge cases'. Then stop. You do not "
            "need to return proposals for this smoke check."
        )

        runner = SubprocessCodexRunner(
            sqlite_path=sqlite_path,
            reward_policy_json=reward_policy().model_dump_json(),
            evaluator_spec=(
                "whetstone.optimization.stub_evaluator:make_stub_evaluator"
            ),
            codex_binary="codex",
            model=model,
            prompt_builder=lambda _request: prompt,
            timeout_s=300.0,
        )
        request = codex_request(
            config=config,
            candidates=tuple(
                candidate(f"S{i}", route, "base template")
                for i, route in enumerate(_ROUTES)
            ),
        )

        try:
            runner.run(request, config)
        except OpaqueStepError as exc:  # pragma: no cover - live diagnostics
            pytest.fail(f"live codex run failed: {exc}")

        # The real CLI connected to the bridge and called the tool at least
        # once: the shared durable Tool Call Store shows an accepted call.
        accepted = tool_store.accepted_count(config.identity_hash())
        assert accepted >= 1, (
            "expected the live codex agent to make at least one accepted "
            f"evaluate_candidate call; accepted={accepted}"
        )
