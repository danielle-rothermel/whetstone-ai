"""Opt-in gate and shared world for the real-Codex ladder.

Every test in this package drives the *real* Codex CLI against a live
subscription session, so the whole package is skipped unless
``WHETSTONE_REAL_CODEX=1`` is set. The task model stays fake throughout:
evaluations run on the reference transport, so a ladder run spends Codex
agent turns and nothing else.

No test here ever reads, copies, or prints credential material. The
runner's own ``stage_auth`` copies ``~/.codex/auth.json`` into each run's
scratch ``CODEX_HOME``; that is production code and the bytes never enter
the test process.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from dr_store.sync import open_sqlite

from tests.codex_support import (
    toy_codex_control,
    toy_codex_run,
    toy_codex_step_request,
    toy_tool_args,
)
from whetstone.core.leasing import EffectLeaseAuthority, ReplayPolicy
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.adapters import MappingAdapterRegistry
from whetstone.optim.codex.adapter import CODEX_ADAPTER_KEY, CodexAdapter
from whetstone.optim.codex.containment import (
    CODEX_AUTH_FILENAMES,
    CODEX_DEFAULT_MAX_OUTPUT_BYTES,
)
from whetstone.optim.codex.executor import build_codex_executor
from whetstone.optim.codex.runner import SubprocessCodexRunner
from whetstone.optim.harness import OptimHarness
from whetstone.optim.tools.admission import _ENTRY_TABLE, ToolCallState
from whetstone.optim.tools.contracts import RefusalClass
from whetstone.optim.tools.evaluator import EngineToolEvaluator
from whetstone.optim.tools.execution import EvaluatingToolExecutor
from whetstone.optim.tools.facade import ToolAdmissionAuthority, ToolCallStore
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    toy_template_render_contract,
)

#: The opt-in. Absent or not "1", the whole ladder is skipped.
REAL_CODEX_ENV = "WHETSTONE_REAL_CODEX"
#: Where the real binary is expected. Overridable for a non-brew install.
REAL_CODEX_BINARY_ENV = "WHETSTONE_REAL_CODEX_BINARY"
DEFAULT_REAL_CODEX_BINARY = "/opt/homebrew/bin/codex"

#: Every rung is bounded well under the ladder's 180 s per-session cap.
RUNG_WALL_SECONDS = 180.0

_FIXED_LEASE_TOKEN = "e" * 64


def real_codex_binary() -> str:
    return os.environ.get(REAL_CODEX_BINARY_ENV) or DEFAULT_REAL_CODEX_BINARY


#: This conftest is nested, but pytest still calls its collection hook with
#: every collected item in the session -- including the ordinary CI suites.
#: Skipping unconditionally would silently disable all Codex coverage in CI,
#: so the hook filters to items that live under this directory.
_LADDER_ROOT = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config, items) -> None:
    """Skip the ladder -- and only the ladder -- unless opted into."""
    if os.environ.get(REAL_CODEX_ENV) == "1":
        return
    skip = pytest.mark.skip(
        reason=(
            f"real-Codex ladder is opt-in: set {REAL_CODEX_ENV}=1 "
            "(drives the real CLI against a live subscription session)"
        )
    )
    for item in items:
        try:
            path = Path(str(item.fspath)).resolve()
        except (AttributeError, OSError):
            continue
        if path.is_relative_to(_LADDER_ROOT):
            item.add_marker(skip)


#: Where macOS keeps the only process-isolation mechanism this ladder
#: will run under. Mirrors ``runner._MACOS_SANDBOX_EXEC``: a real rung
#: that reaches the runner without it raises ``OpaqueStepError``, so the
#: precondition names the same file rather than trusting the platform tag.
SANDBOX_EXEC_PATH = Path("/usr/bin/sandbox-exec")


def real_codex_precondition_failure(
    *,
    opted_in: bool,
    platform: str,
    binary_found: bool,
    binary: str,
    sandbox_exec_found: bool,
    auth_found: bool,
    auth_home: Path,
) -> str | None:
    """Why this machine cannot host the ladder, or ``None`` if it can.

    A pure function of the environment so the decision itself is
    testable without a macOS host, a Codex binary, or a live session.

    The opt-in is what makes an unmet precondition an *error*. Without
    ``WHETSTONE_REAL_CODEX=1`` the ladder is simply not requested and the
    collection hook skips it. With the opt-in, the operator asked for a
    real run, and every one of these conditions means they will not get
    one -- so the answer is a message the caller raises, never a skip.
    Skipping here is what let a Linux host, a missing binary, or an
    absent ``sandbox-exec`` produce an all-skipped session that
    ``scripts/check-real-codex.sh`` reported as "all rungs passed":
    pytest exits 0 on a fully skipped session.
    """
    if not opted_in:
        return None
    if platform != "darwin":
        return (
            f"{REAL_CODEX_ENV}=1 was set on {platform!r}, but the Codex "
            "sandbox is macOS sandbox-exec only. Run the ladder on macOS "
            f"or unset {REAL_CODEX_ENV}."
        )
    if not sandbox_exec_found:
        return (
            f"{REAL_CODEX_ENV}=1 was set but {SANDBOX_EXEC_PATH} is not "
            "present; the ladder refuses to drive the real CLI without "
            "kernel-enforced process isolation."
        )
    if not binary_found:
        return (
            f"{REAL_CODEX_ENV}=1 was set but the real Codex binary was "
            f"not found at {binary!r}; set {REAL_CODEX_BINARY_ENV} to "
            "its path."
        )
    if not auth_found:
        return (
            f"{REAL_CODEX_ENV}=1 was set but no Codex session was found "
            f"under {auth_home} ({'/'.join(CODEX_AUTH_FILENAMES)}); "
            "run `codex login` first."
        )
    return None


def _observe_real_codex_preconditions() -> str | None:
    """Read the machine, then let the pure function decide."""
    binary = real_codex_binary()
    home = Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))
    return real_codex_precondition_failure(
        opted_in=os.environ.get(REAL_CODEX_ENV) == "1",
        platform=sys.platform,
        binary_found=(
            shutil.which(binary) is not None or Path(binary).is_file()
        ),
        binary=binary,
        sandbox_exec_found=SANDBOX_EXEC_PATH.is_file(),
        # Existence only. The ladder never opens these files.
        auth_found=any(
            (home / name).is_file() for name in CODEX_AUTH_FILENAMES
        ),
        auth_home=home,
    )


@pytest.fixture(scope="session", autouse=True)
def _real_codex_preconditions() -> None:
    """Fail loudly, before any rung runs, if the machine cannot host one."""
    failure = _observe_real_codex_preconditions()
    if failure is not None:
        # exit, not fail: an unhostable machine makes every remaining
        # rung meaningless, and a session-scoped fail would be reported
        # once per rung as an error rather than once as a refusal.
        pytest.exit(failure, returncode=1)


class RealCodexWorld:
    """One real Codex Step's worth of wiring, on the toy experiment.

    Mirrors ``tests/test_codex_harness_e2e.py``'s world, with two
    deliberate differences: the binary is the real Codex CLI, and no
    ``OPENAI_API_KEY`` is injected -- the agent authenticates from the
    staged subscription session, and the *task* model stays on the
    reference (fake) transport so evaluations cost nothing.
    """

    def __init__(
        self,
        *,
        tmp_path: Path,
        store,
        sqlite_path: str,
        max_tool_calls: int,
    ) -> None:
        self.store = store
        self.tmp_path = tmp_path
        self.sqlite_path = sqlite_path
        self.engine = ReferenceEvalRuntimeConfig().build_engine(store)
        self.control = toy_codex_control(
            engine=self.engine, max_tool_calls=max_tool_calls
        )
        self.run, self.config, self.candidate = toy_codex_run(
            control=self.control, engine=self.engine
        )
        self.effect_authority = EffectLeaseAuthority.sqlite(self.sqlite_path)
        self.tool_store = ToolCallStore(
            store,
            ToolAdmissionAuthority.sqlite(self.sqlite_path),
            self.effect_authority,
        )
        self.tool_executor = EvaluatingToolExecutor(
            EngineToolEvaluator(self.engine),
            self.engine.reward_policy,
            self.effect_authority,
            owner_id="codex-real-owner",
            replay_policy=ReplayPolicy.IDEMPOTENT,
            lease_duration=timedelta(minutes=5),
        )

    def tool_args(self, template: str) -> dict:
        return toy_tool_args(
            candidate=self.candidate, engine=self.engine, template=template
        )

    def capacity_refusals(self) -> tuple[dict, ...]:
        """Every durable CAPACITY refusal this world's namespace recorded.

        A capacity refusal debits no capacity, so it is deliberately
        absent from ``admitted_entries`` -- that absence is what makes
        ``len(admitted_entries)`` agree with ``accepted_count``, and it is
        pinned by ``tests/test_admitted_entries.py``. There is therefore
        no public API that enumerates refusals, and ``find_entry`` needs a
        ``call_id`` the real agent chose for itself and never reported.

        So this reads the admission entry table directly. It is a
        test-only assertion helper: a real rung has to distinguish "the
        agent was refused" from "the agent never tried a second call",
        and only the durable ledger can tell those apart. Widening the
        production surface to enumerate refusals would add an API no
        production caller wants.
        """
        connection = sqlite3.connect(self.sqlite_path)
        try:
            rows = connection.execute(
                f"SELECT entry_json FROM {_ENTRY_TABLE} "
                "WHERE store_namespace_key = ?",
                (str(self.config.store_namespace_key),),
            ).fetchall()
        finally:
            connection.close()
        entries = [json.loads(row[0]) for row in rows]
        return tuple(
            entry
            for entry in entries
            if entry.get("state") == ToolCallState.REFUSED.value
            and (entry.get("refusal") or {}).get("refusal_class")
            == RefusalClass.CAPACITY.value
        )

    def step_request(self, **kwargs):
        return toy_codex_step_request(
            control=self.control,
            run=self.run,
            candidate=self.candidate,
            **kwargs,
        )

    def harness(self, adapter) -> OptimHarness:
        harness = OptimHarness(
            store=self.store,
            adapter_registry=MappingAdapterRegistry(
                {CODEX_ADAPTER_KEY: adapter}
            ),
            tool_store=self.tool_store,
            effect_authority=self.effect_authority,
            owner_id="codex-real-owner",
            adapter_replay_policy=ReplayPolicy.NO_REDRIVE,
            lease_duration=timedelta(minutes=5),
            tool_executor=self.tool_executor,
        )
        harness.bind_run(self.run)
        return harness

    def production_prompt(self, context, *, extra: str):
        """The production prompt, plus one rung-specific instruction.

        Rungs that need to steer the agent must extend the real prompt
        rather than replace it: the prompt is where the agent learns the
        fixed ``model_route`` and ``base_ref`` values, and a builder that
        drops them makes every call refused after admission -- which looks
        exactly like the behavior some rungs are trying to assert.

        Every one of those facts now arrives on the runner's
        ``CodexPromptContext``, so this passes them straight through. It
        used to rebuild ``model_route``, ``base_ref``, and the tool name
        from its own copies of the world -- a second derivation that
        could silently disagree with the route the Step's evaluation
        server actually advertises.
        """
        from whetstone.optim.codex.runner import _default_prompt

        return (
            _default_prompt(
                context.request,
                tool_name=context.tool_name,
                lease_token_hash=context.lease_token_hash,
                max_tool_calls=context.max_tool_calls,
                model_route=context.model_route,
                base_ref=context.base_ref,
            )
            + "\n\n"
            + extra
        )

    def runner(
        self,
        *,
        timeout_seconds: float = RUNG_WALL_SECONDS,
        reasoning_effort: str = "",
        max_output_bytes: int = CODEX_DEFAULT_MAX_OUTPUT_BYTES,
        prompt_builder=None,
    ) -> SubprocessCodexRunner:
        return SubprocessCodexRunner(
            executor=build_codex_executor(run_root=self.tmp_path / "runs"),
            sqlite_path=self.sqlite_path,
            runtime_config=ReferenceEvalRuntimeConfig(
                mutation_field=TOY_MUTATION_FIELD,
                render_contract=toy_template_render_contract(),
            ),
            runtime_config_class=(
                "whetstone.eval.reference_runtime:ReferenceEvalRuntimeConfig"
            ),
            reward_policy=self.engine.reward_policy,
            codex_binary=real_codex_binary(),
            reasoning_effort=reasoning_effort,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            prompt_builder=prompt_builder,
            # The real environment: PATH plus whatever proxy/TLS settings
            # the machine uses. No OPENAI_API_KEY -- the agent runs on the
            # staged subscription session, and the task model is fake.
            environment={
                key: value
                for key, value in os.environ.items()
                if key
                in {
                    "PATH",
                    "CODEX_HOME",
                    "SSL_CERT_FILE",
                    "SSL_CERT_DIR",
                    "HTTP_PROXY",
                    "HTTPS_PROXY",
                    "ALL_PROXY",
                    "NO_PROXY",
                }
            },
        )

    def adapter(self, runner: SubprocessCodexRunner) -> CodexAdapter:
        adapter = CodexAdapter(
            runner,
            store=self.store,
            lease_token_factory=lambda: _FIXED_LEASE_TOKEN,
        )
        adapter.bind_tool_store(self.tool_store)
        return adapter


@pytest.fixture
def real_codex_world(tmp_path):
    sqlite_path = str((tmp_path / "codex-real.sqlite").resolve())
    with open_sqlite(sqlite_path) as store:
        yield lambda max_tool_calls=4: RealCodexWorld(
            tmp_path=tmp_path,
            store=store,
            sqlite_path=sqlite_path,
            max_tool_calls=max_tool_calls,
        )
