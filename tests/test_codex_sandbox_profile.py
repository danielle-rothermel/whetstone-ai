"""What the macOS sandbox profile actually enforces.

These assert kernel-enforced facts, so they need ``sandbox-exec`` and
skip elsewhere. The guarantees whetstone enforces in Python -- the
environment allowlist, the run lease binding, ledger totality, selection,
and timeout terminalization -- are asserted without a sandbox in
:mod:`tests.test_codex_containment_boundary` and run on every platform.
"""

from __future__ import annotations

import subprocess
import sys
import sqlite3
from pathlib import Path

import pytest

from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.codex.runner import (
    SubprocessCodexRunner,
    _MacOsProcessIsolation,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="the Codex sandbox is macOS sandbox-exec only",
)


class _NeverRunExecutor:
    def run_blocking(self, job):
        raise AssertionError("this test must not spawn a Codex process")


def _runner(tmp_path, sqlite_store, store_path: Path) -> SubprocessCodexRunner:
    runtime_config = ReferenceEvalRuntimeConfig()
    return SubprocessCodexRunner(
        executor=_NeverRunExecutor(),
        sqlite_path=str(store_path),
        runtime_config=runtime_config,
        runtime_config_class=(
            "whetstone.eval.reference_runtime:ReferenceEvalRuntimeConfig"
        ),
        reward_policy=runtime_config.build_engine(sqlite_store).reward_policy,
        environment={"PATH": "/usr/bin"},
    )


def _run_under_agent_profile(
    tmp_path,
    runner: SubprocessCodexRunner,
    *,
    source: str,
) -> subprocess.CompletedProcess[str]:
    """Run a probe under exactly the profile the Codex process gets."""
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    command = _MacOsProcessIsolation().wrap(
        [sys.executable, "-I", "-c", source],
        profile_path=tmp_path / "probe.sb",
        readable_paths=runner._readable_runtime_paths(
            resolved_binary=Path(sys.executable),
        ),
        writable_paths=runner._writable_runtime_paths(scratch),
    )
    return subprocess.run(
        command, capture_output=True, text=True, check=False
    )


def test_the_agent_profile_cannot_open_the_store_for_write(
    tmp_path,
    sqlite_store,
) -> None:
    """The ledger is outside the agent's writable set.

    ``whetstone_tool_admission_capacity`` is the per-run cap on paid
    evaluations and ``objects`` holds the Tool Results the adapter
    rebuilds candidates from. A process that can write this file can
    clear its own budget or forge evidence, which is why the store is
    granted to the evaluation server's process and to nothing inside the
    agent's profile.
    """
    store_path = (tmp_path / "ledger.sqlite").resolve()
    connection = sqlite3.connect(store_path)
    connection.execute("create table capacity (used integer)")
    connection.execute("insert into capacity values (1)")
    connection.commit()
    connection.close()

    runner = _runner(tmp_path, sqlite_store, store_path)
    probe = _run_under_agent_profile(
        tmp_path,
        runner,
        source=(
            "import sqlite3\n"
            "try:\n"
            f"    c = sqlite3.connect({str(store_path)!r})\n"
            "    c.execute('delete from capacity')\n"
            "    c.commit()\n"
            "    print('WRITE_OK')\n"
            "except Exception as exc:\n"
            "    print('DENIED', type(exc).__name__)\n"
        ),
    )

    assert "WRITE_OK" not in probe.stdout, probe.stdout
    assert "DENIED" in probe.stdout, (probe.stdout, probe.stderr)

    # The rows the agent tried to clear are still there.
    verify = sqlite3.connect(store_path)
    assert verify.execute("select count(*) from capacity").fetchone()[0] == 1
    verify.close()


def test_the_agent_profile_may_write_only_its_own_scratch(
    tmp_path,
    sqlite_store,
) -> None:
    """The scratch directory is the whole writable set.

    This is the positive half: the agent must be able to write its own
    working directory, or it could not produce an output artifact.
    """
    store_path = (tmp_path / "ledger.sqlite").resolve()
    store_path.touch()
    runner = _runner(tmp_path, sqlite_store, store_path)
    scratch = (tmp_path / "scratch").resolve()

    probe = _run_under_agent_profile(
        tmp_path,
        runner,
        source=(
            "from pathlib import Path\n"
            f"Path({str(scratch / 'artifact.json')!r}).write_text('{{}}')\n"
            "print('SCRATCH_WRITE_OK')\n"
        ),
    )

    assert "SCRATCH_WRITE_OK" in probe.stdout, (probe.stdout, probe.stderr)
    assert runner._writable_runtime_paths(scratch) == (scratch,)
