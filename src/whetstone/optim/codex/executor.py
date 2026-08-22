"""The one production dr-exec ``Executor`` construction site.

Codex is the only optimizer that spawns a foreign binary, so it owns the
repository's single ``ProcessExecutor``. Every other execution path in
whetstone runs importable-JSON jobs in process.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dr_exec import (
    DirectoryRunStore,
    Executor,
    IsolatedHostPythonRuntime,
    ProcessExecutor,
)


def build_codex_executor(*, run_root: Path) -> Executor:
    """Build the executor that spawns Codex under dr-exec containment.

    The runtime's executable is the *host* Python, not the Codex binary:
    :class:`~whetstone.optim.codex.runner.SubprocessCodexRunner` builds an
    argv that runs a one-line ``os.execv`` shim under ``python -I``, so the
    spawned process replaces itself with the real ``codex`` binary while
    dr-exec still sees a Python entry it can account for.

    ``run_root`` must be a directory this process may write; the run store
    keeps one durable record per job beneath it.
    """
    run_root.mkdir(parents=True, exist_ok=True)
    return ProcessExecutor(
        IsolatedHostPythonRuntime(Path(sys.executable)),
        DirectoryRunStore(run_root),
    )


__all__ = ["build_codex_executor"]
