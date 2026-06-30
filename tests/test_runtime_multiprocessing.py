"""Legacy v0 script runtime multiprocessing contract tests.

These lock the fork/spawn policy used by v0 Typer entrypoints through
``dr_dspy.runtime.run_typer_app``. New platform entrypoints should keep runtime
setup at their CLI boundary rather than changing this legacy policy silently.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_configure_multiprocessing_uses_legacy_start_method_policy() -> None:
    script = textwrap.dedent(
        """
        import platform
        import multiprocessing as mp

        from dr_dspy.runtime import configure_multiprocessing

        configure_multiprocessing()
        expected = "spawn" if platform.system() == "Windows" else "fork"
        if mp.get_start_method() != expected:
            raise SystemExit(
                f"expected {expected}, got {mp.get_start_method()}"
            )
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
