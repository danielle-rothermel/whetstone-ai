from __future__ import annotations

import subprocess
import sys
import textwrap


def test_lm_boundary_import_does_not_load_provider_or_dspy_modules() -> None:
    script = textwrap.dedent(
        """
        import sys

        import dr_dspy.lm.boundary

        blocked = ("dspy", "openai", "httpx", "dbos", "psycopg")
        loaded = [module for module in blocked if module in sys.modules]
        if loaded:
            raise SystemExit(",".join(loaded))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
