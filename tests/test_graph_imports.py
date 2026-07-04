from __future__ import annotations

import subprocess
import sys
import textwrap


def test_graph_import_does_not_load_lm_or_eval_failures() -> None:
    script = textwrap.dedent(
        """
        import sys

        import whetstone.graph

        blocked = (
            "dspy",
            "openai",
            "httpx",
            "dbos",
            "psycopg",
            "whetstone.lm",
            "whetstone.eval_failures",
        )
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


def test_node_error_from_exception_does_not_load_policy() -> None:
    script = textwrap.dedent(
        """
        import sys

        from whetstone.eval_failures import PermanentFailureError
        from whetstone.graph.models import NodeError

        NodeError.from_exception(
            PermanentFailureError(
                "classified failure",
                underlying=ValueError("bad payload"),
                metadata={"stage": "parse"},
            )
        )

        if "whetstone.eval_failures.policy" in sys.modules:
            raise SystemExit("whetstone.eval_failures.policy")
        if "whetstone.lm" in sys.modules:
            raise SystemExit("whetstone.lm")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
