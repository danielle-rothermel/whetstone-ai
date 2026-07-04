from __future__ import annotations

import subprocess
import sys
import textwrap


def test_lm_boundary_import_does_not_load_provider_or_dspy_modules() -> None:
    script = textwrap.dedent(
        """
        import sys

        import whetstone.lm.boundary

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


def test_parse_provider_response_does_not_load_recording_or_psycopg() -> None:
    script = textwrap.dedent(
        """
        import sys

        import whetstone.lm.boundary as boundary

        boundary.parse_provider_response(
            {
                "choices": [
                    {"message": {"content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {"total_tokens": 3},
            },
            config=boundary.openrouter_chat_config(model="model/test"),
        )

        blocked = ("psycopg", "whetstone.eval_failures.recording")
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


def test_serialization_import_does_not_load_dspy_module() -> None:
    script = textwrap.dedent(
        """
        import sys

        from whetstone.eval_failures import ensure_recordable

        ensure_recordable({"telemetry": {"ok": True}})

        if "dspy" in sys.modules:
            raise SystemExit("dspy")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
