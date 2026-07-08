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


def test_provider_result_conversion_defers_recording_and_psycopg() -> None:
    script = textwrap.dedent(
        """
        import sys

        from dr_providers import LlmResponse

        import whetstone.lm.boundary as boundary

        boundary.provider_result_from_response(
            LlmResponse(
                text="ok",
                provider_metadata={"usage": {"total_tokens": 3}},
            )
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
