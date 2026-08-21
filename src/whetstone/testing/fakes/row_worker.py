from __future__ import annotations

import time
from pathlib import Path

from pydantic import JsonValue

from whetstone.eval.drivers.graph_worker import run_row

__all__ = [
    "GATED_ROW_RELEASE_ENV",
    "WATCHDOG_SECONDS",
    "gated_run_row",
]

#: Request key naming the file whose existence releases a gated row.
GATED_ROW_RELEASE_ENV = "whetstone_gated_row_release_path"

#: Ceiling on how long a gated row blocks when nothing ever releases it.
#:
#: This is a framework watchdog, not the contract under test: reaching it
#: means the caller's wall budget never fired, which is a test failure.
WATCHDOG_SECONDS = 120.0

_POLL_SECONDS = 0.01


def gated_run_row(payload: JsonValue) -> JsonValue:
    """Run one row only after its release gate opens.

    The gate is a path carried in ``prompt_inputs``. A caller that never
    creates the file holds the row open until its wall budget terminates the
    worker, which is how a per-row deadline is exercised without sleeping for
    a fixed duration.
    """

    release_path = _release_path(payload)
    if release_path is not None:
        deadline = time.monotonic() + WATCHDOG_SECONDS
        while not release_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "the gated row watchdog expired before its release gate "
                    "opened and before any wall budget stopped it"
                )
            time.sleep(_POLL_SECONDS)
    return run_row(_without_gate(payload))


def _release_path(payload: JsonValue) -> Path | None:
    prompt_inputs = _prompt_inputs(payload)
    raw = prompt_inputs.get(GATED_ROW_RELEASE_ENV)
    return None if raw is None else Path(str(raw))


def _prompt_inputs(payload: JsonValue) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    prompt_inputs = payload.get("prompt_inputs")
    return prompt_inputs if isinstance(prompt_inputs, dict) else {}


def _without_gate(payload: JsonValue) -> JsonValue:
    if not isinstance(payload, dict):
        return payload
    prompt_inputs = _prompt_inputs(payload)
    if GATED_ROW_RELEASE_ENV not in prompt_inputs:
        return payload
    stripped = dict(payload)
    stripped["prompt_inputs"] = {
        key: value
        for key, value in prompt_inputs.items()
        if key != GATED_ROW_RELEASE_ENV
    }
    return stripped  # type: ignore[return-value]
