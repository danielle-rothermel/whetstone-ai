"""ONE env-gated live smoke test for the runner (OFF by default).

Gated by ``WS_LIVE_VALIDATE=1``. This is the only permitted skip in the runner
suite: a genuine live paid call to OpenRouter's canonical task route. Every
other runner test runs unconditionally against fakes. This smoke test drives a
single-instance pilot through the REAL dr-providers transport to prove the
route registry + stage-03 driver reach a live endpoint. It is never run in the
normal gate (``WS_LIVE_VALIDATE`` unset -> skipped) so the workflow makes NO
live paid calls.
"""

from __future__ import annotations

import os

import pytest

from whetstone.runner.pilot import run_pilot
from whetstone.runner.routes import canonical_task_route

_LIVE = os.environ.get("WS_LIVE_VALIDATE") == "1"


@pytest.mark.skipif(
    not _LIVE,
    reason="live validate smoke is env-gated: set WS_LIVE_VALIDATE=1 to run",
)
def test_live_pilot_reaches_openrouter() -> None:  # pragma: no cover - live
    from dr_providers.transport import HttpProvider

    route = canonical_task_route(temperature=0.0)
    provider = HttpProvider(policy=route.transport_policy)
    report = run_pilot(
        env="c11",
        lane="openrouter",
        model=route.model,
        transport=provider.invoke,
        execution_policy=route.execution_policy,
        instance_count=1,
    )
    # A live call produced at least one per-call spot-record with real tokens.
    assert report.calls
    assert any(c.total_tokens for c in report.calls)
