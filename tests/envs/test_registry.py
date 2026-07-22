"""Registry-level checks for the bound env specs (no live calls).

Covers the committed per-env token estimates the pilot's token-sanity check
falls back to when ``--spec-estimate-tokens`` is not passed. Values are pinned
verbatim from each baseline-spec §5 ("Per-instance token estimate") total row.
"""

from __future__ import annotations

import pytest

from whetstone.envs.registry import ENV_NAMES, TokenEstimate, env_spec

#: The committed per-env (naive, ceiling) total-tokens/call from each spec §5.
_EXPECTED: dict[str, TokenEstimate] = {
    "c11": TokenEstimate(naive=350, ceiling=800),
    "c19": TokenEstimate(naive=280, ceiling=1150),
    "c18": TokenEstimate(naive=253, ceiling=650),
    "c22": TokenEstimate(naive=170, ceiling=420),
    "c23": TokenEstimate(naive=140, ceiling=420),
}


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_every_env_has_committed_token_estimate(env_name: str) -> None:
    estimate = env_spec(env_name).token_estimate
    assert estimate == _EXPECTED[env_name]
    # Both probes are distinguished by every spec: naive is the cheaper probe.
    assert estimate.naive < estimate.ceiling
    assert estimate.naive > 0


def test_token_estimates_cover_exactly_the_five_envs() -> None:
    assert set(_EXPECTED) == set(ENV_NAMES)
