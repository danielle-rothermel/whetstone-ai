"""Registry-level checks for the bound env specs (no live calls).

Covers the per-env token estimates the pilot's token-sanity check falls back to
when ``--spec-estimate-tokens`` is not passed. After the round-2 robustness
upgrade the values are LIVE-MEASURED for c22 & c11 (from the smoke) and the
other three keep their baseline-spec §5 totals scaled ~4x for the reasoning-
model correction (``scaled-pending-measurement``), to be overwritten by their
pilots' measured means.
"""

from __future__ import annotations

import pytest

from whetstone.envs.registry import (
    ENV_NAMES,
    ESTIMATE_LIVE_MEASURED,
    ESTIMATE_SCALED_PENDING,
    TokenEstimate,
    env_spec,
)

#: The per-env (naive, ceiling, source) estimates after the round-2 upgrade.
_EXPECTED: dict[str, TokenEstimate] = {
    "c22": TokenEstimate(
        naive=950, ceiling=1700, estimate_source=ESTIMATE_LIVE_MEASURED
    ),
    "c11": TokenEstimate(
        naive=656, ceiling=907, estimate_source=ESTIMATE_LIVE_MEASURED
    ),
    "c19": TokenEstimate(
        naive=1120, ceiling=4600, estimate_source=ESTIMATE_SCALED_PENDING
    ),
    "c18": TokenEstimate(
        naive=1012, ceiling=2600, estimate_source=ESTIMATE_SCALED_PENDING
    ),
    "c23": TokenEstimate(
        naive=560, ceiling=1680, estimate_source=ESTIMATE_SCALED_PENDING
    ),
}


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_every_env_has_committed_token_estimate(env_name: str) -> None:
    estimate = env_spec(env_name).token_estimate
    assert estimate == _EXPECTED[env_name]
    # Both probes are distinguished by every spec: naive is the cheaper probe.
    assert estimate.naive < estimate.ceiling
    assert estimate.naive > 0


@pytest.mark.parametrize("env_name", ["c22", "c11"])
def test_measured_envs_are_marked_live_measured(env_name: str) -> None:
    assert (
        env_spec(env_name).token_estimate.estimate_source
        == ESTIMATE_LIVE_MEASURED
    )


@pytest.mark.parametrize("env_name", ["c19", "c18", "c23"])
def test_unmeasured_envs_are_scaled_pending(env_name: str) -> None:
    assert (
        env_spec(env_name).token_estimate.estimate_source
        == ESTIMATE_SCALED_PENDING
    )


def test_token_estimates_cover_exactly_the_five_envs() -> None:
    assert set(_EXPECTED) == set(ENV_NAMES)
