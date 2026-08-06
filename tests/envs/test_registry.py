from __future__ import annotations

import pytest

from whetstone.envs.registry import (
    ENV_NAMES,
    ESTIMATE_INHERITED_PENDING,
    ESTIMATE_LIVE_MEASURED,
    TokenEstimate,
    env_spec,
)

_EXPECTED: dict[str, TokenEstimate] = {
    "c22": TokenEstimate(
        naive=2526, ceiling=3046, estimate_source=ESTIMATE_LIVE_MEASURED
    ),
    "c22h": TokenEstimate(
        naive=2526,
        ceiling=3046,
        estimate_source=ESTIMATE_INHERITED_PENDING,
    ),
    "c11": TokenEstimate(
        naive=1735, ceiling=1831, estimate_source=ESTIMATE_LIVE_MEASURED
    ),
    "c19": TokenEstimate(
        naive=4377, ceiling=5009, estimate_source=ESTIMATE_LIVE_MEASURED
    ),
    "c18": TokenEstimate(
        naive=1306, ceiling=2448, estimate_source=ESTIMATE_LIVE_MEASURED
    ),
    "c18h": TokenEstimate(
        naive=1959,
        ceiling=3672,
        estimate_source=ESTIMATE_INHERITED_PENDING,
    ),
    "c23": TokenEstimate(
        naive=5468, ceiling=4953, estimate_source=ESTIMATE_LIVE_MEASURED
    ),
}


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_every_env_has_committed_token_estimate(env_name: str) -> None:
    estimate = env_spec(env_name).token_estimate
    assert estimate == _EXPECTED[env_name]
    assert estimate.naive > 0
    assert estimate.ceiling > 0


_INHERITED_VARIANT_ENVS = frozenset({"c22h", "c18h"})
_LIVE_MEASURED_ENVS = tuple(
    n for n in ENV_NAMES if n not in _INHERITED_VARIANT_ENVS
)


@pytest.mark.parametrize("env_name", _LIVE_MEASURED_ENVS)
def test_all_base_envs_are_marked_live_measured(env_name: str) -> None:
    assert (
        env_spec(env_name).token_estimate.estimate_source
        == ESTIMATE_LIVE_MEASURED
    )


def test_c22h_estimate_is_inherited_pending_its_own_pilot() -> None:
    est = env_spec("c22h").token_estimate
    assert est.estimate_source == ESTIMATE_INHERITED_PENDING
    assert (est.naive, est.ceiling) == (2526, 3046)


def test_token_estimates_cover_exactly_the_bound_envs() -> None:
    assert set(_EXPECTED) == set(ENV_NAMES)
