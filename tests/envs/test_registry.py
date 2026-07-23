"""Registry-level checks for the bound env specs (no live calls).

Covers the per-env token estimates the pilot's token-sanity check falls back to
when ``--spec-estimate-tokens`` is not passed. After the round-3 update ALL
five envs are LIVE-MEASURED from their pilots' measured per-call means; c18's
ceiling (2448) is the measurement taken BEFORE the verdict-extraction scoring
fix (the fix changes scoring, not emitted token counts).
"""

from __future__ import annotations

import pytest

from whetstone.envs.registry import (
    ENV_NAMES,
    ESTIMATE_INHERITED_PENDING,
    ESTIMATE_LIVE_MEASURED,
    TokenEstimate,
    env_spec,
)

#: The per-env (naive, ceiling, source) estimates. The five base envs are
#: live-measured; c22h inherits c22's means pending its own pilot.
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


#: The base envs whose estimates are live-measured pilot means. c22h is the
#: sole variant: it inherits c22's means pending its own pilot measurement.
_LIVE_MEASURED_ENVS = tuple(n for n in ENV_NAMES if n != "c22h")


@pytest.mark.parametrize("env_name", _LIVE_MEASURED_ENVS)
def test_all_base_envs_are_marked_live_measured(env_name: str) -> None:
    # Every base env's estimate is a live-measured pilot mean.
    assert (
        env_spec(env_name).token_estimate.estimate_source
        == ESTIMATE_LIVE_MEASURED
    )


def test_c22h_estimate_is_inherited_pending_its_own_pilot() -> None:
    # c22h seeds from c22's live-measured means, flagged for overwrite by its
    # own pilot -- distinct provenance from a live-measured base env.
    est = env_spec("c22h").token_estimate
    assert est.estimate_source == ESTIMATE_INHERITED_PENDING
    assert (est.naive, est.ceiling) == (2526, 3046)


def test_token_estimates_cover_exactly_the_bound_envs() -> None:
    assert set(_EXPECTED) == set(ENV_NAMES)
