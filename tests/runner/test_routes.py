"""Config-identity tests for the provider route registry (no live calls)."""

from __future__ import annotations

import pytest

from whetstone.runner.routes import (
    CANONICAL_PROPOSER_MODEL,
    CANONICAL_TASK_MODEL,
    LANE_NAMES,
    OPENROUTER_BASE_URL,
    OPENROUTER_KEY_ENV,
    PLAN_LANES,
    canonical_proposer_route,
    canonical_task_route,
    lane_route,
    route_for,
)


def test_canonical_task_route_openrouter_gpt5_nano() -> None:
    route = canonical_task_route()
    assert route.model == CANONICAL_TASK_MODEL == "openai/gpt-5-nano"
    assert route.lane == "openrouter"
    assert route.key_env == OPENROUTER_KEY_ENV == "OPENROUTER_API_KEY"
    assert route.transport_policy.native_retry_count == 0
    # Absolute cap 600s (accommodate reasoning-model streams), idle ~90s.
    assert route.transport_policy.timeout_seconds == 600.0
    assert route.transport_policy.idle_timeout_seconds == 90.0


def test_canonical_proposer_route_distinct_identity() -> None:
    task = canonical_task_route()
    proposer = canonical_proposer_route()
    assert proposer.model == CANONICAL_PROPOSER_MODEL == "openai/gpt-5.4-nano"
    # Proposer route identity is distinct from the task route (never collides
    # with an encoder/decoder graph route hash).
    assert (
        proposer.call_config.identity_hash
        != task.call_config.identity_hash
    )


@pytest.mark.parametrize("lane", LANE_NAMES)
def test_plan_lane_routes_use_window_starts_data(lane: str) -> None:
    route = lane_route(lane)
    spec = PLAN_LANES[lane]
    assert route.lane == lane
    assert route.model == spec.model
    assert route.transport_policy.base_url == spec.base_url
    assert route.key_env == spec.key_env
    # Anthropic-messages protocol lanes: sane transport policy.
    assert route.transport_policy.native_retry_count == 0
    assert route.transport_policy.timeout_seconds == 600.0
    assert route.transport_policy.idle_timeout_seconds == 90.0


def test_all_four_plan_lanes_present() -> None:
    assert set(LANE_NAMES) == {"kimi", "glm", "minimax", "stepfun"}
    assert set(PLAN_LANES) == set(LANE_NAMES)


def test_plan_lane_key_envs_match_window_starts() -> None:
    assert PLAN_LANES["kimi"].key_env == "KIMI_CODE_API_KEY"
    assert PLAN_LANES["glm"].key_env == "ZAI_API_KEY"
    assert PLAN_LANES["minimax"].key_env == "MINIMAX_API_KEY"
    assert PLAN_LANES["stepfun"].key_env == "STEPFUN_API_KEY"


def test_route_for_selects_lane_and_role() -> None:
    assert route_for("openrouter", role="task").model == CANONICAL_TASK_MODEL
    assert (
        route_for("openrouter", role="proposer").model
        == CANONICAL_PROPOSER_MODEL
    )
    assert route_for("kimi").lane == "kimi"


def test_identity_summary_carries_no_secret() -> None:
    summary = canonical_task_route().identity_summary()
    # Only the env-var NAME is present, never a key value.
    assert summary["key_env"] == "OPENROUTER_API_KEY"
    assert "call_config_hash" in summary
    assert "execution_policy_hash" in summary
    text = str(summary)
    assert "Bearer" not in text


def test_unknown_lane_rejected() -> None:
    with pytest.raises(ValueError, match="unknown plan lane"):
        lane_route("nope")


def test_canonical_routes_pin_openrouter_base_url() -> None:
    # Regression (live round-1 blocker): the canonical routes MUST carry a
    # base_url or every OpenRouter call fails pre-flight with missing_base_url.
    assert OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
    for route in (canonical_task_route(), canonical_proposer_route()):
        assert route.transport_policy.base_url == OPENROUTER_BASE_URL


def test_every_producible_route_has_well_formed_base_url() -> None:
    # Regression: every route the registry can produce -- canonical task,
    # canonical proposer, and every plan lane, via route_for for both roles --
    # has a non-None, well-formed (https://) base_url.
    routes = [canonical_task_route(), canonical_proposer_route()]
    for lane in ("openrouter", *LANE_NAMES):
        for role in ("task", "proposer"):
            routes.append(route_for(lane, role=role))
    for route in routes:
        base_url = route.transport_policy.base_url
        assert base_url is not None, route.identity_summary()
        assert base_url.startswith("https://"), base_url
        # Well-formed: a host after the scheme, no trailing whitespace.
        assert base_url == base_url.strip()
        assert len(base_url) > len("https://")


def test_temperature_folds_into_config_identity() -> None:
    t0 = canonical_task_route(temperature=0.0)
    t1 = canonical_task_route(temperature=1.0)
    assert (
        t0.call_config.identity_hash != t1.call_config.identity_hash
    )
