"""Route registry tests. Constructing a route makes no live call."""

from __future__ import annotations

import pytest
from dr_providers import ReasoningEffort

from whetstone.runner.routes import (
    CANONICAL_PROPOSER_MODEL,
    CANONICAL_TASK_MODEL,
    DEEPSEEK_TASK_MODEL,
    ENCDEC_DEFAULT_TASK_MODEL,
    LANE_NAMES,
    OPENAI_BASE_URL,
    OPENAI_KEY_ENV,
    OPENROUTER_BASE_URL,
    OPENROUTER_KEY_ENV,
    PLAN_LANES,
    REASONING_EFFORT_CHOICES,
    canonical_proposer_route,
    canonical_task_route,
    completeness_for_env,
    lane_route,
    openai_direct_route,
    reasoning_effort_for,
    route_for,
    task_model_for_env,
)


def test_canonical_models_are_the_pinned_slugs() -> None:
    assert CANONICAL_TASK_MODEL == "openai/gpt-5-nano"
    assert CANONICAL_PROPOSER_MODEL == "openai/gpt-5.4-nano"


def test_canonical_task_route_pins_openrouter_transport() -> None:
    route = canonical_task_route()

    assert route.role == "task"
    assert route.lane == "openrouter"
    assert route.model == CANONICAL_TASK_MODEL
    assert route.key_env == OPENROUTER_KEY_ENV
    assert route.transport_policy.base_url == OPENROUTER_BASE_URL
    # Whetstone owns every semantic retry, so native retries stay pinned off.
    assert route.transport_policy.native_retry_count == 0
    assert route.execution_policy.transport_policy == route.transport_policy


def test_proposer_route_identity_differs_from_task_route() -> None:
    task = canonical_task_route()
    proposer = canonical_proposer_route()

    assert proposer.role == "proposer"
    assert proposer.model == CANONICAL_PROPOSER_MODEL
    assert proposer.call_config.identity_hash != task.call_config.identity_hash


def test_openai_direct_route_is_a_distinct_provider_identity() -> None:
    direct = openai_direct_route()
    via_openrouter = canonical_task_route()

    assert direct.lane == "openai"
    assert direct.key_env == OPENAI_KEY_ENV
    assert direct.transport_policy.base_url == OPENAI_BASE_URL
    # Same model, same protocol shape, but the provider identity differs, so
    # the graph route identity differs too.
    assert direct.model == via_openrouter.model
    assert (
        direct.call_config.identity_hash
        != via_openrouter.call_config.identity_hash
    )


def test_task_model_override_changes_the_route_identity() -> None:
    canonical = canonical_task_route()
    deepseek = canonical_task_route(model=DEEPSEEK_TASK_MODEL)

    assert (
        deepseek.call_config.identity_hash
        != canonical.call_config.identity_hash
    )


def test_reasoning_effort_folds_into_the_config_identity() -> None:
    default = canonical_task_route()
    high = canonical_task_route(reasoning=ReasoningEffort.HIGH)
    low = canonical_task_route(reasoning=ReasoningEffort.LOW)

    assert high.call_config.identity_hash != default.call_config.identity_hash
    assert high.call_config.identity_hash != low.call_config.identity_hash


def test_task_routes_accept_exact_token_limits() -> None:
    openrouter = canonical_task_route(token_limit=2048)
    openai = openai_direct_route(token_limit=1024)

    assert openrouter.call_config.controls.token_limit == 2048
    assert openai.call_config.controls.token_limit == 1024


def test_absent_reasoning_leaves_the_provider_default() -> None:
    assert reasoning_effort_for(None) is None
    assert reasoning_effort_for("high") is ReasoningEffort.HIGH
    assert set(REASONING_EFFORT_CHOICES) == {"none", "low", "medium", "high"}


def test_unknown_reasoning_effort_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown reasoning effort"):
        reasoning_effort_for("extreme")


@pytest.mark.parametrize("lane", LANE_NAMES)
def test_every_plan_lane_builds_an_anthropic_route(lane: str) -> None:
    route = lane_route(lane)
    spec = PLAN_LANES[lane]

    assert route.lane == lane
    assert route.model == spec.model
    assert route.key_env == spec.key_env
    assert route.transport_policy.base_url == spec.base_url
    # The anthropic-messages definition requires an explicit token limit, so
    # a lane route that built without one would not have validated.
    assert route.call_config.controls.token_limit == 4096


def test_unknown_lane_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown plan lane"):
        lane_route("nonexistent")


def test_route_for_dispatches_by_lane_and_role() -> None:
    assert route_for("openrouter", role="task").model == CANONICAL_TASK_MODEL
    assert (
        route_for("openrouter", role="proposer").model
        == CANONICAL_PROPOSER_MODEL
    )
    assert route_for("openai", role="proposer").lane == "openai"
    assert route_for("kimi").lane == "kimi"


def test_route_for_applies_the_task_model_override() -> None:
    route = route_for("openrouter", role="task", task_model="vendor/other")

    assert route.model == "vendor/other"


def test_task_model_for_env_prefers_the_explicit_override() -> None:
    assert task_model_for_env("c18") == DEEPSEEK_TASK_MODEL
    assert task_model_for_env("ed1") == ENCDEC_DEFAULT_TASK_MODEL
    assert task_model_for_env("unlisted") == CANONICAL_TASK_MODEL
    assert task_model_for_env("c18", override="vendor/x") == "vendor/x"


def test_completeness_defaults_to_strict_propagate() -> None:
    assert completeness_for_env("unlisted") == ("propagate", 0.0)
    assert completeness_for_env("c18") == ("skip", 0.02)
    assert completeness_for_env("ed1") == ("skip", 0.15)


def test_identity_summary_carries_no_secret_material() -> None:
    summary = canonical_task_route().identity_summary()

    assert summary["key_env"] == OPENROUTER_KEY_ENV
    assert summary["native_retry_count"] == 0
    assert set(summary) == {
        "role",
        "lane",
        "model",
        "call_config_hash",
        "execution_policy_hash",
        "key_env",
        "base_url",
        "timeout_seconds",
        "native_retry_count",
    }
