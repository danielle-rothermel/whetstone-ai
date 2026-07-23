"""Config-identity tests for the provider route registry (no live calls)."""

from __future__ import annotations

import pytest

from whetstone.runner.routes import (
    CANONICAL_PROPOSER_MODEL,
    CANONICAL_TASK_MODEL,
    LANE_NAMES,
    OPENAI_BASE_URL,
    OPENAI_KEY_ENV,
    OPENROUTER_BASE_URL,
    OPENROUTER_KEY_ENV,
    PLAN_LANES,
    REASONING_EFFORT_CHOICES,
    canonical_proposer_route,
    canonical_task_route,
    lane_route,
    openai_direct_route,
    reasoning_effort_for,
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


# --- Task 19: the openai-direct lane -----------------------------------------


def test_openai_direct_route_construction() -> None:
    # (Task 19.1) The openai lane: OpenAI's own API, OPENAI_API_KEY, chat-
    # completions, DISTINCT config identity from the openrouter route.
    route = openai_direct_route(model="gpt-5.4-nano", temperature=0.0)
    assert route.lane == "openai"
    assert route.model == "gpt-5.4-nano"
    assert route.key_env == OPENAI_KEY_ENV == "OPENAI_API_KEY"
    assert route.transport_policy.base_url == OPENAI_BASE_URL
    assert route.transport_policy.native_retry_count == 0
    # Same stall/wall policy shape as openrouter (600s cap, ~90s idle).
    assert route.transport_policy.timeout_seconds == 600.0
    assert route.transport_policy.idle_timeout_seconds == 90.0


def test_openai_lane_config_identity_distinct_from_openrouter() -> None:
    # (Task 19.1) Same model, different provider -> DISTINCT config id, so
    # the lane folds into route/graph identity and records distinctly.
    oa = openai_direct_route(model="gpt-5.4-nano", temperature=0.0)
    orr = route_for(
        "openrouter", role="task", task_model="gpt-5.4-nano", temperature=0.0
    )
    assert oa.call_config.identity_hash != orr.call_config.identity_hash
    assert oa.identity_summary()["lane"] == "openai"
    assert oa.identity_summary()["base_url"] == OPENAI_BASE_URL


def test_route_for_openai_lane_dispatch() -> None:
    # (Task 19.1) route_for("openai", ...) selects the openai-direct route by
    # role, honoring --task-model / --proposer-model.
    task = route_for(
        "openai", role="task", task_model="gpt-5.4-nano", temperature=0.5
    )
    assert task.lane == "openai" and task.model == "gpt-5.4-nano"
    prop = route_for(
        "openai", role="proposer", proposer_model="gpt-5.4-nano"
    )
    assert prop.lane == "openai" and prop.role == "proposer"


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


def test_c18_completeness_matrix_default_is_skip_2pct() -> None:
    from whetstone.runner.routes import completeness_for_env

    # c18's matrix default declares a SKIP-with-2%-tolerance policy.
    assert completeness_for_env("c18") == ("skip", 0.02)


def test_ed1_completeness_matrix_default_is_skip_15pct() -> None:
    from whetstone.runner.routes import completeness_for_env

    # ed1's matrix default is a SKIP-with-15%-tolerance policy: its per-row
    # failures are genuine stochastic model behavior at tight budgets (empty
    # completions + entry-point name loss), measured at ~10.4% on the anchor,
    # so a bounded 15% tolerance certifies while keeping skipped rows explicit.
    assert completeness_for_env("ed1") == ("skip", 0.15)


def test_unlisted_env_completeness_default_is_strict_propagate() -> None:
    from whetstone.runner.routes import completeness_for_env

    # Every env not in the matrix keeps the strict, untolerant default.
    for env in ("c11", "c19", "c22", "c23"):
        assert completeness_for_env(env) == ("propagate", 0.0)


# --- Task 21.1: --reasoning-effort dial (identity fold + lane mappings) --


def test_reasoning_effort_for_maps_choices() -> None:
    from dr_providers import ReasoningEffort

    assert reasoning_effort_for(None) is None  # absent -> provider default
    assert reasoning_effort_for("none") is ReasoningEffort.NONE
    assert reasoning_effort_for("low") is ReasoningEffort.LOW
    assert reasoning_effort_for("medium") is ReasoningEffort.MEDIUM
    assert reasoning_effort_for("high") is ReasoningEffort.HIGH
    assert set(REASONING_EFFORT_CHOICES) == {"none", "low", "medium", "high"}


def test_reasoning_absent_is_byte_identical() -> None:
    # (Task 21.1) Absent flag -> control UNSET -> byte-identical config to the
    # historical no-flag route (openrouter + openai).
    orr = route_for("openrouter", role="task", task_model="gpt-5-nano")
    orr_none = route_for(
        "openrouter", role="task", task_model="gpt-5-nano",
        reasoning=reasoning_effort_for(None),
    )
    assert orr.call_config.identity_hash == orr_none.call_config.identity_hash
    oa = openai_direct_route(model="gpt-5.4-nano")
    oa_none = openai_direct_route(
        model="gpt-5.4-nano", reasoning=reasoning_effort_for(None)
    )
    assert oa.call_config.identity_hash == oa_none.call_config.identity_hash


def test_reasoning_effort_folds_into_config_identity_openrouter() -> None:
    # (Task 21.1, c23-era rule) OUTPUT-AFFECTING: each effort is a DISTINCT
    # config identity on the openrouter lane (reasoning object shape).
    base = route_for("openrouter", role="task", task_model="gpt-5-nano")
    low = route_for(
        "openrouter", role="task", task_model="gpt-5-nano",
        reasoning=reasoning_effort_for("low"),
    )
    none = route_for(
        "openrouter", role="task", task_model="gpt-5-nano",
        reasoning=reasoning_effort_for("none"),
    )
    high = route_for(
        "openrouter", role="task", task_model="gpt-5-nano",
        reasoning=reasoning_effort_for("high"),
    )
    hashes = {
        base.call_config.identity_hash, low.call_config.identity_hash,
        none.call_config.identity_hash, high.call_config.identity_hash,
    }
    assert len(hashes) == 4  # all distinct


def test_reasoning_effort_folds_into_config_identity_openai() -> None:
    # (Task 21.1) OUTPUT-AFFECTING on the openai lane (reasoning_effort field).
    base = openai_direct_route(model="gpt-5.4-nano")
    low = openai_direct_route(
        model="gpt-5.4-nano", reasoning=reasoning_effort_for("low")
    )
    none = openai_direct_route(
        model="gpt-5.4-nano", reasoning=reasoning_effort_for("none")
    )
    hashes = {
        base.call_config.identity_hash, low.call_config.identity_hash,
        none.call_config.identity_hash,
    }
    assert len(hashes) == 3


def test_reasoning_effort_serializes_per_lane_shape() -> None:
    # (Task 21.1) openrouter -> reasoning object; openai -> reasoning_effort
    # (dr-providers picks the shape from the config's reasoning_shape).
    from whetstone.runner.routes import openai_direct_route

    orr = route_for(
        "openrouter", role="task", task_model="gpt-5-nano",
        reasoning=reasoning_effort_for("low"),
    ).call_config
    oa = openai_direct_route(
        model="gpt-5.4-nano", reasoning=reasoning_effort_for("low")
    ).call_config
    assert (
        orr.definition.constraints.reasoning_shape.value == "reasoning_object"
    )
    assert oa.definition.constraints.reasoning_shape.value == "effort_field"


# --- Task 25: verified reasoning + temperature on-wire serialization -------


def _payload(route):
    from dr_providers import (
        MessageRole,
        PromptMessage,
        ProviderCallRequest,
        Transcript,
    )
    from dr_providers.request import build_payload

    req = ProviderCallRequest(
        config=route.call_config,
        transcript=Transcript(
            messages=(PromptMessage(role=MessageRole.USER, content="x"),)
        ),
    )
    return build_payload(req)


def test_temperature_zero_vs_none_distinct_and_on_wire() -> None:
    # (Task 25.1) --temperature is OUTPUT-AFFECTING: temp 0.0 (the legacy task
    # default) is a DISTINCT config identity from an explicit 0.5 or unset.
    t0 = route_for("openrouter", role="task", task_model="gpt-5-nano",
                   temperature=0.0)
    t5 = route_for("openrouter", role="task", task_model="gpt-5-nano",
                   temperature=0.5)
    tn = route_for("openrouter", role="task", task_model="gpt-5-nano",
                   temperature=None)
    ids = {
        t0.call_config.identity_hash, t5.call_config.identity_hash,
        tn.call_config.identity_hash,
    }
    assert len(ids) == 3
    # temp 0.0 serializes on the wire (not dropped).
    assert _payload(t0)["temperature"] == 0.0


def test_openrouter_reasoning_wire_shape_verified() -> None:
    # (Task 25.2, VERIFIED live) openrouter serializes reasoning as the
    # {reasoning: {effort: <v>}} object; 'none' disables (deepseek honored it,
    # reasoning_tokens -> 0). temp0 co-exists with it (no 400 on openrouter).
    none = route_for(
        "openrouter", role="task", task_model="deepseek/deepseek-v4-flash",
        temperature=0.0, reasoning=reasoning_effort_for("none"),
    )
    low = route_for(
        "openrouter", role="task", task_model="deepseek/deepseek-v4-flash",
        temperature=0.0, reasoning=reasoning_effort_for("low"),
    )
    assert _payload(none)["reasoning"] == {"effort": "none"}
    assert _payload(none)["temperature"] == 0.0
    assert _payload(low)["reasoning"] == {"effort": "low"}


def test_openai_reasoning_wire_shape_verified() -> None:
    # (Task 25.2, VERIFIED live) openai serializes reasoning as the flat
    # reasoning_effort field. temp0+none is ACCEPTED live; temp0+low 400s
    # (OpenAI requires temp=1 with a non-default effort) -- a provider
    # constraint, NOT our mapping. This locks the shape.
    from whetstone.runner.routes import openai_direct_route

    none = openai_direct_route(
        model="gpt-5.4-nano", temperature=0.0,
        reasoning=reasoning_effort_for("none"),
    )
    low = openai_direct_route(
        model="gpt-5.4-nano", temperature=0.0,
        reasoning=reasoning_effort_for("low"),
    )
    assert _payload(none)["reasoning_effort"] == "none"
    assert _payload(none)["temperature"] == 0.0
    assert _payload(low)["reasoning_effort"] == "low"
