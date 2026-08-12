from __future__ import annotations

from dr_providers import ReasoningEffort

from whetstone.runner.routes import (
    ProviderRoute,
    canonical_task_route,
    openai_direct_route,
)

TOKEN_LIMIT = 4096


def _provider_route(
    *, lane: str, model: str, reasoning: ReasoningEffort | None
) -> ProviderRoute:
    route = openai_direct_route if lane == "openai" else canonical_task_route
    return route(
        model=model,
        temperature=None,
        reasoning=reasoning,
        token_limit=TOKEN_LIMIT,
        max_attempts=1,
    )


def baseline_provider_routes() -> tuple[ProviderRoute, ...]:
    """Return the four ordered, fixed behavior-matrix provider routes."""

    return (
        _provider_route(
            lane="openai",
            model="gpt-5.4-nano",
            reasoning=ReasoningEffort.NONE,
        ),
        _provider_route(
            lane="openrouter",
            model="deepseek/deepseek-v4-flash",
            reasoning=ReasoningEffort.NONE,
        ),
        _provider_route(
            lane="openrouter",
            model="qwen/qwen3-coder-flash",
            reasoning=None,
        ),
        _provider_route(
            lane="openrouter",
            model="google/gemini-3.1-flash-lite",
            reasoning=ReasoningEffort.NONE,
        ),
    )


__all__ = [
    "TOKEN_LIMIT",
    "baseline_provider_routes",
]
