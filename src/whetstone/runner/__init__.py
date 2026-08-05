"""The validation runner: budget guards, provider routes, run telemetry."""

from __future__ import annotations

from whetstone.runner.budget import (
    DEFAULT_EXPECTED_CELL_USD,
    RESERVE_USD,
    STOP_LOSS_MULTIPLIER,
    BudgetGuard,
    CreditsSnapshot,
    ReserveError,
    StopLossError,
    credits_from_payload,
    openrouter_credits_fetcher,
)
from whetstone.runner.events import (
    EVENTS_SCHEMA,
    EventStream,
    EventUnit,
    RunEvent,
)
from whetstone.runner.routes import (
    CANONICAL_PROPOSER_MODEL,
    CANONICAL_TASK_MODEL,
    LANE_NAMES,
    PLAN_LANES,
    PlanLane,
    ProviderRoute,
    completeness_for_env,
    route_for,
    task_model_for_env,
)

__all__ = [
    "CANONICAL_PROPOSER_MODEL",
    "CANONICAL_TASK_MODEL",
    "DEFAULT_EXPECTED_CELL_USD",
    "EVENTS_SCHEMA",
    "LANE_NAMES",
    "PLAN_LANES",
    "RESERVE_USD",
    "STOP_LOSS_MULTIPLIER",
    "BudgetGuard",
    "CreditsSnapshot",
    "EventStream",
    "EventUnit",
    "PlanLane",
    "ProviderRoute",
    "ReserveError",
    "RunEvent",
    "StopLossError",
    "completeness_for_env",
    "credits_from_payload",
    "openrouter_credits_fetcher",
    "route_for",
    "task_model_for_env",
]
