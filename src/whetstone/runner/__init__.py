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
from whetstone.runner.cell import (
    CellConfig,
    CellError,
    CellOutcome,
    bind_cell_launch,
    prepare_cell_launch,
    run_cell,
)
from whetstone.runner.events import (
    EVENTS_SCHEMA,
    EventStream,
    EventUnit,
    RunEvent,
)
from whetstone.runner.ledger import (
    CELLS_SCHEMA,
    OFFICIAL_ANCHOR_SCHEMA,
    SPEND_SCHEMA,
    CellRecord,
    Ledger,
    OfficialAnchorRecord,
    SpendRecord,
)
from whetstone.runner.optimization_run import (
    HarnessRunController,
    OptimizationRunControl,
    RunControlError,
)
from whetstone.runner.refinalize import RefinalizeOutcome, refinalize_cell
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
from whetstone.runner.viewer_projection import (
    VIEWER_GENERATION_ROW_SCHEMA,
    VIEWER_PROJECTION_SCHEMA,
    ViewerCellProjection,
    ViewerGenerationRow,
    build_viewer_cell_projection,
)

__all__ = [
    "CANONICAL_PROPOSER_MODEL",
    "CANONICAL_TASK_MODEL",
    "CELLS_SCHEMA",
    "DEFAULT_EXPECTED_CELL_USD",
    "EVENTS_SCHEMA",
    "LANE_NAMES",
    "OFFICIAL_ANCHOR_SCHEMA",
    "PLAN_LANES",
    "RESERVE_USD",
    "SPEND_SCHEMA",
    "STOP_LOSS_MULTIPLIER",
    "VIEWER_GENERATION_ROW_SCHEMA",
    "VIEWER_PROJECTION_SCHEMA",
    "BudgetGuard",
    "CellConfig",
    "CellError",
    "CellOutcome",
    "CellRecord",
    "CreditsSnapshot",
    "EventStream",
    "EventUnit",
    "HarnessRunController",
    "Ledger",
    "OfficialAnchorRecord",
    "OptimizationRunControl",
    "PlanLane",
    "ProviderRoute",
    "RefinalizeOutcome",
    "ReserveError",
    "RunControlError",
    "RunEvent",
    "SpendRecord",
    "StopLossError",
    "ViewerCellProjection",
    "ViewerGenerationRow",
    "bind_cell_launch",
    "build_viewer_cell_projection",
    "completeness_for_env",
    "credits_from_payload",
    "openrouter_credits_fetcher",
    "prepare_cell_launch",
    "refinalize_cell",
    "route_for",
    "run_cell",
    "task_model_for_env",
]
