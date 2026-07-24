"""The ``whetstone-validate`` runner: provider routes + resumable cell driver.

This package delivers the validation runner described in
``reports/validation-plan.md``:

* :mod:`whetstone.runner.routes` -- the provider route registry (canonical
  OpenRouter task/proposer routes + the four anthropic-messages plan lanes as
  alternates) with sane transport policies.
* :mod:`whetstone.runner.execution_mode` -- postgres / docker-postgres /
  in-process execution-mode detection (recorded per cell).
* :mod:`whetstone.runner.eval_run` -- the shared split-evaluation entry that
  wires the existing stage-03 driver + internal-eval loop + Result Store.
* :mod:`whetstone.runner.optimizers` -- brief-documented hyperparameters scaled
  to pool sizes + the internal-split proposal/measure loop.
* :mod:`whetstone.runner.statistics` -- the cheap paired bootstrap CI.
* :mod:`whetstone.runner.ledger` -- the ``cells.jsonl`` / ``spend.jsonl``
  ledgers (exact schema) and resumability.
* :mod:`whetstone.runner.budget` -- the reserve + per-cell stop-loss guards and
  the OpenRouter credits snapshot.
* :mod:`whetstone.runner.pilot` -- the checklist-B pilot.
* :mod:`whetstone.runner.cell` -- one full validation cell.
* :mod:`whetstone.runner.cli` -- the ``whetstone-validate`` CLI entry point.
"""

from __future__ import annotations

from whetstone.runner.budget import (
    RESERVE_USD,
    BudgetGuard,
    CreditsSnapshot,
    ReserveError,
    StopLossError,
)
from whetstone.runner.cell import CellConfig, CellOutcome, run_cell
from whetstone.runner.eval_run import SplitEvaluation, evaluate_split
from whetstone.runner.execution_mode import (
    ExecutionMode,
    ExecutionModeDecision,
    detect_execution_mode,
)
from whetstone.runner.ledger import CellRecord, Ledger, SpendRecord
from whetstone.runner.optimizers import (
    OPTIMIZERS,
    OptimizeResult,
    run_optimize,
    scaled_hyperparameters,
)
from whetstone.runner.pilot import PilotReport, run_pilot
from whetstone.runner.routes import (
    LANE_NAMES,
    PLAN_LANES,
    ProviderRoute,
    route_for,
)
from whetstone.runner.statistics import BootstrapCI, bootstrap_delta_ci

__all__ = [
    "LANE_NAMES",
    "OPTIMIZERS",
    "PLAN_LANES",
    "RESERVE_USD",
    "BootstrapCI",
    "BudgetGuard",
    "CellConfig",
    "CellOutcome",
    "CellRecord",
    "CreditsSnapshot",
    "ExecutionMode",
    "ExecutionModeDecision",
    "Ledger",
    "OptimizeResult",
    "PilotReport",
    "ProviderRoute",
    "ReserveError",
    "SpendRecord",
    "SplitEvaluation",
    "StopLossError",
    "bootstrap_delta_ci",
    "detect_execution_mode",
    "evaluate_split",
    "route_for",
    "run_cell",
    "run_optimize",
    "run_pilot",
    "scaled_hyperparameters",
]
