from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt

from whetstone.core.identity import ImmutableJsonObject
from whetstone.optim.contracts import BudgetDelta
from whetstone.optim.gepa.control import GepaControl
from whetstone.optim.gepa.engine import (
    GepaDetailedResult,
    GepaEngineAdapter,
    _project_result,
    _validate_adapter_authorities,
)
from whetstone.optim.gepa.source import verify_installed_gepa_source

GEPA_STATE_KEY = "gepa_checkpoint"
GEPA_STEP_ENGINE_SCHEMA_VERSION = 1


class GepaStepCheckpoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: StrictInt = GEPA_STEP_ENGINE_SCHEMA_VERSION
    metric_calls_consumed: StrictInt = 0
    terminal: StrictBool = False

    @property
    def budget_delta(self) -> BudgetDelta:
        return BudgetDelta(
            consumed=ImmutableJsonObject({"metric_calls": 1}),
        )


def load_gepa_checkpoint(request: Any) -> GepaStepCheckpoint | None:
    if request.step_index == 0:
        return GepaStepCheckpoint()
    pools = dict(request.pools)
    raw = pools.get(GEPA_STATE_KEY)
    if raw is None:
        return GepaStepCheckpoint()
    return GepaStepCheckpoint.model_validate(raw)


def run_one_gepa_iteration[DataInst](
    *,
    control: GepaControl,
    seed_candidate: dict[str, str],
    trainset: tuple[DataInst, ...],
    valset: tuple[DataInst, ...] | None,
    adapter: GepaEngineAdapter,
    checkpoint: GepaStepCheckpoint | None,
) -> tuple[GepaDetailedResult, GepaStepCheckpoint]:
    current = checkpoint or GepaStepCheckpoint()
    next_budget = min(
        control.resolved_max_metric_calls,
        current.metric_calls_consumed + 1,
    )
    if next_budget <= current.metric_calls_consumed:
        raise ValueError("GEPA step budget is already exhausted")

    verify_installed_gepa_source()
    from gepa import optimize

    _validate_adapter_authorities(control=control, adapter=adapter)
    adapter.reset_effect_ordinal()
    ordered_seed = dict(seed_candidate)
    kwargs = control.upstream_kwargs()
    kwargs["max_metric_calls"] = next_budget
    result = optimize(
        seed_candidate=ordered_seed,
        trainset=list(trainset),
        valset=None if valset is None else list(valset),
        adapter=cast(Any, adapter),
        reflection_lm=None,
        custom_candidate_proposer=None,
        logger=None,
        callbacks=None,
        **kwargs,
    )
    detailed = _project_result(result, control=control)
    consumed = (
        detailed.total_metric_calls
        if detailed.total_metric_calls is not None
        else next_budget
    )
    terminal = consumed >= control.resolved_max_metric_calls
    updated = GepaStepCheckpoint(
        metric_calls_consumed=consumed,
        terminal=terminal,
    )
    return detailed, updated


__all__ = [
    "GEPA_STATE_KEY",
    "GepaStepCheckpoint",
    "load_gepa_checkpoint",
    "run_one_gepa_iteration",
]
