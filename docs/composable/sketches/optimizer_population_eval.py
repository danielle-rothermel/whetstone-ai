"""Consumer sketch 2: optimizer population evaluation.

Design-validation artifact for the dr-platform facade (platform.md).
Not executable. Models the existing prototype of this shape,
whetstone's `optimization/copro.py` (COPRO today, GEPA/RL later):
an outer loop written as ordinary code that manufactures a population
of graph specs (genomes, per graph_runner.md), evaluates them through
the platform, and reads scores back.

The loop the facade must make first-class:
    submit population under one operation key
    -> await batch outcomes
    -> read scores.

What copro hand-rolled today that collapses into facade calls here:
`wait_for_generation_runs` (bespoke polling over domain tables) ->
`await_operation`; ad-hoc frame loading -> a versioned projection.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel
from sqlalchemy import create_engine

from dr_platform import (
    ProjectionSpec,
    await_operation,
    load_projection_frame,
    rebuild_projection,
    submit_batch,
)

# --- app-side domain (stays in whetstone) ----------------------------------
# Spec manufacture stays on the app's frozen identity contract:
# spec_builder + records/hashing own prediction_id / fair_order_key.

from whetstone.platform.spec_builder import (  # noqa: E402 -- illustrative
    build_specs_for_candidates,  # imaginary thin wrapper over today's API
)
from whetstone.records import PredictionSpecRecord  # noqa: E402


class SpecItem(BaseModel):
    """Adapter: PredictionSpecRecord -> SubmittableItem. The protocol
    maps onto the app's frozen axes; nothing is re-hashed."""

    spec: PredictionSpecRecord

    @property
    def item_id(self) -> str:
        return self.spec.prediction_id

    @property
    def order_key(self) -> str:
        return self.spec.fair_order_key

    @property
    def group_key(self) -> str:
        return self.spec.experiment_name


def insert_domain_rows(engine: Any, specs: Sequence[Any]) -> None:
    """App-side: experiment + prediction-spec rows in the app's outcome
    schema (idempotent inserts). The library never writes these."""
    raise NotImplementedError("domain code, not part of the sketch")


def start_generation_workflow(item_id: str) -> str:
    """App-side enqueue target (today:
    start_prediction_graph_workflow with its deterministic
    workflow id). item_id is prediction_id."""
    raise NotImplementedError("domain code, not part of the sketch")


class CandidateScoreRow(BaseModel):
    candidate_id: str
    prediction_id: str
    generation_terminal: bool
    score_passed: bool | None


def candidate_score_rows(connection: Any) -> list[CandidateScoreRow]:
    """App-side query joining prediction specs, generation runs, and
    score attempts (today's analysis frame, as a versioned projection).
    """
    raise NotImplementedError("domain code, not part of the sketch")


SCORES_PROJECTION = ProjectionSpec(
    name="copro_candidate_scores",
    version="v1",
    row_model=CandidateScoreRow,
    build=candidate_score_rows,
)


def propose(best: Any, depth: int) -> list[Any]:
    """Ordinary (durable, app-side) optimizer code — mutates specs as
    data. Never enters the library."""
    raise NotImplementedError("domain code, not part of the sketch")


def select_best(frame: Any, candidates: Sequence[Any]) -> Any:
    raise NotImplementedError("domain code, not part of the sketch")


# --- the loop ---------------------------------------------------------------


def run_optimizer(
    database_url: str,
    *,
    run_id: str,
    experiment_name: str,
    max_depth: int,
) -> Any:
    engine = create_engine(database_url)
    best: Any = None

    for depth in range(max_depth):
        candidates = propose(best, depth)
        specs = build_specs_for_candidates(
            candidates,
            experiment_name=experiment_name,
        )
        insert_domain_rows(engine, specs)

        # One operation key per population: re-running a crashed depth
        # reconciles instead of double-submitting (stable item ids).
        operation_key = f"copro-{run_id}-d{depth}"
        submit_batch(
            engine,
            operation_key=operation_key,
            items=[SpecItem(spec=spec) for spec in specs],
            enqueue=start_generation_workflow,
        )
        await_operation(
            engine,
            operation_key=operation_key,
            poll_interval_seconds=15.0,
            timeout_seconds=3600.0,
        )

        rebuild_projection(engine, SCORES_PROJECTION)
        frame = load_projection_frame(
            engine,
            SCORES_PROJECTION,
            group_key=experiment_name,
        )
        best = select_best(frame, candidates)

    return best
