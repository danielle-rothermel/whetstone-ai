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

# 6d validation note: whetstone ships this loop for real —
# platform/submission.py wires the seed hook + enqueue target, and
# optimization/copro.py's evaluate_specs_queue awaits the operation.
from whetstone.platform.platform_db import PLATFORM_SCHEMA

# --- app-side domain (stays in whetstone) ----------------------------------
# Spec manufacture stays on the app's frozen identity contract:
# spec_builder + records/hashing own prediction_id / fair_order_key.

from whetstone.platform.spec_builder import (  # noqa: E402 -- illustrative
    build_specs_for_candidates,  # imaginary thin wrapper over today's API
)
from whetstone.records import PredictionSpecRecord  # noqa: E402


# (6d validation: no adapter class needed — PredictionSpecRecord
# itself satisfies SubmittableItem via item_id/order_key/group_key
# properties on the frozen axes; nothing is re-hashed.)


def insert_domain_rows(connection: Any, specs: Sequence[Any]) -> set[str]:
    """App-side seed hook: experiment + prediction-spec rows in the
    app's outcome schema, inside the library's registration
    transaction; returns newly-inserted ids (whetstone ships this as
    submission._seed_experiment_and_specs)."""
    raise NotImplementedError("domain code, not part of the sketch")


def start_generation_workflow(item_id: str) -> "EnqueueOutcome":
    """App-side enqueue target (whetstone ships this: queue_worker's
    enqueue_prediction_graph_workflow over dr_platform.dedup_enqueue).
    item_id is prediction_id."""
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

        # One operation key per population: re-running a crashed depth
        # reconciles instead of double-submitting (stable item ids).
        operation_key = f"copro-{run_id}-d{depth}"
        submit_batch(
            engine,
            operation_key=operation_key,
            group_key=experiment_name,
            items=list(specs),
            enqueue=start_generation_workflow,
            schema=PLATFORM_SCHEMA,
            seed=lambda connection, window: insert_domain_rows(
                connection, window
            ),
        )
        await_operation(
            engine,
            operation_key=operation_key,
            schema=PLATFORM_SCHEMA,
            poll_interval_seconds=15.0,
            timeout_seconds=3600.0,
        )

        rebuild_projection(engine, SCORES_PROJECTION, schema=PLATFORM_SCHEMA)
        frame = load_projection_frame(
            engine,
            SCORES_PROJECTION,
            schema=PLATFORM_SCHEMA,
            group_key=experiment_name,
        )
        best = select_best(frame, candidates)

    return best
