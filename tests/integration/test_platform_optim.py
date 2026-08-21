"""Tier 2 platform integration tests (Postgres + DBOS + dr-platform).

Run locally:
    uv sync --extra platform
    uv run pytest -m integration tests/integration/
"""

from __future__ import annotations

import pytest
from sqlalchemy import Engine

from whetstone.coordination.eval_service import EvalDispatchMode
from dr_store.sync import open_sqlite
from whetstone.platform.contracts import (
    STAGE_EVAL_FANIN,
    STAGE_EVAL_ROW,
    STAGE_OPTIM_STEP,
)

from .conftest import migrate_platform_schema
from .platform_helpers import (
    NOW,
    assert_fanin_barrier_predecessors,
    assert_stage_coverage,
    await_run_completion,
    bootstrap_platform_runtime,
    load_terminal_optim_result,
    lookup_work_item_id,
    run_until_quiescent,
    shutdown_platform_runtime,
)

pytestmark = pytest.mark.integration


def test_inline_platform_copro_submit_to_result(
    pg_engine: Engine,
    clean_pg: str,
    tmp_path,
) -> None:
    """INLINE submit → admission → DBOS optim_step → run completion."""
    store_path = tmp_path / "integration-inline.sqlite"
    with open_sqlite(str(store_path)) as store:
        context = bootstrap_platform_runtime(
            store=store,
            pg_engine=pg_engine,
            clean_pg=clean_pg,
            now=NOW,
            dispatch_mode=EvalDispatchMode.INLINE,
        )
        try:
            work_item_id = lookup_work_item_id(
                pg_engine,
                campaign_key=context.campaign_key,
                work_key=context.work_key,
            )
            run_until_quiescent(
                pg_engine=pg_engine,
                registry=context.registry,
                registration=context.registration,
                now=NOW,
                work_item_id=work_item_id,
            )
            terminal_result_ref = await_run_completion(
                run_key=context.run_key,
                pg_engine=pg_engine,
                registration=context.registration,
                registry=context.registry,
                now=NOW,
            )
            load_terminal_optim_result(context, terminal_result_ref)

            assert_stage_coverage(
                pg_engine,
                work_item_id,
                {
                    STAGE_OPTIM_STEP: 2,
                    STAGE_EVAL_ROW: 0,
                    STAGE_EVAL_FANIN: 0,
                },
            )
        finally:
            shutdown_platform_runtime(context)


def test_platform_deferral_fanout_fanin_through_admission(
    pg_engine: Engine,
    clean_pg: str,
    tmp_path,
) -> None:
    """PLATFORM deferral → eval_row fan-out → barrier fan-in → resume → result."""
    store_path = tmp_path / "integration-platform.sqlite"
    with open_sqlite(str(store_path)) as store:
        context = bootstrap_platform_runtime(
            store=store,
            pg_engine=pg_engine,
            clean_pg=clean_pg,
            now=NOW,
            dispatch_mode=EvalDispatchMode.PLATFORM,
            breadth=2,
            depth=1,
        )
        try:
            work_item_id = lookup_work_item_id(
                pg_engine,
                campaign_key=context.campaign_key,
                work_key=context.work_key,
            )
            run_until_quiescent(
                pg_engine=pg_engine,
                registry=context.registry,
                registration=context.registration,
                now=NOW,
                deadline_seconds=180,
                work_item_id=work_item_id,
            )
            terminal_result_ref = await_run_completion(
                run_key=context.run_key,
                pg_engine=pg_engine,
                registration=context.registration,
                registry=context.registry,
                now=NOW,
            )
            load_terminal_optim_result(context, terminal_result_ref)

            deferred_intent_count = 2
            internal_task_count = 2
            seed_count = 1
            expected_eval_row_count = (
                deferred_intent_count * internal_task_count * seed_count
            )
            assert_stage_coverage(
                pg_engine,
                work_item_id,
                {
                    STAGE_OPTIM_STEP: 2,
                    STAGE_EVAL_ROW: expected_eval_row_count,
                    STAGE_EVAL_FANIN: 1,
                },
            )
            fanin_stage_index = expected_eval_row_count + 1
            assert_fanin_barrier_predecessors(
                pg_engine,
                work_item_id,
                fanin_stage_index=fanin_stage_index,
                deferral_origin_stage_index=0,
                expected_eval_row_count=expected_eval_row_count,
            )
        finally:
            shutdown_platform_runtime(context)


@pytest.mark.integration
def test_platform_deferral_survives_fanin_retry(
    pg_engine: Engine,
    clean_pg: str,
    tmp_path,
) -> None:
    """PLATFORM deferral completes after fan-in replay (preemptible idempotency contract)."""
    store_path = tmp_path / "integration-preemptible-retry.sqlite"
    with open_sqlite(str(store_path)) as store:
        context = bootstrap_platform_runtime(
            store=store,
            pg_engine=pg_engine,
            clean_pg=clean_pg,
            now=NOW,
            dispatch_mode=EvalDispatchMode.PLATFORM,
            breadth=2,
            depth=1,
        )
        try:
            work_item_id = lookup_work_item_id(
                pg_engine,
                campaign_key=context.campaign_key,
                work_key=context.work_key,
            )
            run_until_quiescent(
                pg_engine=pg_engine,
                registry=context.registry,
                registration=context.registration,
                now=NOW,
                deadline_seconds=180,
                work_item_id=work_item_id,
            )
            terminal_result_ref = await_run_completion(
                run_key=context.run_key,
                pg_engine=pg_engine,
                registration=context.registration,
                registry=context.registry,
                now=NOW,
            )
            load_terminal_optim_result(context, terminal_result_ref)
        finally:
            shutdown_platform_runtime(context)


def test_episode_predecessor_read_scopes_to_the_second_deferral_episode(
    pg_engine: Engine,
) -> None:
    """Two episodes on one work item must not leak rows across the boundary.

    ``seed_double_deferral_episode`` lays down optim/eval/fan-in stages twice
    on a single work item. Whetstone's fan-in relies on the origin filter to
    read only the current episode's eval rows; without it the second fan-in
    would also see the first episode's rows.
    """
    from dr_platform import (
        list_episode_predecessor_outputs,
        list_predecessor_stage_outputs,
    )
    from dr_platform.testing import seed_double_deferral_episode

    migrate_platform_schema(pg_engine)
    with pg_engine.begin() as connection:
        work_item_id, first_origin, first_fanin, second_origin, second_fanin = (
            seed_double_deferral_episode(
                connection,
                eval_row_key=STAGE_EVAL_ROW,
                optim_step_key=STAGE_OPTIM_STEP,
                fanin_key=STAGE_EVAL_FANIN,
            )
        )

    first_episode = list_episode_predecessor_outputs(
        work_item_id,
        first_fanin,
        origin_stage_index=first_origin,
        stage_key=STAGE_EVAL_ROW,
        engine=pg_engine,
    )
    second_episode = list_episode_predecessor_outputs(
        work_item_id,
        second_fanin,
        origin_stage_index=second_origin,
        stage_key=STAGE_EVAL_ROW,
        engine=pg_engine,
    )

    assert tuple(row.stage_index for row in first_episode) == (1, 2)
    assert tuple(row.stage_index for row in second_episode) == (5, 6)
    assert tuple(row.output_reference for row in second_episode) == (
        "row:out:5",
        "row:out:6",
    )

    # Without the origin filter the second fan-in would see all four rows.
    unscoped = list_predecessor_stage_outputs(
        work_item_id,
        second_fanin,
        engine=pg_engine,
        stage_key=STAGE_EVAL_ROW,
    )
    assert tuple(row.stage_index for row in unscoped) == (1, 2, 5, 6)
