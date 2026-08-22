"""GEPA platform integration trio (Postgres + DBOS + dr-platform)."""

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
    succeeded_stage_counts,
)

pytestmark = pytest.mark.integration


def _assert_gepa_search_evidence(result) -> None:
    assert result.step_results
    for step_ref in result.step_results:
        evidence = step_ref.record.search_evidence
        assert evidence, "every completed GEPA step must carry search_evidence"
        for item in evidence:
            assert item.eval_result_ref is not None


def test_inline_platform_gepa_submit_to_result(
    pg_engine: Engine,
    clean_pg: str,
    tmp_path,
) -> None:
    """INLINE submit → terminal GEPA result with resolvable search evidence."""
    store_path = tmp_path / "integration-gepa-inline.sqlite"
    with open_sqlite(str(store_path)) as store:
        context = bootstrap_platform_runtime(
            store=store,
            pg_engine=pg_engine,
            clean_pg=clean_pg,
            now=NOW,
            dispatch_mode=EvalDispatchMode.INLINE,
            optimizer="gepa",
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
                run_key=context.run_key,
            )
            terminal_result_ref = await_run_completion(
                run_key=context.run_key,
                pg_engine=pg_engine,
                registration=context.registration,
                registry=context.registry,
                now=NOW,
            )
            result = load_terminal_optim_result(context, terminal_result_ref)
            _assert_gepa_search_evidence(result)

            counts = succeeded_stage_counts(pg_engine, work_item_id)
            assert counts.get(STAGE_OPTIM_STEP, 0) >= 1
            assert_stage_coverage(
                pg_engine,
                work_item_id,
                {
                    STAGE_OPTIM_STEP: counts[STAGE_OPTIM_STEP],
                    STAGE_EVAL_ROW: 0,
                    STAGE_EVAL_FANIN: 0,
                },
            )
        finally:
            shutdown_platform_runtime(context)


def test_platform_gepa_deferral_fanout_fanin_through_admission(
    pg_engine: Engine,
    clean_pg: str,
    tmp_path,
) -> None:
    """PLATFORM GEPA deferral → eval_row fan-out → same-step resume → result."""
    store_path = tmp_path / "integration-gepa-platform.sqlite"
    with open_sqlite(str(store_path)) as store:
        context = bootstrap_platform_runtime(
            store=store,
            pg_engine=pg_engine,
            clean_pg=clean_pg,
            now=NOW,
            dispatch_mode=EvalDispatchMode.PLATFORM,
            optimizer="gepa",
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
                run_key=context.run_key,
            )
            terminal_result_ref = await_run_completion(
                run_key=context.run_key,
                pg_engine=pg_engine,
                registration=context.registration,
                registry=context.registry,
                now=NOW,
            )
            result = load_terminal_optim_result(context, terminal_result_ref)
            _assert_gepa_search_evidence(result)

            counts = succeeded_stage_counts(pg_engine, work_item_id)
            expected_eval_row_count = counts.get(STAGE_EVAL_ROW, 0)
            assert expected_eval_row_count >= 1
            assert counts.get(STAGE_EVAL_FANIN, 0) >= 1
            assert counts.get(STAGE_OPTIM_STEP, 0) >= 2
            if counts.get(STAGE_EVAL_FANIN, 0) == 1:
                assert_fanin_barrier_predecessors(
                    pg_engine,
                    work_item_id,
                    fanin_stage_index=expected_eval_row_count + 1,
                    deferral_origin_stage_index=0,
                    expected_eval_row_count=expected_eval_row_count,
                )
        finally:
            shutdown_platform_runtime(context)


def test_platform_gepa_deferral_survives_fanin_retry(
    pg_engine: Engine,
    clean_pg: str,
    tmp_path,
) -> None:
    """PLATFORM GEPA deferral completes after fan-in replay."""
    store_path = tmp_path / "integration-gepa-preemptible-retry.sqlite"
    with open_sqlite(str(store_path)) as store:
        context = bootstrap_platform_runtime(
            store=store,
            pg_engine=pg_engine,
            clean_pg=clean_pg,
            now=NOW,
            dispatch_mode=EvalDispatchMode.PLATFORM,
            optimizer="gepa",
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
                run_key=context.run_key,
            )
            terminal_result_ref = await_run_completion(
                run_key=context.run_key,
                pg_engine=pg_engine,
                registration=context.registration,
                registry=context.registry,
                now=NOW,
            )
            result = load_terminal_optim_result(context, terminal_result_ref)
            _assert_gepa_search_evidence(result)
        finally:
            shutdown_platform_runtime(context)
