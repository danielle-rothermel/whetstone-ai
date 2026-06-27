from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from dr_dspy import humaneval_dbos_flow as flow
from dr_dspy.dbos_runtime import EvalDbosConfig
from dr_dspy.eval_repair import RepairPlan


def _config() -> EvalDbosConfig:
    return EvalDbosConfig(
        database_url="postgresql://x",
        dbos_system_database_url="postgresql://y",
        generation_concurrency=1,
        scoring_concurrency=1,
    )


def test_run_submit_jobs_invokes_backend_in_order() -> None:
    backend = MagicMock()
    backend.insert_prediction_jobs.return_value = 2
    jobs = ["j1", "j2"]
    flow.run_submit_jobs(
        backend,
        config=_config(),
        experiment_name="exp",
        seed=0,
        sample_count=1,
        metadata={},
        jobs=jobs,
        score_timeout=10.0,
    )
    backend.create_schema.assert_called_once_with("postgresql://x")
    backend.upsert_experiment.assert_called_once()
    backend.insert_prediction_jobs.assert_called_once_with(
        "postgresql://x", jobs
    )
    backend.configure_runtime.assert_called_once()
    backend.enqueue_generation_jobs.assert_called_once()


def test_run_status_command_skips_table_when_empty() -> None:
    backend = MagicMock()
    backend.fetch_status_counts.return_value = []
    flow.run_status_command(
        backend, database_url="postgresql://x", experiment_name=None
    )
    backend.status_counts_table.assert_not_called()


def test_run_status_command_prints_table_when_rows() -> None:
    backend = MagicMock()
    backend.fetch_status_counts.return_value = [{"count": 1}]
    flow.run_status_command(
        backend, database_url="postgresql://x", experiment_name="exp"
    )
    backend.status_counts_table.assert_called_once()


def test_run_repair_command_dry_run_does_not_apply() -> None:
    backend = MagicMock()
    backend.build_repair_plan.return_value = RepairPlan()
    flow.run_repair_command(
        backend,
        config=_config(),
        experiment_name="exp",
        generation_limit=10,
        scoring_limit=10,
        score_timeout=10.0,
        apply=False,
    )
    backend.build_repair_plan.assert_called_once()
    backend.apply_repair.assert_not_called()


def test_run_enqueue_scores_marks_queued() -> None:
    backend = MagicMock()
    backend.fetch_scoreable_prediction_ids.return_value = ["p1", "p2"]
    flow.run_enqueue_scores_command(
        backend,
        config=_config(),
        experiment_name="exp",
        limit=100,
        timeout=10.0,
    )
    backend.enqueue_score_jobs.assert_called_once()
    backend.mark_scoring_queued.assert_called_once_with(
        "postgresql://x", ["p1", "p2"]
    )


def test_run_analyze_command_writes_csv(tmp_path: Path) -> None:
    backend = MagicMock()
    backend.fetch_analysis_records.return_value = []
    backend.summarize_analysis_records.return_value = []
    flow.run_analyze_command(
        backend,
        database_url="postgresql://x",
        experiment_name="exp",
        csv_path=tmp_path / "out.csv",
        markdown=False,
    )
    backend.analysis_table.assert_called_once()
    backend.write_analysis_csv.assert_called_once()
