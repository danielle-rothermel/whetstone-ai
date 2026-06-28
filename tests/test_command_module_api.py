from __future__ import annotations

from types import ModuleType

from dr_dspy import humaneval_direct_dbos as direct
from dr_dspy import humaneval_encdec_dbos as encdec

REMOVED_SHARED_API_NAMES = (
    "QueueSelection",
    "EvalDbosConfig",
    "DbPoolConfig",
    "OpenFileLimitResult",
    "DB_POOL_AUTO",
    "DB_POOLS",
    "connect_db",
    "close_db_connection_pools",
    "WorkerMonitorConfig",
    "WorkerQueueSnapshot",
    "open_file_limit_line",
    "open_file_limit_style",
    "resolve_database_url",
    "build_eval_dbos_config",
    "raise_open_file_limit",
    "configure_worker_db_connection_pools",
    "create_eval_schema",
    "configure_dbos_runtime",
    "queue_names_for_selection",
    "configure_pooled_worker_runtime",
    "enqueue_generation_jobs",
    "enqueue_score_job",
    "enqueue_score_jobs",
    "start_worker_monitor",
    "resolve_worker_log_path",
    "configure_worker_file_logging",
    "operator_log",
    "emit_worker_detail_log",
    "prediction_context_from_job",
    "emit_prediction_log_event",
)


def _assert_removed_shared_api_names(module: ModuleType) -> None:
    leaked_names = [
        name for name in REMOVED_SHARED_API_NAMES if hasattr(module, name)
    ]
    assert leaked_names == []


def test_direct_module_does_not_reexport_shared_runtime_api() -> None:
    _assert_removed_shared_api_names(direct)


def test_encdec_module_does_not_reexport_shared_runtime_api() -> None:
    _assert_removed_shared_api_names(encdec)


def test_batch_fanout_functions_are_not_dbos_steps() -> None:
    batch_functions = (
        direct.submit_batch_step,
        direct.enqueue_scores_batch_step,
        direct.repair_batch_step,
        encdec.submit_batch_step,
        encdec.enqueue_scores_batch_step,
        encdec.repair_batch_step,
    )

    decorated_functions = [
        function.__name__
        for function in batch_functions
        if hasattr(function, "dbos_function_name")
    ]

    assert decorated_functions == []


def test_direct_constraint_migration_compares_text_arrays() -> None:
    migration_sql = "\n".join(direct.PREDICTION_CONSTRAINT_MIGRATION_SQL)

    assert "SELECT a.attname::text" in migration_sql
