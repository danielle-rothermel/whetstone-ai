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
