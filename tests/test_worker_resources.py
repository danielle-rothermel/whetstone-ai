from __future__ import annotations

from dr_dspy.dbos_runtime import QueueSelection
from dr_dspy.worker_resources import (
    build_worker_resource_budget,
    http_pool_size,
    resolve_open_file_limit_request,
    scoring_subprocess_fd_budget,
)


def test_worker_resource_budget_accounts_for_active_queues() -> None:
    budget = build_worker_resource_budget(
        queue=QueueSelection.BOTH,
        generation_concurrency=64,
        scoring_concurrency=32,
        db_pool_max_size=104,
    )

    assert budget.http_max_connections == http_pool_size(
        queue=QueueSelection.BOTH,
        generation_concurrency=64,
    )
    assert budget.scoring_subprocess_fds == scoring_subprocess_fd_budget(
        queue=QueueSelection.BOTH,
        scoring_concurrency=32,
    )
    assert budget.estimated_open_files > 64 + 32 + 104


def test_auto_open_file_limit_uses_estimated_budget() -> None:
    budget = build_worker_resource_budget(
        queue=QueueSelection.GENERATION,
        generation_concurrency=8,
        scoring_concurrency=4,
        db_pool_max_size=16,
    )

    assert (
        resolve_open_file_limit_request("auto", budget=budget)
        == budget.estimated_open_files
    )
    assert resolve_open_file_limit_request("512", budget=budget) == 512
