from __future__ import annotations

from whetstone.experiment.task_selection.manifest import TaskSplitManifest
from whetstone.experiment.task_selection.roles import (
    TaskRoleSelection,
    TaskSplitManifestError,
    TaskSplitRole,
    TaskSplitRoles,
)


def roles_for_pool(
    manifest: TaskSplitManifest,
    pool_key: str,
) -> TaskSplitRoles:
    """Resolve the role arrays applicable to one manifest pool."""
    if pool_key not in manifest.pools:
        raise TaskSplitManifestError(
            f"manifest has no pool {pool_key!r}; "
            f"pools present: {sorted(manifest.pools)}"
        )
    return manifest.pool_roles(pool_key)


def select_role_for_pool(
    manifest: TaskSplitManifest,
    *,
    pool_key: str,
    role: TaskSplitRole,
) -> TaskRoleSelection:
    """Resolve and record one ordered role for one manifest pool."""
    return manifest.select_role(pool_key=pool_key, role=role)


def select_lowest_historical_pass_rate_for_pool(
    manifest: TaskSplitManifest,
    *,
    pool_key: str,
    role: TaskSplitRole,
    count: int,
    excluded_task_ids: tuple[str, ...] = (),
) -> TaskRoleSelection:
    """Select lowest-rate tasks for one pool with stable ID tie-breaking."""
    return manifest.select_lowest_historical_pass_rate(
        pool_key=pool_key,
        role=role,
        count=count,
        excluded_task_ids=excluded_task_ids,
    )


__all__ = [
    "roles_for_pool",
    "select_lowest_historical_pass_rate_for_pool",
    "select_role_for_pool",
]
