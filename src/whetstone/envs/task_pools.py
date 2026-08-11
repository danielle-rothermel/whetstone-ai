from __future__ import annotations

from whetstone.experiment.task_selection import (
    TaskRoleSelection,
    TaskSplitManifest,
    TaskSplitManifestError,
    TaskSplitRole,
    TaskSplitRoles,
)

ENV_TO_POOL: dict[str, str] = {"d1": "d1", "ed1": "ed1"}


def _pool_key_for_env(env: str) -> str:
    pool_key = ENV_TO_POOL.get(env)
    if pool_key is not None:
        return pool_key
    if env == "ed1m":
        raise TaskSplitManifestError(
            "task-selection manifests do not apply to ed1m: "
            "manifest pools contain HumanEval task ids, while ed1m "
            "uses behavioral-mutant ids"
        )
    raise TaskSplitManifestError(
        "task-selection manifests apply only to "
        f"{sorted(ENV_TO_POOL)}; got env {env!r}"
    )


def roles_for_env(manifest: TaskSplitManifest, env: str) -> TaskSplitRoles:
    """Resolve the role arrays applicable to ``env``."""
    pool_key = _pool_key_for_env(env)
    if pool_key not in manifest.pools:
        raise TaskSplitManifestError(
            f"manifest has no pool {pool_key!r} for env {env!r}; "
            f"pools present: {sorted(manifest.pools)}"
        )
    return manifest.pool_roles(pool_key)


def select_role_for_env(
    manifest: TaskSplitManifest,
    *,
    env: str,
    role: TaskSplitRole,
) -> TaskRoleSelection:
    """Resolve and record one ordered role for ``env``."""
    return manifest.select_role(
        pool_key=_pool_key_for_env(env),
        role=role,
    )


def select_lowest_historical_pass_rate_for_env(
    manifest: TaskSplitManifest,
    *,
    env: str,
    role: TaskSplitRole,
    count: int,
    excluded_task_ids: tuple[str, ...] = (),
) -> TaskRoleSelection:
    """Select lowest-rate tasks for ``env`` with stable ID tie-breaking."""
    return manifest.select_lowest_historical_pass_rate(
        pool_key=_pool_key_for_env(env),
        role=role,
        count=count,
        excluded_task_ids=excluded_task_ids,
    )


__all__ = [
    "ENV_TO_POOL",
    "roles_for_env",
    "select_lowest_historical_pass_rate_for_env",
    "select_role_for_env",
]
