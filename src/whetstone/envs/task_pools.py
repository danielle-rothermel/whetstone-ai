from __future__ import annotations

from whetstone.envs.code_comp.constants import CODE_COMP_ENV_NAME
from whetstone.envs.code_comp.registry import CodeCompMode
from whetstone.experiment.task_selection import (
    TaskRoleSelection,
    TaskSplitManifest,
    TaskSplitManifestError,
    TaskSplitRole,
    TaskSplitRoles,
)

_CODE_COMP_POOL_BY_MODE: dict[CodeCompMode, str] = {
    CodeCompMode.DIRECT: "direct",
    CodeCompMode.ENCDEC: "encdec",
}


def pool_key_for(env: str, mode: CodeCompMode | None = None) -> str:
    """Resolve the manifest pool key for one env and code_comp mode."""
    if env != CODE_COMP_ENV_NAME:
        raise TaskSplitManifestError(
            "task-selection manifests apply only to "
            f"{CODE_COMP_ENV_NAME!r}; got env {env!r}"
        )
    if mode is CodeCompMode.ENCDEC_MUTANT:
        raise TaskSplitManifestError(
            "task-selection manifests do not apply to encdec_mutant: "
            "manifest pools contain HumanEval task ids, while encdec_mutant "
            "uses behavioral-mutant ids"
        )
    if mode is None:
        mode = CodeCompMode.ENCDEC
    pool_key = _CODE_COMP_POOL_BY_MODE.get(mode)
    if pool_key is None:
        raise TaskSplitManifestError(
            f"unsupported code_comp mode {mode!r} for task-selection manifests"
        )
    return pool_key


def roles_for_env(
    manifest: TaskSplitManifest,
    env: str,
    mode: CodeCompMode | None = None,
) -> TaskSplitRoles:
    """Resolve the role arrays applicable to ``env``."""
    pool_key = pool_key_for(env, mode)
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
    mode: CodeCompMode | None = None,
    role: TaskSplitRole,
) -> TaskRoleSelection:
    """Resolve and record one ordered role for ``env``."""
    return manifest.select_role(
        pool_key=pool_key_for(env, mode),
        role=role,
    )


def select_lowest_historical_pass_rate_for_env(
    manifest: TaskSplitManifest,
    *,
    env: str,
    mode: CodeCompMode | None = None,
    role: TaskSplitRole,
    count: int,
    excluded_task_ids: tuple[str, ...] = (),
) -> TaskRoleSelection:
    """Select lowest-rate tasks for ``env`` with stable ID tie-breaking."""
    return manifest.select_lowest_historical_pass_rate(
        pool_key=pool_key_for(env, mode),
        role=role,
        count=count,
        excluded_task_ids=excluded_task_ids,
    )


__all__ = [
    "pool_key_for",
    "roles_for_env",
    "select_lowest_historical_pass_rate_for_env",
    "select_role_for_env",
]
