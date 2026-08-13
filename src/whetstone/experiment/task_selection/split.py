from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from whetstone.experiment.task_selection.roles import (
    TaskSplitManifestError,
    TaskSplitRoles,
)


@dataclass(frozen=True, slots=True)
class ResolvedSplit[T]:
    internal: tuple[T, ...]
    official: tuple[T, ...]
    manifest_tag: str
    official_capped: str | None


def resolve_manifest_split[T](
    *,
    roles: TaskSplitRoles,
    items: Sequence[T],
    id_of: Callable[[T], str],
    official_n: int | None = None,
) -> ResolvedSplit[T]:
    by_id = {str(id_of(item)): item for item in items}
    missing = sorted(roles.all_role_ids() - frozenset(by_id))
    if missing:
        raise TaskSplitManifestError(
            f"manifest pool {roles.pool_key!r} references "
            f"{len(missing)} task id(s) absent from the loaded task pool "
            f"(unknown ids: {missing})"
        )
    if official_n is not None and official_n < 1:
        raise TaskSplitManifestError(
            f"official_n must be at least 1; got {official_n}"
        )
    internal = tuple(by_id[item_id] for item_id in roles.internal_ids)
    official = tuple(by_id[item_id] for item_id in roles.official_ids)
    capped: str | None = None
    if official_n is not None and official_n < len(official):
        capped = (
            f"official_n={official_n} caps the manifest test split "
            f"({len(official)} tasks) to its first {official_n}"
        )
        official = official[:official_n]
    elif official_n is not None and official_n > len(official):
        capped = (
            f"official_n={official_n} exceeds the manifest test split size "
            f"({len(official)}); using all {len(official)} test tasks"
        )
    return ResolvedSplit(
        internal=internal,
        official=official,
        manifest_tag=f"tsm:{roles.content_hash[:16]}.{roles.pool_key}",
        official_capped=capped,
    )


__all__ = [
    "ResolvedSplit",
    "resolve_manifest_split",
]
