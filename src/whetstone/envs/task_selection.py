"""Typed, environment-neutral task-selection manifest contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

TASK_SELECTION_SCHEMA = "whetstone.run.task_selection/v1"
_ENV_TO_POOL: dict[str, str] = {"d1": "d1", "ed1": "ed1"}


class TaskSplitManifestError(ValueError):
    """A typed failure parsing or applying a task-selection manifest."""


class _PoolRoles(BaseModel):
    """One persisted pool's role arrays."""

    model_config = ConfigDict(extra="allow", frozen=True)

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def model_post_init(self, _context: object) -> None:
        for role in ("train", "val", "test"):
            ids = getattr(self, role)
            if len(set(ids)) != len(ids):
                raise ValueError(f"role {role!r} has duplicate task ids")


class TaskSplitManifest(BaseModel):
    """Validated persisted task-selection manifest."""

    model_config = ConfigDict(extra="allow", frozen=True)

    schema_name: Literal["whetstone.run.task_selection/v1"] = Field(
        alias="schema"
    )
    pools: dict[str, _PoolRoles]
    content_hash: str = Field(exclude=True)

    def for_env(self, env: str) -> TaskSplitRoles:
        """Resolve the role arrays applicable to ``env``."""
        pool_key = _ENV_TO_POOL.get(env)
        if pool_key is None:
            if env == "ed1m":
                raise TaskSplitManifestError(
                    "task-selection manifests do not apply to ed1m: "
                    "manifest pools contain HumanEval task ids, while ed1m "
                    "uses behavioral-mutant ids"
                )
            raise TaskSplitManifestError(
                "task-selection manifests apply only to "
                f"{sorted(_ENV_TO_POOL)}; got env {env!r}"
            )
        pool = self.pools.get(pool_key)
        if pool is None:
            raise TaskSplitManifestError(
                f"manifest has no pool {pool_key!r} for env {env!r}; "
                f"pools present: {sorted(self.pools)}"
            )
        return TaskSplitRoles(
            pool_key=pool_key,
            train_ids=pool.train,
            val_ids=pool.val,
            test_ids=pool.test,
            content_hash=self.content_hash,
        )


@dataclass(frozen=True, slots=True)
class TaskSplitRoles:
    """One pool's ordered train, validation, and test role sets."""

    pool_key: str
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    content_hash: str

    @property
    def internal_ids(self) -> tuple[str, ...]:
        return self.train_ids + self.val_ids

    @property
    def official_ids(self) -> tuple[str, ...]:
        return self.test_ids

    def all_role_ids(self) -> frozenset[str]:
        return frozenset(self.train_ids + self.val_ids + self.test_ids)


@dataclass(frozen=True, slots=True)
class ResolvedSplit[T]:
    """A manifest-resolved internal and official task partition."""

    internal: tuple[T, ...]
    official: tuple[T, ...]
    manifest_tag: str
    official_capped: str | None


def parse_task_split_manifest(
    payload: object,
) -> TaskSplitManifest:
    """Parse and validate manifest JSON or an already-decoded mapping."""
    try:
        if isinstance(payload, bytes | str):
            raw = json.loads(payload)
        elif isinstance(payload, Mapping):
            raw = dict(payload)
        else:
            raise TaskSplitManifestError(
                "task-selection manifest must be JSON or a mapping"
            )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TaskSplitManifestError(
            "task-selection manifest is not valid JSON"
        ) from exc
    if not isinstance(raw, dict):
        raise TaskSplitManifestError(
            "task-selection manifest must be a JSON object"
        )
    canonical = json.dumps(
        raw,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    content_hash = hashlib.sha256(canonical).hexdigest()
    try:
        return TaskSplitManifest.model_validate(
            {**raw, "content_hash": content_hash}
        )
    except ValidationError as exc:
        raise TaskSplitManifestError(
            f"invalid task-selection manifest: {exc}"
        ) from exc


def resolve_manifest_split[T](
    *,
    roles: TaskSplitRoles,
    items: Sequence[T],
    id_of: Callable[[T], str],
    official_n: int | None = None,
) -> ResolvedSplit[T]:
    """Resolve manifest membership in manifest order, refusing unknown ids."""
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
    "TASK_SELECTION_SCHEMA",
    "ResolvedSplit",
    "TaskSplitManifest",
    "TaskSplitManifestError",
    "TaskSplitRoles",
    "parse_task_split_manifest",
    "resolve_manifest_split",
]
