from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import UNIQUE, StrEnum, verify
from pathlib import Path
from typing import Literal

from dr_serialize import StrictJsonDecodeError, decode_strict_json_bytes
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

TASK_SELECTION_SCHEMA = "whetstone.run.task_selection/v1"
_ENV_TO_POOL: dict[str, str] = {"d1": "d1", "ed1": "ed1"}


class TaskSplitManifestError(ValueError):
    """A typed failure parsing or applying a task-selection manifest."""


@verify(UNIQUE)
class TaskSplitRole(StrEnum):
    """One explicit role from a persisted task-selection manifest."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@verify(UNIQUE)
class TaskRoleSelectionMethod(StrEnum):
    """How an ordered selection was derived from its manifest role."""

    FULL_ROLE = "full_role"
    LOWEST_HISTORICAL_PASS_RATE = "lowest_historical_pass_rate"


class TaskRoleSelection(BaseModel):
    """The exact persisted manifest-derived selection for one evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_content_hash: str
    pool_key: str
    role: TaskSplitRole
    task_ids: tuple[str, ...]
    selection_method: TaskRoleSelectionMethod = (
        TaskRoleSelectionMethod.FULL_ROLE
    )
    source_role_count: int | None = None
    eligible_pool_count: int | None = None
    excluded_task_ids: tuple[str, ...] = ()
    historical_pass_rates: tuple[float, ...] = ()

    @model_validator(mode="after")
    def _validate_selection(self) -> TaskRoleSelection:
        if self.historical_pass_rates and len(
            self.historical_pass_rates
        ) != len(self.task_ids):
            raise ValueError(
                "historical pass rates must align with selected task IDs"
            )
        if self.source_role_count is not None and self.source_role_count < len(
            self.task_ids
        ):
            raise ValueError(
                "source role count cannot be smaller than selection"
            )
        if (
            self.eligible_pool_count is not None
            and self.eligible_pool_count < len(self.task_ids)
        ):
            raise ValueError(
                "eligible pool count cannot be smaller than selection"
            )
        return self


class _HistoricalSelectionMetadata(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    historical_pass_rates: dict[str, float]


class _PoolRoles(BaseModel):
    """One persisted pool's role arrays."""

    model_config = ConfigDict(extra="allow", frozen=True)

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def model_post_init(self, _context: object) -> None:
        seen: dict[str, str] = {}
        for role in ("train", "val", "test"):
            ids = getattr(self, role)
            if len(set(ids)) != len(ids):
                raise ValueError(f"role {role!r} has duplicate task ids")
            # Roles are DISJOINT: an id in more than one role would leak a
            # train/val task into the held-out test split (train and val both
            # fold into the internal split, test is the official one), which
            # silently invalidates every official measurement.
            for task_id in ids:
                other = seen.get(task_id)
                if other is not None:
                    raise ValueError(
                        f"task id {task_id!r} appears in both role {other!r} "
                        f"and role {role!r}; manifest roles must be disjoint"
                    )
                seen[task_id] = role


class TaskSplitManifest(BaseModel):
    """Validated persisted task-selection manifest."""

    model_config = ConfigDict(extra="allow", frozen=True)

    schema_name: Literal["whetstone.run.task_selection/v1"] = Field(
        alias="schema"
    )
    pools: dict[str, _PoolRoles]
    selection: _HistoricalSelectionMetadata | None = None
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

    def select_role(
        self, *, env: str, role: TaskSplitRole
    ) -> TaskRoleSelection:
        """Resolve and record one ordered role without folding role meaning."""
        roles = self.for_env(env)
        return TaskRoleSelection(
            manifest_content_hash=self.content_hash,
            pool_key=roles.pool_key,
            role=role,
            task_ids=roles.ids_for(role),
            source_role_count=len(roles.ids_for(role)),
            eligible_pool_count=len(roles.ids_for(role)),
        )

    def select_lowest_historical_pass_rate(
        self,
        *,
        env: str,
        role: TaskSplitRole,
        count: int,
        excluded_task_ids: tuple[str, ...] = (),
    ) -> TaskRoleSelection:
        """Select lowest-rate tasks with stable ID tie-breaking."""
        if count < 1:
            raise TaskSplitManifestError(
                "selected task count must be positive"
            )
        if len(set(excluded_task_ids)) != len(excluded_task_ids):
            raise TaskSplitManifestError("excluded task IDs must be unique")
        roles = self.for_env(env)
        unknown_exclusions = tuple(
            task_id
            for task_id in excluded_task_ids
            if task_id not in roles.all_role_ids()
        )
        if unknown_exclusions:
            raise TaskSplitManifestError(
                "excluded task IDs are absent from the manifest pool: "
                f"{unknown_exclusions}"
            )
        metadata = self.selection
        if metadata is None:
            raise TaskSplitManifestError(
                "manifest has no historical pass-rate metadata"
            )
        excluded = frozenset(excluded_task_ids)
        source_ids = roles.ids_for(role)
        eligible_ids = tuple(
            task_id for task_id in source_ids if task_id not in excluded
        )
        if count > len(eligible_ids):
            raise TaskSplitManifestError(
                f"requested {count} tasks from {len(eligible_ids)} eligible "
                f"tasks in role {role.value!r}"
            )
        rates = metadata.historical_pass_rates
        missing_rates = tuple(
            task_id for task_id in eligible_ids if task_id not in rates
        )
        if missing_rates:
            raise TaskSplitManifestError(
                "historical pass rates are missing for eligible tasks: "
                f"{missing_rates}"
            )
        invalid_rates = tuple(
            task_id
            for task_id in eligible_ids
            if not math.isfinite(rates[task_id])
        )
        if invalid_rates:
            raise TaskSplitManifestError(
                f"historical pass rates are non-finite: {invalid_rates}"
            )
        selected_ids = tuple(
            sorted(
                eligible_ids, key=lambda task_id: (rates[task_id], task_id)
            )[:count]
        )
        return TaskRoleSelection(
            manifest_content_hash=self.content_hash,
            pool_key=roles.pool_key,
            role=role,
            task_ids=selected_ids,
            selection_method=(
                TaskRoleSelectionMethod.LOWEST_HISTORICAL_PASS_RATE
            ),
            source_role_count=len(source_ids),
            eligible_pool_count=len(eligible_ids),
            excluded_task_ids=excluded_task_ids,
            historical_pass_rates=tuple(
                rates[task_id] for task_id in selected_ids
            ),
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

    def ids_for(self, role: TaskSplitRole) -> tuple[str, ...]:
        """Return one role exactly as ordered in the manifest."""
        if role is TaskSplitRole.TRAIN:
            return self.train_ids
        if role is TaskSplitRole.VALIDATION:
            return self.val_ids
        return self.test_ids


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
            raw_bytes = (
                payload if isinstance(payload, bytes) else payload.encode()
            )
            raw = decode_strict_json_bytes(
                raw_bytes,
                max_bytes=len(raw_bytes),
                max_depth=len(raw_bytes),
            )
        elif isinstance(payload, Mapping):
            raw = dict(payload)
        else:
            raise TaskSplitManifestError(
                "task-selection manifest must be JSON or a mapping"
            )
    except (UnicodeEncodeError, StrictJsonDecodeError) as exc:
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


def load_task_split_manifest(path: Path) -> TaskSplitManifest:
    """Read and validate one environment-owned manifest from disk."""
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise TaskSplitManifestError(
            f"cannot read task-selection manifest {path}: {exc}"
        ) from exc
    return parse_task_split_manifest(payload)


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
    "TaskRoleSelection",
    "TaskRoleSelectionMethod",
    "TaskSplitManifest",
    "TaskSplitManifestError",
    "TaskSplitRole",
    "TaskSplitRoles",
    "load_task_split_manifest",
    "parse_task_split_manifest",
    "resolve_manifest_split",
]
