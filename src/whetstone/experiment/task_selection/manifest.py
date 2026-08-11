from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from dr_serialize import StrictJsonDecodeError, decode_strict_json_bytes
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from whetstone.experiment.task_selection.roles import (
    TaskRoleSelection,
    TaskRoleSelectionMethod,
    TaskSplitManifestError,
    TaskSplitRole,
    TaskSplitRoles,
)

TASK_SELECTION_SCHEMA = "whetstone.run.task_selection/v1"


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

    def pool_roles(self, pool_key: str) -> TaskSplitRoles:
        """Resolve the role arrays for one persisted pool key."""
        pool = self.pools.get(pool_key)
        if pool is None:
            raise TaskSplitManifestError(
                f"manifest has no pool {pool_key!r}; "
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
        self, *, pool_key: str, role: TaskSplitRole
    ) -> TaskRoleSelection:
        """Resolve and record one ordered role without folding role meaning."""
        roles = self.pool_roles(pool_key)
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
        pool_key: str,
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
        roles = self.pool_roles(pool_key)
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
    """Read and validate one persisted manifest from disk."""
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise TaskSplitManifestError(
            f"cannot read task-selection manifest {path}: {exc}"
        ) from exc
    return parse_task_split_manifest(payload)


__all__ = [
    "TASK_SELECTION_SCHEMA",
    "TaskSplitManifest",
    "load_task_split_manifest",
    "parse_task_split_manifest",
]
