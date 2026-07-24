"""Run task-selection manifest reader (``whetstone.run.task_selection/v1``).

Task 29: the runner consumes a run's task-selection manifest with TRUE
train/val/test role semantics -- distinct from the ``--task-filter`` seam
(:func:`whetstone.runner.task_screen.load_exclusion_ids`), which reads a flat
per-model ``excluded_task_ids`` list and drives an EXCLUSION-only, first-N
internal/official slice.

The manifest is a cross-model, three-way-split artifact:

    {
      "schema": "whetstone.run.task_selection/v1",
      "pools": {
        "d1":  {"train": [...], "val": [...], "test": [...], "dropped_*": ...},
        "ed1": {"train": [...], "val": [...], "test": [...], "dropped_*": ...}
      },
      ...
    }

Role mapping this reader exposes (see :meth:`TaskSplitManifest.for_env`):

* **internal split** (what the optimizer sees during proposal/scoring) =
  ``train + val`` (the manifest's train ids FOLLOWED BY its val ids). The
  internal machinery has NO val sub-split seam -- the enc-dec/direct builders
  carry only ``internal_eval`` + ``official`` (``held_out`` is never sampled;
  see ``whetstone/envs/sampling.py``). So the manifest's ``val`` is folded into
  the internal split alongside ``train`` rather than held separately. Ordering
  is identity-bearing, so the train-then-val order is fixed and documented.
* **official split** (baseline / ceiling / best certification) = ``test``
  EXACTLY, by MEMBERSHIP -- never a first-N slice of the kept pool.

The manifest's ``content_hash`` (over the whole file) folds into each split's
Task Set ``manifest_id`` so a manifest-driven cell is a DISTINCT, provenance-
bearing ``eval_config_hash`` -- exactly how ``d1``'s frozen ``input_arm`` folds
(``whetstone/envs/d1.py``). A cell built WITHOUT a manifest is byte-identical
to before (the fold is conditional).

Pool selection is by env: ``d1 -> pools.d1``; ``ed1 -> pools.ed1``. The
manifest carries HumanEval task ids and so applies ONLY to the HumanEval-task
envs (``ed1`` + ``d1``). ``ed1m`` runs on the behavioral-MUTANT pool whose
Instance ids are ``HumanEval/NN::family::sNN::nSITE`` (see
``whetstone/envs/ed1m_dataset.py``), which the manifest's bare
``HumanEval/NN`` role arrays can NEVER match -- so this reader REFUSES ``ed1m``
with a typed error rather than silently misfilter.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

TASK_SELECTION_SCHEMA = "whetstone.run.task_selection/v1"

#: Envs the manifest's HumanEval task-id role arrays legitimately apply to.
#: Maps the CLI env name to the manifest pool key. ``ed1m`` is DELIBERATELY
#: absent (its Instance ids are mutant ids, not HumanEval task ids).
_ENV_TO_POOL: dict[str, str] = {"d1": "d1", "ed1": "ed1"}


class TaskSplitManifestError(ValueError):
    """A typed failure reading or applying a task-selection manifest.

    Distinct from a bare ``ValueError`` so callers (CLI, env builders) can
    surface a manifest problem as its own class of refusal -- never a silent
    drop or a misfilter.
    """


@dataclass(frozen=True, slots=True)
class TaskSplitRoles:
    """One pool's resolved role sets for a manifest-driven cell.

    ``internal_ids`` is ``train`` FOLLOWED BY ``val`` (identity-bearing order);
    ``official_ids`` is ``test`` exactly. ``pool_key`` is the manifest pool the
    env selected. ``content_hash`` is the whole-manifest content hash folded
    into each split's Task Set identity.
    """

    pool_key: str
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    content_hash: str

    @property
    def internal_ids(self) -> tuple[str, ...]:
        """The optimizer-facing split: ``train`` then ``val`` (documented)."""
        return self.train_ids + self.val_ids

    @property
    def official_ids(self) -> tuple[str, ...]:
        """The certification split: ``test`` (membership, not slice)."""
        return self.test_ids

    def all_role_ids(self) -> frozenset[str]:
        """Every id the manifest assigns a role (train | val | test)."""
        return frozenset(self.train_ids + self.val_ids + self.test_ids)


@dataclass(frozen=True, slots=True)
class TaskSplitManifest:
    """A loaded ``whetstone.run.task_selection/v1`` manifest.

    ``content_hash`` is a stable sha256 over the canonical JSON of the whole
    file (provenance-bearing; folds into every split's Task Set identity).
    ``pools`` is the raw pools map (each pool carries ``train`` / ``val`` /
    ``test`` plus ``dropped_*`` arrays).
    """

    path: Path
    content_hash: str
    pools: dict[str, dict[str, object]]

    def for_env(self, env: str) -> TaskSplitRoles:
        """The resolved role sets for ``env`` (``d1`` or ``ed1``).

        Refuses ``ed1m`` (mutant ids, not HumanEval task ids) and any other env
        not backed by the manifest's HumanEval-task role arrays -- a typed
        ``TaskSplitManifestError`` rather than a silent misfilter.
        """
        pool_key = _ENV_TO_POOL.get(env)
        if pool_key is None:
            if env == "ed1m":
                raise TaskSplitManifestError(
                    "--task-split-manifest does NOT apply to ed1m: the "
                    "manifest's pools carry bare HumanEval task ids "
                    "(HumanEval/NN), but ed1m runs on the behavioral-mutant "
                    "pool whose Instance ids are "
                    "'HumanEval/NN::family::sNN::nSITE' mutant ids -- they "
                    "can never match, so applying the manifest would silently "
                    "misfilter. Scope this flag to ed1 + d1."
                )
            raise TaskSplitManifestError(
                f"--task-split-manifest applies only to envs "
                f"{sorted(_ENV_TO_POOL)}; got env {env!r}."
            )
        pool = self.pools.get(pool_key)
        if pool is None:
            raise TaskSplitManifestError(
                f"manifest {self.path} has no pool {pool_key!r} for env "
                f"{env!r} (pools present: {sorted(self.pools)})."
            )
        return TaskSplitRoles(
            pool_key=pool_key,
            train_ids=_role_ids(pool, "train", pool_key, self.path),
            val_ids=_role_ids(pool, "val", pool_key, self.path),
            test_ids=_role_ids(pool, "test", pool_key, self.path),
            content_hash=self.content_hash,
        )


def _role_ids(
    pool: dict[str, object], role: str, pool_key: str, path: Path
) -> tuple[str, ...]:
    """One role's ordered task ids from a pool, or a typed failure."""
    raw = pool.get(role)
    if not isinstance(raw, list):
        raise TaskSplitManifestError(
            f"manifest {path} pool {pool_key!r} is missing a list-valued "
            f"{role!r} role array."
        )
    ids = tuple(str(x) for x in raw)
    if len(set(ids)) != len(ids):
        raise TaskSplitManifestError(
            f"manifest {path} pool {pool_key!r} role {role!r} has duplicate "
            "task ids."
        )
    return ids


@dataclass(frozen=True, slots=True)
class ResolvedSplit[T]:
    """A manifest-resolved internal/official partition over a task pool.

    ``internal`` = the role sets' ``train + val`` items (membership-selected
    from the pool, in manifest order); ``official`` = ``test`` items
    (membership, capped by ``official_n`` AFTER selection). ``official_capped``
    records a loud cap note (or ``None``). ``manifest_tag`` folds the manifest
    provenance + selected pool into each split's Task Set identity.
    """

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
    """Partition ``items`` by the manifest ``roles`` (membership, not slicing).

    ``id_of`` maps an item to its task id. Every role id (train | val | test)
    MUST be present in ``items`` -- an unknown id is a typed
    :class:`TaskSplitManifestError` NAMING the offenders (no silent drop). The
    ``internal`` split preserves the manifest's train-then-val order; the
    ``official`` split preserves the manifest's test order. ``official_n``
    caps the official (``test``) split to its first N AFTER membership; a cap
    greater than the test size is a LOUD note (matching ``--official-n`` slice
    semantics), never a hard error.
    """
    by_id: dict[str, T] = {}
    for item in items:
        by_id[str(id_of(item))] = item
    known = frozenset(by_id)
    missing = sorted(roles.all_role_ids() - known)
    if missing:
        raise TaskSplitManifestError(
            f"manifest pool {roles.pool_key!r} references "
            f"{len(missing)} task id(s) absent from the loaded task pool "
            f"(unknown ids: {missing}). The manifest and the pool disagree -- "
            "refusing rather than silently dropping."
        )
    internal = tuple(by_id[i] for i in roles.internal_ids)
    official = tuple(by_id[i] for i in roles.official_ids)
    capped: str | None = None
    if official_n is not None and official_n < len(official):
        capped = (
            f"--official-n={official_n} caps the manifest test split "
            f"({len(official)} tasks) to its first {official_n}."
        )
        official = official[:official_n]
    elif official_n is not None and official_n > len(official):
        capped = (
            f"--official-n={official_n} exceeds the manifest test split size "
            f"({len(official)}); using all {len(official)} test tasks."
        )
    manifest_tag = f"tsm:{roles.content_hash[:16]}.{roles.pool_key}"
    return ResolvedSplit(
        internal=internal,
        official=official,
        manifest_tag=manifest_tag,
        official_capped=capped,
    )


def load_task_split_manifest(path: Path) -> TaskSplitManifest:
    """Load + validate a ``whetstone.run.task_selection/v1`` manifest.

    Reads the JSON, verifies the ``schema`` string, and computes a stable
    whole-file ``content_hash`` (sha256 over canonical JSON) that folds into
    every manifest-driven split's identity. A wrong schema, a missing
    ``pools`` map, or unparsable JSON raises a typed
    :class:`TaskSplitManifestError`.
    """
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise TaskSplitManifestError(
            f"task-split manifest not found: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise TaskSplitManifestError(
            f"task-split manifest {path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise TaskSplitManifestError(
            f"task-split manifest {path} must be a JSON object."
        )
    schema = data.get("schema")
    if schema != TASK_SELECTION_SCHEMA:
        raise TaskSplitManifestError(
            f"task-split manifest {path} has schema {schema!r}; expected "
            f"{TASK_SELECTION_SCHEMA!r}."
        )
    pools = data.get("pools")
    if not isinstance(pools, dict):
        raise TaskSplitManifestError(
            f"task-split manifest {path} is missing a 'pools' object."
        )
    # Canonical serialization -> the content hash is order-insensitive over
    # object keys but preserves array order (role-membership, not identity-
    # bearing here since roles fold in explicitly). Provenance-bearing.
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    resolved_pools = {
        str(k): dict(v) for k, v in pools.items() if isinstance(v, dict)
    }
    return TaskSplitManifest(
        path=path,
        content_hash=content_hash,
        pools=resolved_pools,
    )


__all__ = [
    "TASK_SELECTION_SCHEMA",
    "ResolvedSplit",
    "TaskSplitManifest",
    "TaskSplitManifestError",
    "TaskSplitRoles",
    "load_task_split_manifest",
    "resolve_manifest_split",
]
