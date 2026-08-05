"""The validation ledgers: ``cells.jsonl`` and ``spend.jsonl``.

``cells.jsonl`` is the authoritative resumability ledger -- one line per cell
*attempt*. A completed ``(optimizer, env, attempt)`` cell is skipped on resume;
an interrupted cell is resumed, because the optimizer harness owns its
optimization state, or restarted, recording which.

``spend.jsonl`` records the OpenRouter credits snapshot before and after each
cell, so cumulative spend is auditable and the budget guards (reserve plus
per-cell stop-loss) key off the persisted remaining.

Both ledgers are append-only JSONL. Schema validation is enforced on write: a
malformed cell line is refused, never silently appended. Canonical viewer
sources are published under an exclusive ``.viewer-sources.lock``; snapshot
consumers hold its shared side through their publication boundary.

Cell records are content-complete: every derived statistic they carry can be
recomputed from the durable evidence the record cites. The ledger is the
runner's own durable record, not a second source of truth for evaluation
evidence, which stays content-addressed in the ObjectStore.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import stat
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    model_validator,
)

from whetstone.core.identity import TypedRef, require_full_hash
from whetstone.evaluation.schema_names import EVALUATION_EVIDENCE_SCHEMA

__all__ = [
    "BOUNDARY_SPEND_PHASES",
    "CELLS_SCHEMA",
    "CELL_STATUSES",
    "CHECKPOINT_SPEND_PHASE_PREFIX",
    "COMPLETED_CELL_STATUSES",
    "OFFICIAL_ANCHOR_COUNT_MAX",
    "OFFICIAL_ANCHOR_SCHEMA",
    "SPEND_SCHEMA",
    "CellArtifacts",
    "CellControls",
    "CellModels",
    "CellRecord",
    "CellSampling",
    "CellTelemetry",
    "Ledger",
    "OfficialAnchorRecord",
    "PromptCacheTelemetry",
    "SpendRecord",
    "ViewerCellPublicationRef",
    "ViewerPublishedFileRef",
    "cell_key",
]

#: Persisted-format contract: the versioned schema stamp on each ledger
#: artifact type's rows. A structured reader branches on the stamp instead of
#: sniffing which keys happen to be present.
CELLS_SCHEMA = "whetstone.runner.cells/v1"
SPEND_SCHEMA = "whetstone.runner.spend/v1"
OFFICIAL_ANCHOR_SCHEMA = "whetstone.runner.official_anchor/v1"

#: Official-anchor repeats and counts remain representable as signed 64-bit
#: integers in every persisted consumer.
OFFICIAL_ANCHOR_COUNT_MAX = 2**63 - 1

#: Persisted-format contract: the paired outer boundary of one cell's spend
#: timeline. Exactly one ``before`` is open at a time, and it is the stop-loss
#: baseline a resumed invocation recovers.
BOUNDARY_SPEND_PHASES: frozenset[str] = frozenset({"before", "after"})

#: Persisted-format contract: the prefix marking one paid boundary inside a
#: cell. The suffix names the boundary being guarded.
CHECKPOINT_SPEND_PHASE_PREFIX = "checkpoint:"


def _canonical_cell_filename(
    cell_id: str,
    *,
    expected_env: str | None = None,
) -> str:
    """Validate and encode one canonical cell identity for storage."""
    components = cell_id.split(":")
    if len(components) != 3:
        raise ValueError(
            "cell_id must be exactly "
            "'<optimizer>:<env>:a<nonnegative integer>'"
        )
    optimizer, env, attempt = components
    if not optimizer or not env:
        raise ValueError(
            "cell_id must be exactly "
            "'<optimizer>:<env>:a<nonnegative integer>'"
        )
    if any(
        "/" in component or "\\" in component or "__" in component
        for component in components
    ):
        raise ValueError(
            "cell_id components must not contain '/', '\\\\', or '__'"
        )
    attempt_digits = attempt[1:] if attempt.startswith("a") else ""
    if not (
        attempt_digits == "0"
        or (
            attempt_digits
            and attempt_digits.isascii()
            and attempt_digits.isdecimal()
            and attempt_digits[0] != "0"
        )
    ):
        raise ValueError(
            "cell_id must be exactly "
            "'<optimizer>:<env>:a<nonnegative integer>'"
        )
    if expected_env is not None and env != expected_env:
        raise ValueError("cell_id env segment must equal env")
    return "__".join(components)


def _official_anchor_filename(
    cell_id: str,
    *,
    expected_env: str | None = None,
) -> str:
    return (
        _canonical_cell_filename(cell_id, expected_env=expected_env) + ".json"
    )


#: Persisted-format contract: the closed set of cell statuses.
#:
#: ``improved`` -- a positive delta whose paired CI excludes 0.
#: ``inconclusive`` -- a positive delta whose paired CI still spans 0.
#: ``no-improvement`` -- real candidates were scored and none improved.
#: ``plumbing-retry`` -- the attempt failed for non-statistical reasons.
#: ``halted`` -- the cell crossed its stop-loss.
#: ``incomplete-arm`` -- an official arm's aggregate never resolved, so the
#: cell emits no headroom determination and no terminal statistical status.
#: ``proposer-failure`` -- every draft in the run was a typed proposer-draft
#: failure, so no real candidate was ever explored. This is not an honest
#: ``no-improvement``: it is a proposer outage.
#:
#: ``incomplete-arm`` and ``proposer-failure`` are not certified results; a
#: re-run supersedes them.
CELL_STATUSES: frozenset[str] = frozenset(
    {
        "improved",
        "inconclusive",
        "no-improvement",
        "plumbing-retry",
        "halted",
        "incomplete-arm",
        "proposer-failure",
    }
)

#: Persisted-format contract: the statuses that mark an attempt terminal, so a
#: resume skips the cell rather than re-running it.
COMPLETED_CELL_STATUSES: frozenset[str] = frozenset(
    {"improved", "inconclusive", "no-improvement", "halted"}
)


class ViewerPublishedFileRef(BaseModel):
    """Content hash and ledger-relative path of one immutable file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    relative_path: StrictStr
    sha256: StrictStr

    @model_validator(mode="after")
    def _validate_ref(self) -> ViewerPublishedFileRef:
        require_full_hash(self.sha256, field="sha256")
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != self.relative_path
        ):
            raise ValueError("published path must be canonical and relative")
        return self


class ViewerCellPublicationRef(BaseModel):
    """The two immutable files committed as one viewer-cell directory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    projection: ViewerPublishedFileRef
    rollout_outputs: ViewerPublishedFileRef

    @model_validator(mode="after")
    def _validate_pair(self) -> ViewerCellPublicationRef:
        projection = PurePosixPath(self.projection.relative_path)
        rollouts = PurePosixPath(self.rollout_outputs.relative_path)
        if (
            projection.name != "projection.json"
            or rollouts.name != "rollout_outputs.jsonl"
            or projection.parent != rollouts.parent
            or len(projection.parts) != 3
            or projection.parts[0] != "viewer_cells"
        ):
            raise ValueError(
                "viewer publication files must share one canonical cell "
                "directory"
            )
        return self


class CellModels(BaseModel):
    """The ``models: {task, proposer}`` sub-object."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task: StrictStr
    proposer: StrictStr


class CellSampling(BaseModel):
    """The official-arm sampling a cell ran under.

    ``official_n`` and ``official_repeats`` fold into the composite Eval Config
    Identity Hash, so this is the auditable record of why a reduced cell has a
    distinct Eval Config identity. ``None`` means the spec default applied.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    official_n: StrictInt | None = None
    official_repeats: StrictInt | None = None


class CellTelemetry(BaseModel):
    """Per-cell task-side usage and latency totals over the partial log.

    Every field is coverage-honest: a total is ``None`` when no row reported it
    (never a fake 0), and each ``*_coverage`` counts the rows the total was
    actually summed over, so a partial-coverage cell is never mistaken for a
    full one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_prompt_tokens: StrictInt | None = None
    total_completion_tokens: StrictInt | None = None
    total_tokens: StrictInt | None = None
    total_reasoning_tokens: StrictInt | None = None
    total_latency_s: float | None = None
    mean_latency_s: float | None = None
    token_coverage: StrictInt = 0
    reasoning_coverage: StrictInt = 0
    latency_coverage: StrictInt = 0


class PromptCacheTelemetry(BaseModel):
    """Recording-only prompt-cache telemetry for a cell.

    Present only when the prompt cache was on. ``hits`` / ``misses`` /
    ``stores`` are the run-scoped store's counters this cell observed: a hit
    reused a stored Result, a miss drove the transport and then stored it.
    Recording-only -- these never fold into any identity hash.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: StrictBool = True
    hits: StrictInt = 0
    misses: StrictInt = 0
    stores: StrictInt = 0


class CellControls(BaseModel):
    """The literal sampling-control values a cell ran under.

    ``temperature`` and ``reasoning_effort`` fold into the Provider Call Config
    identity, so recording the literal values here makes "did this anchor run
    at temperature 0 or 1?" answerable by reading the line rather than
    re-deriving it from a hash. ``None`` means the control was unset (the
    provider default), never conflated with an explicit 0. ``prompt_cache`` is
    recording-only and never participates in identity.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature: float | None = None
    reasoning_effort: StrictStr | None = None
    prompt_cache: PromptCacheTelemetry | None = None


class CellArtifacts(BaseModel):
    """Typed canonical records and human-readable reporting projections."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    optimization_result_ref: TypedRef | None = None
    optimization_trace_ref: StrictStr | None = None
    best_candidate_id: StrictStr | None = None
    official_record_before: TypedRef | None = None
    official_record_after: TypedRef | None = None
    viewer_publication: ViewerCellPublicationRef | None = None


class CellRecord(BaseModel):
    """One ``cells.jsonl`` line: the durable record of one cell attempt.

    Identity and configuration: ``cell_id`` / ``optimizer`` / ``env`` /
    ``attempt`` / ``canonical`` / ``lane`` / ``models`` / ``controls`` /
    ``sampling`` / ``graph_hash`` / ``eval_config_hash``.

    Official-arm scores: ``baseline_official`` (the naive arm),
    ``ceiling_official``, and ``best_official``, with the marginal
    task-bootstrap intervals ``naive_ci95`` and ``ceiling_ci95``.

    The statistical verdict: ``delta`` with its paired best-minus-naive
    interval ``delta_ci95``, which is the interval the ``status`` is read off;
    ``headroom_delta`` with its paired ceiling-minus-naive interval
    ``headroom_ci95``, which is the Eval-row headroom gate;
    ``official_repeats_used``; ``escalated`` for a cell that auto-doubled its
    repeats after an inconclusive verdict; and ``pooled_observation_counts``,
    the per-arm total observations behind the reported per-task means.

    Accounting: ``internal_evals_count`` / ``optimizer_steps`` / ``spend_usd``
    / ``wall_s`` / ``started_at`` / ``finished_at`` / ``telemetry``.

    Every interval is a ``(low, high)`` pair or ``None``. Null means unknown,
    never zero.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True
    )

    #: Persisted-format contract: keep this exact wire key and version.
    schema_: StrictStr = Field(default=CELLS_SCHEMA, alias="schema")
    cell_id: StrictStr
    optimizer: StrictStr
    env: StrictStr
    attempt: StrictInt
    canonical: StrictBool
    models: CellModels
    baseline_official: float | None
    ceiling_official: float | None
    best_official: float | None
    delta: float | None
    delta_ci95: tuple[float, float] | None = None
    naive_ci95: tuple[float, float] | None = None
    ceiling_ci95: tuple[float, float] | None = None
    headroom_delta: float | None = None
    headroom_ci95: tuple[float, float] | None = None
    official_repeats_used: StrictInt = 0
    escalated: StrictBool = False
    escalation_note: StrictStr = ""
    pooled_observation_counts: dict[StrictStr, StrictInt] = Field(
        default_factory=dict
    )
    internal_evals_count: StrictInt
    optimizer_steps: StrictInt
    spend_usd: float
    wall_s: float
    lane: StrictStr
    window_notes: StrictStr = ""
    status: StrictStr
    artifacts: CellArtifacts = Field(default_factory=CellArtifacts)
    sampling: CellSampling = Field(default_factory=CellSampling)
    telemetry: CellTelemetry = Field(default_factory=CellTelemetry)
    controls: CellControls = Field(default_factory=CellControls)
    #: The content-addressed identity of the exact resolved graph and Eval
    #: Config the official arm ran under, recorded on every cell including
    #: anchors. Recording-only: the runner already computes both hashes.
    graph_hash: StrictStr | None = None
    eval_config_hash: StrictStr | None = None
    #: ISO-8601 UTC wall-clock the cell started and finished, so concurrency
    #: interleaving is reconstructable from the line's absolute timestamps
    #: rather than from ``wall_s`` alone.
    started_at: StrictStr | None = None
    finished_at: StrictStr | None = None

    @model_validator(mode="after")
    def _validate(self) -> CellRecord:
        if self.schema_ != CELLS_SCHEMA:
            raise ValueError(
                f"schema must be exactly {CELLS_SCHEMA!r}; "
                f"got {self.schema_!r}"
            )
        if self.status not in CELL_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(CELL_STATUSES)}; "
                f"got {self.status!r}"
            )
        _canonical_cell_filename(self.cell_id, expected_env=self.env)
        if self.cell_id != f"{self.optimizer}:{self.env}:a{self.attempt}":
            raise ValueError("cell identity fields do not align")
        if self.attempt < 0:
            raise ValueError("cell attempt cannot be negative")
        for name in (
            "delta_ci95",
            "naive_ci95",
            "ceiling_ci95",
            "headroom_ci95",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if len(value) != 2:
                raise ValueError(f"{name} must be a (low, high) pair or null")
            if value[0] > value[1]:
                raise ValueError(f"{name} bounds must be ordered (low, high)")
        if self.official_repeats_used < 0:
            raise ValueError("official_repeats_used cannot be negative")
        if any(count < 0 for count in self.pooled_observation_counts.values()):
            raise ValueError("pooled observation counts cannot be negative")
        # A completed cell always publishes its immutable viewer directory, so
        # a terminal line can never cite evidence that was never committed.
        if self.is_completed() and self.artifacts.viewer_publication is None:
            raise ValueError("a completed cell requires viewer publication")
        publication = self.artifacts.viewer_publication
        if publication is not None:
            expected_parent = (
                Path("viewer_cells") / _canonical_cell_filename(self.cell_id)
            ).as_posix()
            actual_parent = Path(
                publication.projection.relative_path
            ).parent.as_posix()
            if actual_parent != expected_parent:
                raise ValueError(
                    "viewer publication path must match cell identity"
                )
        return self

    def key(self) -> tuple[str, str, int]:
        return cell_key(self.optimizer, self.env, self.attempt)

    def is_completed(self) -> bool:
        """A completed cell is any terminal, non-retry status."""
        return self.status in COMPLETED_CELL_STATUSES

    def to_line(self) -> str:
        return json.dumps(
            self.model_dump(mode="json", by_alias=True), sort_keys=True
        )

    @classmethod
    def from_line(cls, line: str) -> CellRecord:
        return cls.model_validate_json(line)


class OfficialAnchorRecord(BaseModel):
    """Viewer-facing projection of one cell's official anchor evaluations.

    Canonical evaluation evidence stays content-addressed in the ObjectStore;
    this record only projects the aligned official task values, counts, and
    typed evidence references for reporting. Runner execution never reads this
    projection for reuse.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        allow_inf_nan=False,
    )

    #: Persisted-format contract: keep this exact wire key and version.
    schema_: StrictStr = Field(default=OFFICIAL_ANCHOR_SCHEMA, alias="schema")
    cell_id: StrictStr
    env: StrictStr
    task_model: StrictStr
    graph_hash: StrictStr
    eval_config_hash: StrictStr
    #: Raw environment IDs and semantic task identities preserve evaluation
    #: order independently; every per-task vector below aligns to both.
    official_instance_ids: tuple[StrictStr, ...]
    official_task_identities: tuple[StrictStr, ...]
    baseline_evidence_ref: TypedRef
    ceiling_evidence_ref: TypedRef
    baseline_official: StrictFloat
    ceiling_official: StrictFloat
    baseline_per_task: tuple[StrictFloat, ...]
    ceiling_per_task: tuple[StrictFloat, ...]
    baseline_per_task_counts: tuple[StrictInt, ...]
    ceiling_per_task_counts: tuple[StrictInt, ...]
    official_repeats_used: StrictInt

    @model_validator(mode="after")
    def _validate_contract(self) -> OfficialAnchorRecord:
        if self.schema_ != OFFICIAL_ANCHOR_SCHEMA:
            raise ValueError(
                f"schema must be exactly {OFFICIAL_ANCHOR_SCHEMA!r}"
            )
        for field_name in ("cell_id", "env", "task_model"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        _official_anchor_filename(self.cell_id, expected_env=self.env)
        require_full_hash(self.graph_hash, field="graph_hash")
        require_full_hash(self.eval_config_hash, field="eval_config_hash")

        if not self.official_instance_ids:
            raise ValueError(
                "official_instance_ids must contain at least one ID"
            )
        if any(
            not instance_id.strip()
            for instance_id in self.official_instance_ids
        ):
            raise ValueError(
                "official_instance_ids must contain non-empty IDs"
            )
        if len(set(self.official_instance_ids)) != len(
            self.official_instance_ids
        ):
            raise ValueError("official_instance_ids must contain unique IDs")

        for identity in self.official_task_identities:
            require_full_hash(identity, field="official_task_identities")
        if len(set(self.official_task_identities)) != len(
            self.official_task_identities
        ):
            raise ValueError(
                "official_task_identities must contain unique hashes"
            )

        official_count = len(self.official_instance_ids)
        aligned_fields = (
            "official_task_identities",
            "baseline_per_task",
            "ceiling_per_task",
            "baseline_per_task_counts",
            "ceiling_per_task_counts",
        )
        for field_name in aligned_fields:
            if len(getattr(self, field_name)) != official_count:
                raise ValueError(
                    f"{field_name} length must match official_instance_ids"
                )

        if self.official_repeats_used <= 0:
            raise ValueError("official_repeats_used must be positive")
        if self.official_repeats_used > OFFICIAL_ANCHOR_COUNT_MAX:
            raise ValueError(
                "official_repeats_used must be at most "
                f"{OFFICIAL_ANCHOR_COUNT_MAX}"
            )
        for field_name in (
            "baseline_per_task_counts",
            "ceiling_per_task_counts",
        ):
            counts = getattr(self, field_name)
            if any(count > OFFICIAL_ANCHOR_COUNT_MAX for count in counts):
                raise ValueError(
                    f"{field_name} values must be at most "
                    f"{OFFICIAL_ANCHOR_COUNT_MAX}"
                )
            if any(
                count < 0 or count > self.official_repeats_used
                for count in counts
            ):
                raise ValueError(
                    f"{field_name} values must be between 0 and "
                    "official_repeats_used"
                )

        for field_name in (
            "baseline_evidence_ref",
            "ceiling_evidence_ref",
        ):
            reference = getattr(self, field_name)
            if reference.schema_name != EVALUATION_EVIDENCE_SCHEMA:
                raise ValueError(
                    f"{field_name} must reference "
                    f"{EVALUATION_EVIDENCE_SCHEMA!r}"
                )
        return self

    def to_json(self) -> str:
        """Serialize the stable viewer projection as one complete JSON file."""
        return (
            json.dumps(
                self.model_dump(mode="json", by_alias=True),
                indent=2,
            )
            + "\n"
        )


class SpendRecord(BaseModel):
    """One ``spend.jsonl`` line: an OpenRouter credits snapshot.

    ``at`` is a real ISO-8601 UTC wall-clock, and it is ``None`` when genuinely
    never captured -- never an empty string, which would read as recorded but
    blank. ``event_id`` is a per-row unique id so a spend timeline can address
    individual snapshots.

    A cell's spend timeline has three kinds of line. ``before`` and ``after``
    are its paired outer boundary, and exactly one ``before`` is open at a
    time -- that open record is the stop-loss baseline a resumed invocation
    recovers. Between them sit the ``checkpoint:<boundary>`` lines, one per
    paid boundary the cell is about to cross, each keyed by a deterministic
    ``event_id`` so re-entering the same boundary reuses its snapshot instead
    of paying for a second one.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True
    )

    #: Persisted-format contract: keep this exact wire key and version.
    schema_: StrictStr = Field(default=SPEND_SCHEMA, alias="schema")
    event_id: StrictStr | None = None
    cell_id: StrictStr
    #: Persisted-format contract: exactly ``"before"``, ``"after"``, or
    #: ``"checkpoint:<boundary>"`` with a non-empty boundary name.
    phase: StrictStr
    lane: StrictStr
    total_credits: float | None = None
    total_usage: float | None = None
    remaining_usd: float | None = None
    at: StrictStr | None = None

    @model_validator(mode="after")
    def _validate(self) -> SpendRecord:
        if self.schema_ != SPEND_SCHEMA:
            raise ValueError(f"schema must be exactly {SPEND_SCHEMA!r}")
        if self.phase not in BOUNDARY_SPEND_PHASES and not (
            self.phase.startswith(CHECKPOINT_SPEND_PHASE_PREFIX)
            and self.phase[len(CHECKPOINT_SPEND_PHASE_PREFIX) :].strip()
        ):
            raise ValueError(
                "phase must be 'before', 'after', or "
                f"'{CHECKPOINT_SPEND_PHASE_PREFIX}<boundary>'"
            )
        return self

    def to_line(self) -> str:
        return json.dumps(
            self.model_dump(mode="json", by_alias=True), sort_keys=True
        )

    @classmethod
    def from_line(cls, line: str) -> SpendRecord:
        return cls.model_validate_json(line)


def cell_key(optimizer: str, env: str, attempt: int) -> tuple[str, str, int]:
    """The resumability key for a cell attempt."""
    return (optimizer, env, attempt)


@dataclass(slots=True)
class Ledger:
    """Append-only cells and spend JSONL ledgers rooted at a directory.

    ``cells_path`` is ``<root>/cells.jsonl``; ``spend_path`` is
    ``<root>/spend.jsonl``. :meth:`completed_keys` drives resumability: a cell
    whose ``(optimizer, env, attempt)`` key has a completed record is skipped.
    """

    root: Path
    _cells: list[CellRecord] = field(default_factory=list)
    _loaded: bool = False

    @property
    def cells_path(self) -> Path:
        return self.root / "cells.jsonl"

    @property
    def spend_path(self) -> Path:
        return self.root / "spend.jsonl"

    @property
    def official_anchors_dir(self) -> Path:
        return self.root / "official_anchors"

    def official_anchor_path(self, cell_id: str) -> Path:
        """The per-cell official-anchor projection path (``:`` -> ``__``)."""
        return self.official_anchors_dir / _official_anchor_filename(cell_id)

    @contextmanager
    def _exclusive_viewer_source_lock(self) -> Iterator[None]:
        """Exclude viewer snapshots while changing authoritative sources."""
        root_existed = self.root.exists()
        self.root.mkdir(parents=True, exist_ok=True)
        if not root_existed:
            parent_descriptor = os.open(
                self.root.parent,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
            )
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
        lock_path = self.root / ".viewer-sources.lock"
        descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _open_official_anchors_dir(self) -> tuple[int, int, int, str]:
        """Open the root and anchor directory through retained parents.

        The retained parent descriptor pins the boundary immediately above
        ``root``; ancestors above that parent are not recursively pinned.
        """
        absolute_root = Path(os.path.abspath(self.root))
        absolute_root.mkdir(parents=True, exist_ok=True)
        root_entry = absolute_root.name or "."
        parent_descriptor = os.open(
            absolute_root.parent,
            os.O_RDONLY | os.O_DIRECTORY,
        )
        root_descriptor = -1
        directory_descriptor: int | None = None
        created = False
        try:
            root_descriptor = os.open(
                root_entry,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_descriptor,
            )
            try:
                os.mkdir(
                    "official_anchors",
                    mode=0o755,
                    dir_fd=root_descriptor,
                )
                created = True
            except FileExistsError:
                pass
            try:
                directory_descriptor = os.open(
                    "official_anchors",
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=root_descriptor,
                )
            except OSError as exc:
                raise RuntimeError(
                    f"{self.official_anchors_dir} must be a real directory; "
                    "refusing to follow or replace it"
                ) from exc
            if created:
                os.fsync(root_descriptor)
            opened_descriptor = directory_descriptor
            directory_descriptor = None
            opened_root_descriptor = root_descriptor
            root_descriptor = -1
            opened_parent_descriptor = parent_descriptor
            parent_descriptor = -1
            return (
                opened_parent_descriptor,
                opened_root_descriptor,
                opened_descriptor,
                root_entry,
            )
        finally:
            if directory_descriptor is not None:
                os.close(directory_descriptor)
            if root_descriptor >= 0:
                os.close(root_descriptor)
            if parent_descriptor >= 0:
                os.close(parent_descriptor)

    def _validate_root_binding(
        self,
        parent_descriptor: int,
        root_descriptor: int,
        root_entry: str,
    ) -> None:
        """Require the opened root to remain bound beneath its held parent."""
        opened_stat = os.fstat(root_descriptor)
        try:
            current_stat = os.stat(
                root_entry,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise RuntimeError(
                "ledger root changed while publishing at "
                f"{self.root}; refusing to report success"
            ) from exc
        if (
            not stat.S_ISDIR(opened_stat.st_mode)
            or not stat.S_ISDIR(current_stat.st_mode)
            or (current_stat.st_dev, current_stat.st_ino)
            != (opened_stat.st_dev, opened_stat.st_ino)
        ):
            raise RuntimeError(
                "ledger root changed while publishing at "
                f"{self.root}; refusing to report success"
            )

    def _validate_official_anchors_binding(
        self,
        root_descriptor: int,
        directory_descriptor: int,
    ) -> None:
        """Require the opened directory to remain bound beneath the root."""
        opened_stat = os.fstat(directory_descriptor)
        try:
            current_stat = os.stat(
                "official_anchors",
                dir_fd=root_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise RuntimeError(
                "official-anchor directory changed while publishing at "
                f"{self.official_anchors_dir}; refusing to report success"
            ) from exc
        if (
            not stat.S_ISDIR(opened_stat.st_mode)
            or not stat.S_ISDIR(current_stat.st_mode)
            or (current_stat.st_dev, current_stat.st_ino)
            != (opened_stat.st_dev, opened_stat.st_ino)
        ):
            raise RuntimeError(
                "official-anchor directory changed while publishing at "
                f"{self.official_anchors_dir}; refusing to report success"
            )

    @staticmethod
    def _read_official_anchor(
        directory_descriptor: int,
        filename: str,
        path: Path,
    ) -> OfficialAnchorRecord | None:
        """Read a regular target directly beneath the verified directory."""
        try:
            target_descriptor = os.open(
                filename,
                os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
                dir_fd=directory_descriptor,
            )
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise RuntimeError(
                "official-anchor projection is unsafe at "
                f"{path}; refusing to follow or replace it"
            ) from exc

        try:
            opened_stat = os.fstat(target_descriptor)
            if not stat.S_ISREG(opened_stat.st_mode):
                raise RuntimeError(
                    "official-anchor projection is not a regular file at "
                    f"{path}; refusing to replace it"
                )
            with os.fdopen(target_descriptor, encoding="utf-8") as handle:
                target_descriptor = -1
                body = handle.read()
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(
                "official-anchor projection is invalid at "
                f"{path}; refusing to replace it"
            ) from exc
        finally:
            if target_descriptor >= 0:
                os.close(target_descriptor)

        try:
            current_stat = os.stat(
                filename,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise RuntimeError(
                "official-anchor projection changed while reading at "
                f"{path}; refusing to replace it"
            ) from exc
        if not stat.S_ISREG(current_stat.st_mode) or (
            current_stat.st_dev,
            current_stat.st_ino,
        ) != (opened_stat.st_dev, opened_stat.st_ino):
            raise RuntimeError(
                "official-anchor projection changed while reading at "
                f"{path}; refusing to replace it"
            )

        try:
            return OfficialAnchorRecord.model_validate_json(body)
        except ValidationError as exc:
            raise RuntimeError(
                "official-anchor projection is invalid at "
                f"{path}; refusing to replace it"
            ) from exc

    @staticmethod
    def _create_official_anchor_temp(
        directory_descriptor: int,
        filename: str,
    ) -> tuple[int, str]:
        """Create a unique private temp file beneath the verified directory."""
        for _ in range(128):
            temporary = f".{filename}.{secrets.token_hex(16)}.tmp"
            try:
                descriptor = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory_descriptor,
                )
            except FileExistsError:
                continue
            return descriptor, temporary
        raise RuntimeError(
            f"could not create a unique temporary file for {filename}"
        )

    def write_official_anchor(self, record: OfficialAnchorRecord) -> Path:
        """Publish an anchor outside every stable viewer source snapshot."""
        with self._exclusive_viewer_source_lock():
            return self._write_official_anchor_unlocked(record)

    def _write_official_anchor_unlocked(
        self,
        record: OfficialAnchorRecord,
    ) -> Path:
        """Atomically publish one immutable official-anchor projection.

        An equal existing projection makes the write idempotent. A differing
        or invalid target is preserved and rejected: official evidence may not
        be silently reassigned for an already-published cell identity.
        """
        filename = _official_anchor_filename(
            record.cell_id,
            expected_env=record.env,
        )
        path = self.official_anchors_dir / filename
        (
            parent_descriptor,
            root_descriptor,
            directory_descriptor,
            root_entry,
        ) = self._open_official_anchors_dir()
        temporary: str | None = None
        temporary_descriptor: int | None = None
        try:
            fcntl.flock(directory_descriptor, fcntl.LOCK_EX)
            existing = self._read_official_anchor(
                directory_descriptor,
                filename,
                path,
            )
            if existing is not None:
                if existing != record:
                    raise RuntimeError(
                        "official-anchor projection conflicts at "
                        f"{path}; refusing to replace it"
                    )
                self._validate_root_binding(
                    parent_descriptor,
                    root_descriptor,
                    root_entry,
                )
                self._validate_official_anchors_binding(
                    root_descriptor,
                    directory_descriptor,
                )
                return path

            temporary_descriptor, temporary = (
                self._create_official_anchor_temp(
                    directory_descriptor,
                    filename,
                )
            )
            temporary_handle = os.fdopen(temporary_descriptor, "wb")
            temporary_descriptor = None
            with temporary_handle as handle:
                handle.write(record.to_json().encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(
                    temporary,
                    filename,
                    src_dir_fd=directory_descriptor,
                    dst_dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError:
                os.unlink(temporary, dir_fd=directory_descriptor)
                temporary = None
                os.fsync(directory_descriptor)
                existing = self._read_official_anchor(
                    directory_descriptor,
                    filename,
                    path,
                )
                if existing != record:
                    raise RuntimeError(
                        "official-anchor projection conflicts at "
                        f"{path}; refusing to replace it"
                    ) from None
                self._validate_root_binding(
                    parent_descriptor,
                    root_descriptor,
                    root_entry,
                )
                self._validate_official_anchors_binding(
                    root_descriptor,
                    directory_descriptor,
                )
                return path
            os.unlink(temporary, dir_fd=directory_descriptor)
            temporary = None
            os.fsync(directory_descriptor)
            self._validate_root_binding(
                parent_descriptor,
                root_descriptor,
                root_entry,
            )
            self._validate_official_anchors_binding(
                root_descriptor,
                directory_descriptor,
            )
            return path
        finally:
            if temporary_descriptor is not None:
                os.close(temporary_descriptor)
            if temporary is not None:
                try:
                    os.unlink(temporary, dir_fd=directory_descriptor)
                except FileNotFoundError:
                    pass
            os.close(directory_descriptor)
            os.close(root_descriptor)
            os.close(parent_descriptor)

    @property
    def viewer_cells_dir(self) -> Path:
        return self.root / "viewer_cells"

    def viewer_cell_dir(self, cell_id: str) -> Path:
        """The immutable publication directory for one canonical cell."""
        return self.viewer_cells_dir / _canonical_cell_filename(cell_id)

    def _open_viewer_cells_dir(self) -> tuple[int, int, int, str]:
        """Open and durably create the viewer publication parent."""
        absolute_root = Path(os.path.abspath(self.root))
        absolute_root.parent.mkdir(parents=True, exist_ok=True)
        root_entry = absolute_root.name or "."
        parent_descriptor = os.open(
            absolute_root.parent,
            os.O_RDONLY | os.O_DIRECTORY,
        )
        root_descriptor = -1
        directory_descriptor: int | None = None
        root_created = False
        directory_created = False
        try:
            try:
                os.mkdir(root_entry, mode=0o755, dir_fd=parent_descriptor)
                root_created = True
            except FileExistsError:
                pass
            try:
                root_descriptor = os.open(
                    root_entry,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=parent_descriptor,
                )
            except OSError as exc:
                raise RuntimeError(
                    f"{self.root} must be a real directory; refusing to "
                    "follow or replace it"
                ) from exc
            if root_created:
                os.fsync(parent_descriptor)
            try:
                os.mkdir("viewer_cells", mode=0o755, dir_fd=root_descriptor)
                directory_created = True
            except FileExistsError:
                pass
            try:
                directory_descriptor = os.open(
                    "viewer_cells",
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=root_descriptor,
                )
            except OSError as exc:
                raise RuntimeError(
                    f"{self.viewer_cells_dir} must be a real directory; "
                    "refusing to follow or replace it"
                ) from exc
            if directory_created:
                os.fsync(root_descriptor)
            opened_directory = directory_descriptor
            directory_descriptor = None
            opened_root = root_descriptor
            root_descriptor = -1
            opened_parent = parent_descriptor
            parent_descriptor = -1
            return (
                opened_parent,
                opened_root,
                opened_directory,
                root_entry,
            )
        finally:
            if directory_descriptor is not None:
                os.close(directory_descriptor)
            if root_descriptor >= 0:
                os.close(root_descriptor)
            if parent_descriptor >= 0:
                os.close(parent_descriptor)

    def _validate_viewer_cells_binding(
        self,
        root_descriptor: int,
        directory_descriptor: int,
    ) -> None:
        """Require the opened publication parent to remain beneath root."""
        opened_stat = os.fstat(directory_descriptor)
        try:
            current_stat = os.stat(
                "viewer_cells",
                dir_fd=root_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise RuntimeError(
                "viewer publication directory changed while publishing at "
                f"{self.viewer_cells_dir}"
            ) from exc
        if (
            not stat.S_ISDIR(opened_stat.st_mode)
            or not stat.S_ISDIR(current_stat.st_mode)
            or (current_stat.st_dev, current_stat.st_ino)
            != (opened_stat.st_dev, opened_stat.st_ino)
        ):
            raise RuntimeError(
                "viewer publication directory changed while publishing at "
                f"{self.viewer_cells_dir}"
            )

    @staticmethod
    def _write_publication_file(
        directory_descriptor: int,
        filename: str,
        body: bytes,
    ) -> None:
        descriptor = os.open(
            filename,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_descriptor,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @staticmethod
    def _read_publication_file(
        directory_descriptor: int,
        filename: str,
        path: Path,
    ) -> bytes:
        try:
            descriptor = os.open(
                filename,
                os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
                dir_fd=directory_descriptor,
            )
        except OSError as exc:
            raise RuntimeError(
                f"viewer publication file is unsafe at {path}"
            ) from exc
        try:
            opened_stat = os.fstat(descriptor)
            if not stat.S_ISREG(opened_stat.st_mode):
                raise RuntimeError(
                    f"viewer publication file is not regular at {path}"
                )
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = -1
                return handle.read()
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _read_viewer_publication(
        self,
        directory_descriptor: int,
        cell_filename: str,
        path: Path,
    ) -> tuple[bytes, bytes] | None:
        """Read one published cell's two immutable file bodies verbatim."""
        try:
            cell_descriptor = os.open(
                cell_filename,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_descriptor,
            )
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise RuntimeError(
                f"viewer publication is unsafe at {path}"
            ) from exc
        try:
            opened_cell_stat = os.fstat(cell_descriptor)
            if set(os.listdir(cell_descriptor)) != {
                "projection.json",
                "rollout_outputs.jsonl",
            }:
                raise RuntimeError(
                    f"viewer publication has an invalid file set at {path}"
                )
            projection_body = self._read_publication_file(
                cell_descriptor,
                "projection.json",
                path / "projection.json",
            )
            rollouts_body = self._read_publication_file(
                cell_descriptor,
                "rollout_outputs.jsonl",
                path / "rollout_outputs.jsonl",
            )
            try:
                current_cell_stat = os.stat(
                    cell_filename,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise RuntimeError(
                    f"viewer publication changed while reading at {path}"
                ) from exc
            if (
                not stat.S_ISDIR(opened_cell_stat.st_mode)
                or not stat.S_ISDIR(current_cell_stat.st_mode)
                or (current_cell_stat.st_dev, current_cell_stat.st_ino)
                != (opened_cell_stat.st_dev, opened_cell_stat.st_ino)
            ):
                raise RuntimeError(
                    f"viewer publication changed while reading at {path}"
                )
        finally:
            os.close(cell_descriptor)
        return projection_body, rollouts_body

    @staticmethod
    def _cleanup_publication_temp(
        directory_descriptor: int,
        temporary: str,
    ) -> None:
        try:
            temp_descriptor = os.open(
                temporary,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_descriptor,
            )
        except FileNotFoundError:
            return
        try:
            for filename in ("projection.json", "rollout_outputs.jsonl"):
                try:
                    os.unlink(filename, dir_fd=temp_descriptor)
                except FileNotFoundError:
                    pass
        finally:
            os.close(temp_descriptor)
        try:
            os.rmdir(temporary, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass

    def write_viewer_publication(
        self,
        *,
        cell_id: str,
        env: str,
        projection_body: bytes,
        rollout_lines: Sequence[str],
    ) -> ViewerCellPublicationRef:
        """Publish one cell outside every stable viewer source snapshot.

        The caller supplies already-canonically-serialized bodies, so the
        ledger commits exactly the bytes whose SHA-256 it records. The two
        files land in a private temporary directory, are fsynced, and become
        visible through one atomic directory rename.
        """
        with self._exclusive_viewer_source_lock():
            return self._write_viewer_publication_unlocked(
                cell_id=cell_id,
                env=env,
                projection_body=projection_body,
                rollout_lines=rollout_lines,
            )

    def _write_viewer_publication_unlocked(
        self,
        *,
        cell_id: str,
        env: str,
        projection_body: bytes,
        rollout_lines: Sequence[str],
    ) -> ViewerCellPublicationRef:
        """Atomically publish one immutable two-file viewer cell."""
        cell_filename = _canonical_cell_filename(cell_id, expected_env=env)
        path = self.viewer_cells_dir / cell_filename
        rollouts_body = "".join(rollout_lines).encode("utf-8")
        relative_parent = Path("viewer_cells") / cell_filename
        publication_ref = ViewerCellPublicationRef(
            projection=ViewerPublishedFileRef(
                relative_path=(relative_parent / "projection.json").as_posix(),
                sha256=hashlib.sha256(projection_body).hexdigest(),
            ),
            rollout_outputs=ViewerPublishedFileRef(
                relative_path=(
                    relative_parent / "rollout_outputs.jsonl"
                ).as_posix(),
                sha256=hashlib.sha256(rollouts_body).hexdigest(),
            ),
        )
        (
            parent_descriptor,
            root_descriptor,
            directory_descriptor,
            root_entry,
        ) = self._open_viewer_cells_dir()
        temporary: str | None = None
        try:
            fcntl.flock(directory_descriptor, fcntl.LOCK_EX)
            existing = self._read_viewer_publication(
                directory_descriptor,
                cell_filename,
                path,
            )
            if existing is not None:
                if existing != (projection_body, rollouts_body):
                    raise RuntimeError(
                        f"viewer publication conflicts at {path}"
                    )
                self._validate_root_binding(
                    parent_descriptor,
                    root_descriptor,
                    root_entry,
                )
                self._validate_viewer_cells_binding(
                    root_descriptor,
                    directory_descriptor,
                )
                return publication_ref
            for _ in range(128):
                candidate = f".{cell_filename}.{secrets.token_hex(16)}.tmp"
                try:
                    os.mkdir(
                        candidate,
                        mode=0o700,
                        dir_fd=directory_descriptor,
                    )
                except FileExistsError:
                    continue
                temporary = candidate
                break
            if temporary is None:
                raise RuntimeError(
                    "could not create a unique viewer publication directory"
                )
            temporary_descriptor = os.open(
                temporary,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_descriptor,
            )
            try:
                self._write_publication_file(
                    temporary_descriptor,
                    "projection.json",
                    projection_body,
                )
                self._write_publication_file(
                    temporary_descriptor,
                    "rollout_outputs.jsonl",
                    rollouts_body,
                )
                os.fsync(temporary_descriptor)
            finally:
                os.close(temporary_descriptor)
            try:
                os.rename(
                    temporary,
                    cell_filename,
                    src_dir_fd=directory_descriptor,
                    dst_dir_fd=directory_descriptor,
                )
                temporary = None
            except OSError:
                existing = self._read_viewer_publication(
                    directory_descriptor,
                    cell_filename,
                    path,
                )
                if existing != (projection_body, rollouts_body):
                    raise
            os.fsync(directory_descriptor)
            self._validate_root_binding(
                parent_descriptor,
                root_descriptor,
                root_entry,
            )
            self._validate_viewer_cells_binding(
                root_descriptor,
                directory_descriptor,
            )
            return publication_ref
        finally:
            if temporary is not None:
                self._cleanup_publication_temp(
                    directory_descriptor,
                    temporary,
                )
            os.close(directory_descriptor)
            os.close(root_descriptor)
            os.close(parent_descriptor)

    @property
    def optimization_traces_dir(self) -> Path:
        return self.root / "optimization_traces"

    def optimization_trace_path(self, cell_id: str) -> Path:
        """The per-cell optimizer-search trace artifact path.

        One JSON file per cell id under
        ``<root>/optimization_traces/``, with each ``:`` in the id mapped to
        ``__`` so the name is filesystem-safe. Holds the full per-step
        candidate evidence for human-readable reporting.
        """
        return (
            self.optimization_traces_dir
            / f"{_canonical_cell_filename(cell_id)}.json"
        )

    def write_optimization_trace(
        self, cell_id: str, trace: dict[str, object]
    ) -> Path:
        """Write the per-cell optimizer-search trace artifact.

        Overwrite by cell id: a re-run or resume of the same attempt supersedes
        its prior trace, and distinct attempts have distinct cell ids, so the
        store is append-safe across attempts. The returned path is recorded on
        the cell line so the trace is discoverable from the ledger. It is
        written even for incomplete-arm and halted cells, so a failed cell
        still leaves its partial search evidence.
        """
        self.optimization_traces_dir.mkdir(parents=True, exist_ok=True)
        path = self.optimization_trace_path(cell_id)
        path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
        return path

    def _append_jsonl_durable(self, filename: str, line: str) -> None:
        """Append one complete line under a file lock and fsync it."""
        absolute_root = Path(os.path.abspath(self.root))
        absolute_root.parent.mkdir(parents=True, exist_ok=True)
        root_entry = absolute_root.name or "."
        parent_descriptor = os.open(
            absolute_root.parent,
            os.O_RDONLY | os.O_DIRECTORY,
        )
        root_descriptor = -1
        file_descriptor = -1
        root_created = False
        file_created = False
        try:
            try:
                os.mkdir(root_entry, mode=0o755, dir_fd=parent_descriptor)
                root_created = True
            except FileExistsError:
                pass
            root_descriptor = os.open(
                root_entry,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_descriptor,
            )
            if root_created:
                os.fsync(parent_descriptor)
            try:
                file_descriptor = os.open(
                    filename,
                    os.O_WRONLY
                    | os.O_APPEND
                    | os.O_CREAT
                    | os.O_EXCL
                    | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=root_descriptor,
                )
                file_created = True
            except FileExistsError:
                file_descriptor = os.open(
                    filename,
                    os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW,
                    dir_fd=root_descriptor,
                )
            opened_stat = os.fstat(file_descriptor)
            if not stat.S_ISREG(opened_stat.st_mode):
                raise RuntimeError(
                    f"ledger file is not regular at {self.root / filename}"
                )
            fcntl.flock(file_descriptor, fcntl.LOCK_EX)
            body = (line + "\n").encode("utf-8")
            written = 0
            while written < len(body):
                written += os.write(file_descriptor, body[written:])
            os.fsync(file_descriptor)
            if file_created:
                os.fsync(root_descriptor)
        finally:
            if file_descriptor >= 0:
                os.close(file_descriptor)
            if root_descriptor >= 0:
                os.close(root_descriptor)
            os.close(parent_descriptor)

    def _read_jsonl_locked(self, filename: str) -> str:
        """Read a ledger snapshot while excluding in-progress appends."""
        absolute_root = Path(os.path.abspath(self.root))
        try:
            root_descriptor = os.open(
                absolute_root,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
        except FileNotFoundError:
            return ""
        file_descriptor = -1
        try:
            try:
                file_descriptor = os.open(
                    filename,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=root_descriptor,
                )
            except FileNotFoundError:
                return ""
            if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
                raise RuntimeError(
                    f"ledger file is not regular at {self.root / filename}"
                )
            fcntl.flock(file_descriptor, fcntl.LOCK_SH)
            with os.fdopen(file_descriptor, encoding="utf-8") as handle:
                file_descriptor = -1
                return handle.read()
        finally:
            if file_descriptor >= 0:
                os.close(file_descriptor)
            os.close(root_descriptor)

    def load(self) -> list[CellRecord]:
        """Parse the existing ``cells.jsonl``, validating every line."""
        self._cells = []
        for raw in self._read_jsonl_locked("cells.jsonl").splitlines():
            line = raw.strip()
            if not line:
                continue
            self._cells.append(CellRecord.from_line(line))
        self._loaded = True
        return list(self._cells)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def cells(self) -> list[CellRecord]:
        self._ensure_loaded()
        return list(self._cells)

    def completed_keys(self) -> set[tuple[str, str, int]]:
        """Keys whose *latest* record is terminal-completed; a resume skips.

        The ledger is append-only, so a key can carry several lines and the
        last one wins: a ``refinalize`` correction supersedes the line it
        corrects. Reading completion from any line rather than the latest one
        would let a superseded certified status keep a cell skipped after a
        correction demoted it to a non-terminal status, stranding the cell
        exactly where a rerun is the intended repair.
        """
        self._ensure_loaded()
        latest: dict[tuple[str, str, int], CellRecord] = {}
        for cell in self._cells:
            latest[cell.key()] = cell
        return {key for key, cell in latest.items() if cell.is_completed()}

    def is_completed(self, optimizer: str, env: str, attempt: int) -> bool:
        return cell_key(optimizer, env, attempt) in self.completed_keys()

    def for_attempt(
        self, optimizer: str, env: str, attempt: int
    ) -> CellRecord | None:
        """Return the latest record for one exact persisted attempt."""
        self._ensure_loaded()
        key = cell_key(optimizer, env, attempt)
        matches = [cell for cell in self._cells if cell.key() == key]
        return matches[-1] if matches else None

    def latest_for(self, optimizer: str, env: str) -> CellRecord | None:
        """The most recent recorded attempt for an (optimizer, env), if any."""
        self._ensure_loaded()
        matches = [
            cell
            for cell in self._cells
            if cell.optimizer == optimizer and cell.env == env
        ]
        return matches[-1] if matches else None

    def append_cell(self, record: CellRecord) -> None:
        """Durably append one validated cell commit marker."""
        validated = CellRecord.model_validate(
            record.model_dump(mode="json", by_alias=True)
        )
        with self._exclusive_viewer_source_lock():
            self._append_jsonl_durable("cells.jsonl", validated.to_line())
        self._cells = []
        self._loaded = False

    def append_spend(self, record: SpendRecord) -> None:
        """Durably append one validated spend snapshot line."""
        validated = SpendRecord.model_validate(
            record.model_dump(mode="json", by_alias=True)
        )
        self._append_jsonl_durable("spend.jsonl", validated.to_line())

    def spend_records(self) -> list[SpendRecord]:
        records: list[SpendRecord] = []
        for raw in self._read_jsonl_locked("spend.jsonl").splitlines():
            line = raw.strip()
            if line:
                records.append(SpendRecord.from_line(line))
        return records

    def total_spend_usd(self) -> float:
        """Sum of recorded per-cell ``spend_usd``."""
        self._ensure_loaded()
        return sum(cell.spend_usd for cell in self._cells)

    def spend_for_cell(self, cell_id: str) -> tuple[float, list[str]]:
        """Total credits consumed across all attempts of ``cell_id``.

        Sums the credits deltas over every recorded ``before`` snapshot for
        ``cell_id`` using the ``spend.jsonl`` before/after pairs, including
        crashed attempts. Credits (``remaining_usd``) are monotonically
        non-increasing and the log is append-only chronological.

        A cleanly-completed attempt is bounded by this cell's own next
        ``after`` snapshot, matched by ``cell_id`` and by that exact phase.
        Concurrent cells interleave their snapshots into one shared
        ``spend.jsonl``, so the record immediately following a cell's
        ``before`` is frequently a different cell's; pairing by ``cell_id`` is
        the only correct rule under interleaving. The cell's own
        ``checkpoint:<boundary>`` rows lie inside the pair and never close it.

        For a crashed attempt -- a ``before`` with no matching ``after`` before
        this cell's next ``before`` -- the spend is bounded by the next
        snapshot of any cell in file order, which captures the credits the
        crashed attempt burned before dying. A final trailing ``before`` with
        nothing after it cannot be bounded and is reported as a gap.

        Returns ``(total_usd, gaps)``: the summed spend and a list of
        human-readable notes for any unpairable snapshot.
        """
        records = self.spend_records()
        # Index snapshots that carry a usable remaining_usd, in file order.
        usable = [
            (index, record)
            for index, record in enumerate(records)
            if record.remaining_usd is not None
        ]
        total = 0.0
        gaps: list[str] = []
        for position, (index, record) in enumerate(usable):
            if record.cell_id != cell_id or record.phase != "before":
                continue
            before_remaining = record.remaining_usd
            if before_remaining is None:
                continue
            # Find this cell's own next ``after`` (a clean completion),
            # stopping if this cell's next ``before`` appears first: that
            # ``after`` belongs to a later attempt, and this ``before`` is a
            # crash.
            matched_after: SpendRecord | None = None
            crashed = False
            for _, candidate in usable[position + 1 :]:
                if candidate.cell_id != cell_id:
                    continue
                if candidate.phase == "before":
                    crashed = True
                    break
                # Only ``after`` closes the pair. This cell's own
                # ``checkpoint:<boundary>`` rows sit *between* its before and
                # its after by construction, so treating the first one as the
                # closing snapshot would stop the total at the first paid
                # boundary and silently omit every later one.
                if candidate.phase == "after":
                    matched_after = candidate
                    break
            if matched_after is not None:
                after_remaining = matched_after.remaining_usd
                if after_remaining is None:
                    continue
                delta = before_remaining - after_remaining
                if delta < 0:
                    gaps.append(
                        f"non-monotonic credits between records {index} and "
                        f"this cell's after (delta {delta:.4f}); skipped"
                    )
                    continue
                total += delta
                continue
            # No clean matching after: a crashed attempt, or last in file.
            # Bound it by the next usable snapshot of any cell in file order.
            if position + 1 >= len(usable):
                reason = (
                    "crashed or still running, last in file"
                    if crashed
                    else "no following snapshot to bound its spend"
                )
                gaps.append(
                    f"attempt with before snapshot at record {index} has no "
                    f"following snapshot to bound its spend ({reason}); "
                    "consumption unaccounted"
                )
                continue
            _, following = usable[position + 1]
            following_remaining = following.remaining_usd
            if following_remaining is None:
                continue
            delta = before_remaining - following_remaining
            if delta < 0:
                gaps.append(
                    f"non-monotonic credits between records {index} and the "
                    f"next snapshot (delta {delta:.4f}); skipped"
                )
                continue
            gaps.append(
                f"attempt with before snapshot at record {index} had no "
                f"matching after (crashed); bounded by the next snapshot "
                f"({following.cell_id}:{following.phase}) -> ${delta:.4f}"
            )
            total += delta
        return total, gaps
