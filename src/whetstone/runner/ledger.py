"""The validation ledgers: ``cells.jsonl`` and ``spend.jsonl``.

``cells.jsonl`` is the authoritative resumability ledger -- one line per cell
*attempt*, in the EXACT schema pinned by ``reports/validation-plan.md``
("cells.jsonl schema"). A completed ``(optimizer, env, attempt)`` cell is
skipped on resume; an interrupted cell is resumed (the optimizer harness owns
its optimization state) or restarted, recording which.

``spend.jsonl`` records the OpenRouter credits snapshot before/after each cell
(the ``GET /api/v1/credits`` numbers) so cumulative spend is auditable and the
budget guards (reserve + per-cell stop-loss) key off the persisted remaining.

Both ledgers are append-only JSONL. Schema validation is enforced on write
(a malformed cell line is refused, never silently appended).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

__all__ = [
    "CELL_STATUSES",
    "FULL_CONFIG_EVAL_HASH",
    "CellArtifacts",
    "CellModels",
    "CellRecord",
    "CellSamplingOverrides",
    "EnvOfficialCache",
    "Ledger",
    "SpendRecord",
    "cell_key",
]

#: The migration sentinel for :attr:`EnvOfficialCache.eval_config_hash`. Old
#: cache lines predate the eval-config-hash key field; they resolve to this
#: full-config sentinel, and :meth:`Ledger.env_cache_for` matches such lines
#: ONLY for a read that itself uses the full-config default (an
#: ``eval_config_hash=None`` lookup) -- a reduced-sampling read never matches
#: them. Mirrors the ``task_model`` nano-default migration.
FULL_CONFIG_EVAL_HASH = "__full_config__"

#: The closed set of cell statuses from the validation-plan schema.
#: ``inconclusive`` was added by the statistical-confidence upgrade: a positive
#: delta whose paired CI still spans 0 is inconclusive (not ``improved``).
#: ``incomplete-arm`` was added by the incomplete-official-arm fix: an official
#: arm (naive/best) whose aggregate never resolved (score None -- some rollouts
#: failed after the bounded re-drive) MUST NOT emit a headroom / no-headroom
#: determination or a terminal statistical status. The cell finalizes as
#: ``incomplete-arm`` carrying which arm/rows failed, and is NOT a certified
#: result (it is not a completed terminal status: a re-run supersedes it).
CELL_STATUSES: frozenset[str] = frozenset(
    {
        "improved",
        "inconclusive",
        "no-improvement",
        "plumbing-retry",
        "halted",
        "incomplete-arm",
    }
)


class CellModels(BaseModel):
    """The ``models: {task, proposer}`` sub-object."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task: StrictStr
    proposer: StrictStr


class CellSamplingOverrides(BaseModel):
    """The ``sampling_overrides`` sub-object of a cell record.

    Records the reduced-sampling overrides a cell was run under
    (``--official-n`` / ``--official-repeats``). ``None`` fields mean "spec
    default" (no
    override). Both fold into the composite Eval Config Identity Hash, so this
    is the auditable record of WHY a reduced cell has a distinct Eval Config
    identity (and got a cache MISS against the full-config entry). Absent field
    on an old line -> both None (a full-config cell).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    official_n: int | None = None
    official_repeats: int | None = None


class CellArtifacts(BaseModel):
    """The ``artifacts`` sub-object of a cell record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: The per-cell optimizer-search trace path (relative to the ledger root),
    #: e.g. ``optimization_traces/copro__c11__a0.json``. Historically this held
    #: the bare best-candidate id string; it now points at the on-disk trace
    #: artifact so the search is auditable. The bare id is preserved on
    #: :attr:`best_candidate_id` for backward compatibility.
    optimization_result_ref: StrictStr | None = None
    #: The accepted candidate's id (e.g. ``copro-p2`` or ``<env>-naive``) --
    #: the value ``optimization_result_ref`` carried before the trace artifact
    #: existed. Kept so existing readers of the accepted-candidate id still
    #: work.
    best_candidate_id: StrictStr | None = None
    official_record_before: StrictStr | None = None
    official_record_after: StrictStr | None = None


class CellRecord(BaseModel):
    """One ``cells.jsonl`` line -- the validation-plan schema + stats upgrade.

    Base fields mirror ``reports/validation-plan.md`` "cells.jsonl schema":
    ``{cell_id, optimizer, env, attempt, canonical, models, baseline_official,
    ceiling_official, best_official, delta, ci95, internal_evals_count,
    optimizer_steps, spend_usd, wall_s, lane, window_notes, status,
    artifacts}``.

    The "Statistical confidence" directive adds the bootstrap outputs:
    ``naive_ci95`` / ``ceiling_ci95`` (marginal task-bootstrap CIs),
    ``delta_ci95`` (paired best-naive, the interval behind the sharpened
    status), ``headroom_delta`` / ``headroom_ci95`` (paired ceiling-naive; the
    Eval-row headroom gate), ``official_repeats_used`` (repeats behind the
    official evals, raised on escalation), ``escalated`` (whether an
    inconclusive cell auto-doubled its repeats), and
    ``pooled_observation_counts`` (per-arm total 0/1 observations behind the
    reported per-task means, pre + post escalation pooled). ``ci95`` is kept as
    a back-compatible mirror of ``delta_ci95``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    cell_id: StrictStr
    optimizer: StrictStr
    env: StrictStr
    attempt: int
    canonical: bool
    models: CellModels
    baseline_official: float | None
    ceiling_official: float | None
    best_official: float | None
    delta: float | None
    ci95: tuple[float, float] | None
    naive_ci95: tuple[float, float] | None = None
    ceiling_ci95: tuple[float, float] | None = None
    delta_ci95: tuple[float, float] | None = None
    headroom_delta: float | None = None
    headroom_ci95: tuple[float, float] | None = None
    no_demonstrable_headroom: bool | None = None
    official_repeats_used: int = 0
    escalated: bool = False
    escalation_note: StrictStr = ""
    pooled_observation_counts: dict[str, int] = Field(default_factory=dict)
    internal_evals_count: int
    optimizer_steps: int
    spend_usd: float
    wall_s: float
    lane: StrictStr
    window_notes: StrictStr = ""
    status: StrictStr
    artifacts: CellArtifacts = Field(default_factory=CellArtifacts)
    #: The reduced-sampling overrides this cell ran under (``--official-n`` /
    #: ``--official-repeats``); both None = spec-default (full config). Extends
    #: the schema minimally; absent on old lines -> both None.
    sampling_overrides: CellSamplingOverrides = Field(
        default_factory=CellSamplingOverrides
    )

    @model_validator(mode="after")
    def _validate(self) -> CellRecord:
        if self.status not in CELL_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(CELL_STATUSES)}; "
                f"got {self.status!r}"
            )
        for name in ("ci95", "naive_ci95", "ceiling_ci95", "delta_ci95",
                     "headroom_ci95"):
            value = getattr(self, name)
            if value is not None and len(value) != 2:
                raise ValueError(f"{name} must be a (low, high) pair or null")
        return self

    def key(self) -> tuple[str, str, int]:
        return cell_key(self.optimizer, self.env, self.attempt)

    def is_completed(self) -> bool:
        """A completed cell is any non-retry terminal status."""
        return self.status in {
            "improved",
            "inconclusive",
            "no-improvement",
            "halted",
        }

    def to_line(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True)


class EnvOfficialCache(BaseModel):
    """The per-(env, task-model) Eval-row official cache: scores + vectors.

    WF5A cached the ceiling scalar so later optimizer cells reuse it instead of
    re-driving the ceiling probe. The statistical-confidence upgrade extends
    that caching to the per-task score VECTORS (aligned in official-instance
    order) for both the naive and ceiling arms, so a later optimizer cell can
    compute the paired ``best - naive`` delta CI (and reuse the marginal naive
    CI) WITHOUT re-driving the naive baseline. Established once per (env,
    task-model) by the Eval-row (``optimizer=eval``) cell.

    The cache is keyed by ``(env, task_model, eval_config_hash)``. The task
    model folds into the Provider Call Config identity (graph_hash), so
    naive/ceiling vectors measured under one task model (e.g.
    ``openai/gpt-5-nano``) are NOT comparable to a candidate measured under a
    different task model (e.g. ``deepseek/deepseek-v4-flash``) (FIX 7). The
    ``eval_config_hash`` is the composite Eval Config Identity Hash of the
    OFFICIAL split -- it folds in the official Task Set (ordering + membership)
    and the Repeat Plan repeat count -- so vectors measured under a reduced
    sampling (fewer official-n, or different official-repeats -> a different
    ``eval_config_hash``) are NOT comparable to the full-config vectors and get
    a cache MISS. A reduced-sampling cell (e.g. c23 with ``--official-n`` /
    ``--official-repeats``) thus drives its own naive/ceiling arms rather than
    pairing against the full-config entry for the same ``(env, task_model)``.

    Migration defaults: ``task_model`` defaults to the canonical nano slug so
    pre-FIX-7 cache lines (no task-model field) resolve to the nano key.
    ``eval_config_hash`` defaults to the sentinel :data:`FULL_CONFIG_EVAL_HASH`
    so old cache lines (no eval-config-hash field) resolve to the FULL-config
    identity -- and, per :meth:`Ledger.env_cache_for`, they are matched ONLY by
    a read that itself requests the full-config default (an
    ``eval_config_hash=None`` lookup), never by a reduced-sampling read.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    env: StrictStr
    naive_official: float | None
    ceiling_official: float | None
    naive_per_task: tuple[float, ...]
    ceiling_per_task: tuple[float, ...]
    official_repeats_used: int
    #: The task model the cached vectors were measured under. Part of the cache
    #: key: a cell with a different task model gets a cache MISS and drives its
    #: own naive/ceiling arms. Defaults to the canonical nano slug so old cache
    #: lines (no field) key to nano.
    task_model: StrictStr = "openai/gpt-5-nano"
    #: The composite Eval Config Identity Hash (official split) the cached
    #: vectors were measured under. Part of the cache key: a cell with a
    #: different official Eval Config identity (reduced official-n / repeats ->
    #: a different hash) gets a cache MISS and drives its own arms. Defaults to
    #: the full-config sentinel so old cache lines (no field) resolve to the
    #: full-config identity, matched only by full-config (default) reads.
    eval_config_hash: StrictStr = FULL_CONFIG_EVAL_HASH

    def to_line(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True)


class SpendRecord(BaseModel):
    """One ``spend.jsonl`` line: an OpenRouter credits snapshot pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cell_id: StrictStr
    phase: StrictStr  # "before" | "after"
    lane: StrictStr
    total_credits: float | None = None
    total_usage: float | None = None
    remaining_usd: float | None = None
    at: StrictStr = ""

    def to_line(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True)


def cell_key(optimizer: str, env: str, attempt: int) -> tuple[str, str, int]:
    """The resumability key for a cell attempt."""
    return (optimizer, env, attempt)


@dataclass(slots=True)
class Ledger:
    """Append-only cells + spend JSONL ledgers rooted at a directory.

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
    def env_cache_path(self) -> Path:
        return self.root / "env_official_cache.jsonl"

    @property
    def optimization_traces_dir(self) -> Path:
        return self.root / "optimization_traces"

    def optimization_trace_path(self, cell_id: str) -> Path:
        """The per-cell optimizer-search trace artifact path.

        One JSON file per cell id
        (``<root>/optimization_traces/<cell_id>.json``, with any ``:`` in the
        id -- ``optimizer:env:aN`` -- mapped to ``__`` so the name is
        filesystem-safe). Holds the full per-round candidate evidence the
        in-memory ``OptimizeResult`` would otherwise drop.
        """
        safe = cell_id.replace(":", "__")
        return self.optimization_traces_dir / f"{safe}.json"

    def write_optimization_trace(
        self, cell_id: str, trace: dict[str, object]
    ) -> Path:
        """Write (overwrite) the per-cell optimizer-search trace artifact.

        Overwrite-by-cell-id (a re-run/resume of the SAME attempt supersedes
        its prior trace; distinct attempts have distinct cell ids, so the store
        is append-safe across attempts). Returned path is recorded on the cell
        line's ``artifacts.optimization_result_ref`` so the trace is
        discoverable from the ledger. Written even for incomplete-arm/halted
        cells so a failed cell still leaves its (partial) search evidence.
        """
        self.optimization_traces_dir.mkdir(parents=True, exist_ok=True)
        path = self.optimization_trace_path(cell_id)
        path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
        return path

    def load(self) -> list[CellRecord]:
        """Parse the existing ``cells.jsonl`` (validating every line)."""
        self._cells = []
        if self.cells_path.exists():
            for raw in self.cells_path.read_text().splitlines():
                line = raw.strip()
                if not line:
                    continue
                self._cells.append(
                    CellRecord.model_validate_json(line)
                )
        self._loaded = True
        return list(self._cells)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def cells(self) -> list[CellRecord]:
        self._ensure_loaded()
        return list(self._cells)

    def completed_keys(self) -> set[tuple[str, str, int]]:
        """Keys of cells that are terminal-completed (skip on resume)."""
        self._ensure_loaded()
        return {c.key() for c in self._cells if c.is_completed()}

    def is_completed(
        self, optimizer: str, env: str, attempt: int
    ) -> bool:
        return cell_key(optimizer, env, attempt) in self.completed_keys()

    def latest_for(
        self, optimizer: str, env: str
    ) -> CellRecord | None:
        """The most recent recorded attempt for an (optimizer, env), if any."""
        self._ensure_loaded()
        matches = [
            c for c in self._cells if c.optimizer == optimizer and c.env == env
        ]
        return matches[-1] if matches else None

    def ceiling_for(self, env: str) -> float | None:
        """The cached ceiling_official for an env (evaluated once per env)."""
        self._ensure_loaded()
        for c in self._cells:
            if c.env == env and c.ceiling_official is not None:
                return c.ceiling_official
        return None

    def env_cache_for(
        self,
        env: str,
        *,
        task_model: str | None = None,
        eval_config_hash: str | None = None,
        default_config: bool = False,
    ) -> EnvOfficialCache | None:
        """Cached Eval-row scores + vectors for (env, task_model, eval hash).

        Returns the most recent cache line matching ``env`` AND -- when given
        -- the SAME ``task_model`` (FIX 7) AND the SAME ``eval_config_hash``
        (the composite Eval Config Identity Hash of the official split). Both
        fold into the identity of the cached vectors:

        * A differing task model is a cache MISS (a deepseek cell must never
          reuse cached nano vectors -- different graph identity).
        * A differing ``eval_config_hash`` is a cache MISS (a reduced-sampling
          cell -- fewer official-n or different official-repeats -> a different
          hash -- must never pair against the full-config vectors).

        On a MISS the method returns ``None`` and the caller drives + caches
        its own naive/ceiling arms.

        Migration default: OLD cache lines predate the eval-config-hash field
        and carry the full-config sentinel (:data:`FULL_CONFIG_EVAL_HASH`).
        They are matchable ONLY by a read that itself uses the DEFAULT (full)
        config -- pass ``default_config=True`` (the runner sets this when the
        cell has no sampling overrides). Under that flag a stored sentinel line
        matches when its OTHER key fields (env, task model) match, regardless
        of the requested ``eval_config_hash``; a reduced-sampling read
        (``default_config=False``) never matches a sentinel line. Lines written
        with a concrete hash always match by exact ``eval_config_hash``.
        ``task_model=None`` matches on env only (back-compat); an
        ``eval_config_hash=None`` read only compares the hash when a concrete
        value is supplied.
        """
        if not self.env_cache_path.exists():
            return None
        latest: EnvOfficialCache | None = None
        for raw in self.env_cache_path.read_text().splitlines():
            line = raw.strip()
            if not line:
                continue
            record = EnvOfficialCache.model_validate_json(line)
            if record.env != env:
                continue
            if task_model is not None and record.task_model != task_model:
                continue
            is_sentinel = record.eval_config_hash == FULL_CONFIG_EVAL_HASH
            if is_sentinel:
                # Old (fieldless) line: only a default-config read may match.
                if not default_config:
                    continue
            elif (
                eval_config_hash is not None
                and record.eval_config_hash != eval_config_hash
            ):
                continue
            latest = record
        return latest

    def append_env_cache(self, record: EnvOfficialCache) -> None:
        """Append one per-env official cache line (Eval row establishes it)."""
        self.root.mkdir(parents=True, exist_ok=True)
        with self.env_cache_path.open("a") as handle:
            handle.write(record.to_line() + "\n")

    def append_cell(self, record: CellRecord) -> None:
        """Append one validated cell line (creating the ledger if needed)."""
        self._ensure_loaded()
        self.root.mkdir(parents=True, exist_ok=True)
        with self.cells_path.open("a") as handle:
            handle.write(record.to_line() + "\n")
        self._cells.append(record)

    def append_spend(self, record: SpendRecord) -> None:
        """Append one validated spend snapshot line."""
        self.root.mkdir(parents=True, exist_ok=True)
        with self.spend_path.open("a") as handle:
            handle.write(record.to_line() + "\n")

    def spend_records(self) -> list[SpendRecord]:
        if not self.spend_path.exists():
            return []
        records: list[SpendRecord] = []
        for raw in self.spend_path.read_text().splitlines():
            line = raw.strip()
            if line:
                records.append(SpendRecord.model_validate_json(line))
        return records

    def total_spend_usd(self) -> float:
        """Sum of recorded per-cell ``spend_usd`` (cumulative)."""
        self._ensure_loaded()
        return sum(c.spend_usd for c in self._cells)

    def spend_for_cell(self, cell_id: str) -> tuple[float, list[str]]:
        """Total credits consumed across ALL attempts of ``cell_id``.

        Sums the credits deltas over every recorded ``before`` snapshot for
        ``cell_id`` using the ``spend.jsonl`` before/after pairs, INCLUDING
        crashed attempts. Credits (``remaining_usd``) are monotonically
        non-increasing and the log is append-only chronological.

        A cleanly-completed attempt is bounded by THIS CELL'S OWN next
        ``after`` snapshot (matched by ``cell_id``), not just the following
        record. Concurrent cells interleave their before/after snapshots into
        one shared ``spend.jsonl`` (e.g. the c18:a1 defect: the record right
        after c18:a1's ``before`` was a DIFFERENT cell's ``before``, so the
        old "next record" heuristic mis-bounded the spend to $0.00 while the
        true ``eval:c18:a1:after`` sat two records later). Pairing by cell_id
        is correct under interleaving.

        For a CRASHED attempt (a ``before`` with no matching ``after`` before
        this cell's next ``before``) the spend is bounded by the NEXT snapshot
        of any cell in file order, which captures the credits the crashed
        attempt burned before dying. A final trailing ``before`` with nothing
        after it cannot be bounded and is reported as a gap.

        Returns ``(total_usd, gaps)``: the summed spend and a list of
        human-readable notes for any unpairable snapshot.
        """
        records = self.spend_records()
        # Index snapshots that carry a usable remaining_usd, in file order.
        usable = [
            (i, r)
            for i, r in enumerate(records)
            if r.remaining_usd is not None
        ]
        total = 0.0
        gaps: list[str] = []
        for pos, (idx, rec) in enumerate(usable):
            if rec.cell_id != cell_id or rec.phase != "before":
                continue
            assert rec.remaining_usd is not None
            # Find THIS cell's own next ``after`` (clean completion), stopping
            # if this cell's NEXT ``before`` appears first (that after belongs
            # to a later attempt, and this before is a crash).
            matched_after: SpendRecord | None = None
            crashed = False
            for _, cand in usable[pos + 1:]:
                if cand.cell_id == cell_id and cand.phase == "before":
                    crashed = True
                    break
                if cand.cell_id == cell_id and cand.phase == "after":
                    matched_after = cand
                    break
            if matched_after is not None:
                assert matched_after.remaining_usd is not None
                delta = rec.remaining_usd - matched_after.remaining_usd
                if delta < 0:
                    gaps.append(
                        f"non-monotonic credits between records {idx} and "
                        f"this cell's after (delta {delta:.4f}); skipped"
                    )
                    continue
                total += delta
                continue
            # No clean matching after -> a crashed attempt (or last-in-file).
            # Bound it by the next usable snapshot of any cell in file order.
            if pos + 1 >= len(usable):
                _reason = "crashed/running last-in-file" if crashed else (
                    "no following snapshot to bound its spend"
                )
                gaps.append(
                    f"attempt with before snapshot at record {idx} has no "
                    f"following snapshot to bound its spend ({_reason}); "
                    "consumption unaccounted"
                )
                continue
            _, nxt = usable[pos + 1]
            assert nxt.remaining_usd is not None
            delta = rec.remaining_usd - nxt.remaining_usd
            if delta < 0:
                gaps.append(
                    f"non-monotonic credits between records {idx} and the "
                    f"next snapshot (delta {delta:.4f}); skipped"
                )
                continue
            gaps.append(
                f"attempt with before snapshot at record {idx} had no "
                f"matching after (crashed); bounded by the next snapshot "
                f"({nxt.cell_id}:{nxt.phase}) -> ${delta:.4f}"
            )
            total += delta
        return total, gaps
