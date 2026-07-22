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
    "CellArtifacts",
    "CellModels",
    "CellRecord",
    "EnvOfficialCache",
    "Ledger",
    "SpendRecord",
    "cell_key",
]

#: The closed set of cell statuses from the validation-plan schema.
#: ``inconclusive`` was added by the statistical-confidence upgrade: a positive
#: delta whose paired CI still spans 0 is inconclusive (not ``improved``).
CELL_STATUSES: frozenset[str] = frozenset(
    {
        "improved",
        "inconclusive",
        "no-improvement",
        "plumbing-retry",
        "halted",
    }
)


class CellModels(BaseModel):
    """The ``models: {task, proposer}`` sub-object."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task: StrictStr
    proposer: StrictStr


class CellArtifacts(BaseModel):
    """The ``artifacts`` sub-object of a cell record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    optimization_result_ref: StrictStr | None = None
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

    The cache is keyed by ``(env, task_model)`` (FIX 7): the task model folds
    into the Provider Call Config identity (graph_hash), so naive/ceiling
    vectors measured under one task model (e.g. ``openai/gpt-5-nano``) are NOT
    comparable to a candidate measured under a different task model (e.g.
    ``deepseek/deepseek-v4-flash``). A deepseek cell must never pair against
    cached nano vectors. ``task_model`` defaults to the canonical nano slug so
    pre-FIX-7 cache lines (which had no task-model field) resolve to the nano
    key.
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
        self, env: str, *, task_model: str | None = None
    ) -> EnvOfficialCache | None:
        """The cached Eval-row official scores + vectors for (env, task_model).

        Returns the most recent cache line matching ``env`` AND -- when
        ``task_model`` is given -- the SAME task model (FIX 7). The task model
        folds into the graph identity, so a deepseek cell must never reuse
        cached nano vectors: a differing task model is a cache MISS (returns
        ``None``), and the caller drives + caches its own naive/ceiling arms.
        ``task_model=None`` matches on env only (back-compat for callers that
        don't distinguish). ``None`` overall when no matching Eval row has run.
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
        non-increasing and the log is append-only chronological, so each
        ``before`` snapshot's consumption is ``before.remaining -
        next_snapshot.remaining`` where ``next_snapshot`` is the immediately
        following record in the file. For a cleanly-completed attempt that next
        record is this cell's own ``after``; for a CRASHED attempt (a
        ``before`` with no matching ``after``) it is the NEXT snapshot of any
        cell, which captures the credits the crashed attempt burned before
        dying. The final trailing ``before`` with nothing after it cannot be
        bounded and is reported as a gap.

        Returns ``(total_usd, gaps)``: the summed spend and a list of
        human-readable notes for any unpairable snapshot (e.g. a still-running
        or last-in-file crashed attempt with no following snapshot).
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
            if pos + 1 >= len(usable):
                gaps.append(
                    f"attempt with before snapshot at record {idx} has no "
                    "following snapshot to bound its spend (crashed/running "
                    "last-in-file); consumption unaccounted"
                )
                continue
            _, nxt = usable[pos + 1]
            assert rec.remaining_usd is not None
            assert nxt.remaining_usd is not None
            delta = rec.remaining_usd - nxt.remaining_usd
            if delta < 0:
                # Credits should not increase; a negative delta means a
                # top-up or reordering -- record it as a gap, contribute 0.
                gaps.append(
                    f"non-monotonic credits between records {idx} and the "
                    f"next snapshot (delta {delta:.4f}); skipped"
                )
                continue
            if nxt.cell_id != cell_id or nxt.phase != "after":
                gaps.append(
                    f"attempt with before snapshot at record {idx} had no "
                    f"matching after (crashed); bounded by the next snapshot "
                    f"({nxt.cell_id}:{nxt.phase}) -> ${delta:.4f}"
                )
            total += delta
        return total, gaps
