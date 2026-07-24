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

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.optimization import TypedRef

__all__ = [
    "CELLS_SCHEMA",
    "CELL_STATUSES",
    "FULL_CONFIG_EVAL_HASH",
    "ROLLOUT_OUTPUTS_SCHEMA",
    "SPEND_SCHEMA",
    "CellArtifacts",
    "CellAttractor",
    "CellControls",
    "CellDualScores",
    "CellModels",
    "CellRecord",
    "CellSamplingOverrides",
    "CellTelemetry",
    "EnvOfficialCache",
    "Ledger",
    "PromptCacheControls",
    "SpendRecord",
    "cell_key",
]

#: Versioned schema stamps on each ledger artifact type's rows (task 26 item 9;
#: the ``power_analysis/v1`` + ``events/v1`` precedent). A structured reader
#: branches on the stamp instead of sniffing which keys happen to be present.
CELLS_SCHEMA = "whetstone.runner.cells/v1"
SPEND_SCHEMA = "whetstone.runner.spend/v1"
ROLLOUT_OUTPUTS_SCHEMA = "whetstone.runner.rollout_outputs/v1"

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
#: ``proposer-failure`` was added by the failed-draft fix: EVERY draft in the
#: run was a typed proposer-draft failure (timeout/nonzero/empty/rejected) so
#: no real candidate was ever explored. This is NOT an honest no-improvement
#: (where real candidates WERE scored) -- it is a proposer outage, and (like
#: ``incomplete-arm``) not a completed status (a re-run supersedes it).
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


class CellTelemetry(BaseModel):
    """Per-cell usage + latency totals (task 20), summed over the partial log.

    Task-side rollout calls only. Every field is coverage-honest: a total is
    ``None`` when NO row reported it (never a fake 0); ``*_coverage`` counts
    the rows over which the total was actually summed, so a partial-coverage
    cell (mixed pre/post-telemetry rows) is never mistaken for a full one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_tokens: int | None = None
    total_reasoning_tokens: int | None = None
    total_latency_s: float | None = None
    mean_latency_s: float | None = None
    token_coverage: int = 0
    reasoning_coverage: int = 0
    latency_coverage: int = 0


class CellControls(BaseModel):
    """The literal sampling-control VALUES a cell ran under (task 26 item 5).

    ``temperature`` / ``reasoning_effort`` fold into the Provider Call Config
    identity (graph_hash), so today two cells at different temperatures produce
    identical-looking ledger lines and you must re-derive the value from a hash
    that (for anchors) is not even stored. Recording the literal values here
    makes "did this anchor run at temp 0 or temp 1?" answerable by reading the
    line. ``None`` means the control was UNSET (provider default) -- never
    conflated with an explicit 0.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature: float | None = None
    reasoning_effort: StrictStr | None = None
    #: Task 31: the RECORDING-only prompt-cache marker for this cell. ``None``
    #: when the opt-in ``--prompt-cache`` flag was OFF (the strict default), so
    #: a run without it is byte-identical. It NEVER participates in identity
    #: (like the sampling values above are recording-only mirrors) -- it merely
    #: records that the cache was on and the hit/miss/store counters observed.
    prompt_cache: PromptCacheControls | None = None


class PromptCacheControls(BaseModel):
    """Recording-only prompt-cache telemetry for a cell (task 31).

    Present ONLY when ``--prompt-cache`` was on. ``enabled`` is always True
    when present; ``hits`` / ``misses`` / ``stores`` are the run-scoped store's
    counters observed by this cell (a hit reused a stored Result; a miss drove
    the transport then stored it). RECORDING-only: never folds into any hash.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    hits: int = 0
    misses: int = 0
    stores: int = 0


class CellArtifacts(BaseModel):
    """Typed canonical records and human-readable reporting projections."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    optimization_result_ref: TypedRef | None = None
    optimization_trace_ref: StrictStr | None = None
    best_candidate_id: StrictStr | None = None
    official_record_before: TypedRef | None = None
    official_record_after: TypedRef | None = None
    #: The per-cell power-analysis artifact path (relative to the ledger root),
    #: e.g. ``power_analysis/copro__c22__a0.json``. ``None`` when the opt-in
    #: power stage did not run (the strict default).
    power_analysis_ref: StrictStr | None = None


class PowerSizing(BaseModel):
    """The power stage's recommended-vs-used internal-eval sizing.

    Present on the cell line ONLY when the opt-in power stage ran; ``None``
    otherwise (so a run without the stage is byte-identical). Records BOTH the
    recommendation (from the power analysis) and the AS-USED sizes (clamped to
    the pool), so a later reader sees exactly what was recommended and what was
    driven.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    recommended_n_tasks: StrictInt
    recommended_repeats: StrictInt
    used_n_tasks: StrictInt
    used_repeats: StrictInt
    pool_ceiling: StrictInt
    achievable: StrictBool
    pool_limited: StrictBool
    target_gap: float
    achieved_mdd: float


class CellDualScores(BaseModel):
    """The ed1 SECOND objective (Mean Compression Ratio) per official arm.

    Present ONLY for the ed1 enc-dec env (``None`` for QA envs, which have one
    objective). The primary ``baseline_official`` / ``best_official`` /
    ``ceiling_official`` fields carry the pass-rate (the reward-bearing
    metric);
    these carry the compression scalars REPORTED alongside. Full dual-objective
    /
    Pareto selection is a flagged follow-up -- this is reporting, not
    selection.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    naive_compression: float | None = None
    ceiling_compression: float | None = None
    best_compression: float | None = None
    budget_ratio: float | None = None
    dataset_revision: StrictStr | None = None


class CellAttractor(BaseModel):
    """The ed1m REPORTED attractor-pull measurement, carried on the cell line.

    ed1m's dual oracle reports ``mean_attractor_pull`` -- the fraction of the
    DISCRIMINATING inputs whose reconstruction snapped to the CANONICAL
    behavior (the contamination signal, NEVER a reward objective). Task 28
    item 1: this was previously written ONLY to an ed1 aggregate artifact that
    evaporated when no persistent store was wired, so it vanished from the
    ledger even though ``dual_scores`` (compression) survived. Recording it
    here alongside ``dual_scores`` makes the measurement durable per cell.
    RECORDING-only -- it does NOT enter any identity hash.

    Present ONLY for ed1m cells (``None`` for ed1 / QA / d1). ``mean`` is the
    reported scalar (mean over the tasks that had a discriminating sample);
    ``per_task`` is the aligned per-official-task vector (``None`` for a task
    whose mutant had no discriminating inputs, or an unscored task);
    ``sampled_task_count`` is how many official tasks contributed a non-null
    attractor value (the mean's denominator). A ``mean`` of ``None`` on a
    non-null record means ed1m ran but no task produced a discriminating
    sample.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mean: float | None = None
    per_task: tuple[float | None, ...] = ()
    sampled_task_count: int = 0


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

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True
    )

    #: Versioned schema stamp (:data:`CELLS_SCHEMA`); serialized as ``schema``
    #: (aliased -- ``schema`` shadows a BaseModel method). Absent on old lines
    #: -> ``None`` (a reader branches on it; task 26).
    schema_: StrictStr | None = Field(default=CELLS_SCHEMA, alias="schema")
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
    #: Per-cell task-side usage + latency totals (task 20); coverage-honest.
    #: Absent on pre-telemetry lines -> the empty default (all totals None).
    telemetry: CellTelemetry = Field(default_factory=CellTelemetry)
    #: The opt-in power stage's recommended-vs-used internal sizing; ``None``
    #: when the power stage did not run (a run without it is byte-identical).
    power_sizing: PowerSizing | None = None
    #: The ed1 enc-dec SECOND objective (Mean Compression Ratio per arm) +
    #: budget/dataset provenance; ``None`` for QA envs (single objective).
    dual_scores: CellDualScores | None = None
    #: The ed1m REPORTED attractor-pull measurement (task 28 item 1), carried
    #: on the line so it no longer depends on a persistent ObjectStore being
    #: wired. ``None`` for ed1 / QA / d1 (only ed1m measures attractor pull).
    attractor: CellAttractor | None = None
    #: The content-addressed identity of the exact resolved graph the OFFICIAL
    #: arm ran under (task 26 item 4): recorded on EVERY cell including anchors
    #: (eval rows), which previously persisted no graph/eval-config identity at
    #: all. RECORDING-only -- this is the hash the runner already computes, now
    #: written to the line. ``None`` on old lines.
    graph_hash: StrictStr | None = None
    #: The composite Eval Config Identity Hash of the official split (task 26
    #: item 4), recorded on every cell incl. anchors. ``None`` on old lines.
    eval_config_hash: StrictStr | None = None
    #: The literal sampling-control values (temperature / reasoning_effort) the
    #: cell ran under (task 26 item 5); defaults to all-``None`` (controls
    #: unset -> provider default).
    controls: CellControls = Field(default_factory=CellControls)
    #: ISO-8601 UTC wall-clock the cell started / finished (task 26 item 1).
    #: The line already carries ``wall_s`` (a duration) but no absolute
    #: timestamps, so concurrency interleaving was unreconstructable. ``None``
    #: on old lines (never captured) -- never a populated-but-empty string.
    started_at: StrictStr | None = None
    finished_at: StrictStr | None = None

    @model_validator(mode="after")
    def _validate(self) -> CellRecord:
        if self.status not in CELL_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(CELL_STATUSES)}; "
                f"got {self.status!r}"
            )
        for name in (
            "ci95",
            "naive_ci95",
            "ceiling_ci95",
            "delta_ci95",
            "headroom_ci95",
        ):
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
        return json.dumps(
            self.model_dump(mode="json", by_alias=True), sort_keys=True
        )

    @classmethod
    def from_line(cls, line: str) -> CellRecord:
        return cls.model_validate_json(line)


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
    """One ``spend.jsonl`` line: an OpenRouter credits snapshot pair.

    Task 26: ``at`` is a REAL ISO-8601 UTC wall-clock (it was the empty string
    on every historical row -- a populated-but-empty field that read as
    "recorded but blank"); it is ``None`` when genuinely never captured
    (null-honesty), never ``""``. ``event_id`` is a per-row unique id so a
    spend timeline can address individual snapshots. ``schema`` version-stamps
    the row.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True
    )

    schema_: StrictStr | None = Field(default=SPEND_SCHEMA, alias="schema")
    event_id: StrictStr | None = None
    cell_id: StrictStr
    phase: StrictStr  # "before" | "after"
    lane: StrictStr
    total_credits: float | None = None
    total_usage: float | None = None
    remaining_usd: float | None = None
    at: StrictStr | None = None

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
        filesystem-safe). Holds the full per-step candidate evidence for
        human-readable reporting.
        """
        safe = cell_id.replace(":", "__")
        return self.optimization_traces_dir / f"{safe}.json"

    @property
    def rollout_outputs_dir(self) -> Path:
        return self.root / "rollout_outputs"

    def rollout_outputs_path(self, cell_id: str) -> Path:
        """The per-cell rollout-output sidecar path (``:`` -> ``__``)."""
        safe = cell_id.replace(":", "__")
        return self.rollout_outputs_dir / f"{safe}.jsonl"

    def write_rollout_outputs(
        self, cell_id: str, rows: list[dict[str, object]]
    ) -> Path:
        """Write (overwrite) the per-cell rollout-output sidecar (JSONL).

        One line per driven rollout row across EVERY internal candidate eval,
        EVERY official arm: ``split_role``, ``candidate_id``, ``instance_id``,
        ``repeat``, the FULL untruncated ``output_text``, the extracted 0/1
        ``score``, and any ``failure_code``. Kept SEPARATE from the trace (c23
        streams are long) and one-command-readable (plain JSONL). Additive
        logging -- no behavior change.
        """
        self.rollout_outputs_dir.mkdir(parents=True, exist_ok=True)
        path = self.rollout_outputs_path(cell_id)
        with path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return path

    @property
    def power_analysis_dir(self) -> Path:
        return self.root / "power_analysis"

    def power_analysis_path(self, cell_id: str) -> Path:
        """The per-cell power-analysis artifact path (``:`` -> ``__``)."""
        safe = cell_id.replace(":", "__")
        return self.power_analysis_dir / f"{safe}.json"

    def write_power_analysis(
        self, cell_id: str, artifact: dict[str, object]
    ) -> Path:
        """Write (overwrite) the per-cell ``power_analysis`` artifact.

        One JSON per cell id under ``<root>/power_analysis/``. Holds the full
        (n x r) MDD surface, the variance decomposition, alpha/target/seed, the
        certified headroom used, the recommended n_tasks/repeats +
        achievability verdict, and the pool ceiling -- everything needed to
        re-derive or deepen the analysis later. The path is recorded on the
        cell line so the artifact is discoverable from the ledger.
        """
        self.power_analysis_dir.mkdir(parents=True, exist_ok=True)
        path = self.power_analysis_path(cell_id)
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        return path

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
        """Keys of cells that are terminal-completed (skip on resume)."""
        self._ensure_loaded()
        return {c.key() for c in self._cells if c.is_completed()}

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
                records.append(SpendRecord.from_line(line))
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
            for _, cand in usable[pos + 1 :]:
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
                _reason = (
                    "crashed/running last-in-file"
                    if crashed
                    else ("no following snapshot to bound its spend")
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
