"""Offline attractor-pull recompute for a persisted ed1m rollout sidecar.

Task 28 item 4 -- the recovery utility for item 1's gap: the ed1m REPORTED
``mean_attractor_pull`` was, for already-run cells, only ever written to an
aggregate artifact that evaporated when no persistent ObjectStore was wired.
This tool makes it recoverable for those cells WITHOUT re-running the model: it
reads a persisted ``rollout_outputs`` jsonl (each row carries ``output_text`` =
``ENCODER:\\n...\\n\\nDECODER:\\n...`` + a per-task ``instance_id``, task 26),
extracts the DECODER reconstruction, loads the mutant dataset directly
(:mod:`whetstone.envs.ed1m_dataset` reads ``mutants.jsonl``), and REPLAYS the
dual oracle OFFLINE (local subprocess execution, the same no-Docker path the
dry-run scorer uses -- NO LLM calls). It then reports ``mean_attractor_pull``
+ the per-task attractor values, matching the aggregate the live run would have
recorded.

The reported mean is computed exactly as the live aggregate did (mean over the
tasks that produced a discriminating sample; a null per-task value -- a mutant
with no discriminating inputs, or an unscored task -- is never folded in as a
zero). By default the reported arm is ``official_best`` (the arm task 28 item 1
records on the cell line); every arm present is broken out for transparency.

Read-only: it never writes to the artifact it reads. A ``--out`` path (or the
:func:`write_result` helper) persists the computed record to a SEPARATE file.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from whetstone.envs.ed1m_dataset import Ed1mMutant, load_ed1m_mutants
from whetstone.envs.ed1m_oracle import score_ed1m_reconstruction

#: The sidecar output-text marker separating the encoder blob from the decoder
#: reconstruction (``ed1_eval._row_output_text`` / the screen writer emit it).
_DECODER_MARKER = "\n\nDECODER:\n"

#: The default reported arm -- the one task 28 item 1 records on the cell line.
DEFAULT_REPORTED_ROLE = "official_best"


def extract_decoder_text(output_text: str | None) -> str | None:
    """The DECODER reconstruction from a sidecar ``output_text`` blob.

    The blob is ``ENCODER:\\n<enc>\\n\\nDECODER:\\n<dec>`` (both halves kept).
    The oracle scored the raw decoder generation ``<dec>``, so recovering the
    text after the marker reproduces the exact reconstruction that was scored.
    Returns ``None`` when the text is absent or has no decoder section (e.g. a
    row that failed before the decoder call, or a restored row with no text).
    """
    if output_text is None:
        return None
    idx = output_text.find(_DECODER_MARKER)
    if idx < 0:
        return None
    return output_text[idx + len(_DECODER_MARKER):]


@dataclass(frozen=True, slots=True)
class ArmAttractor:
    """One arm's recomputed attractor pull (mean + per-task, offline)."""

    split_role: str
    #: The REPORTED mean attractor pull: mean over tasks with a non-null value
    #: (a null task -- no discriminating sample -- is excluded, never zeroed).
    #: ``None`` when no task produced a discriminating sample.
    mean_attractor_pull: float | None
    #: Per-task attractor pull keyed by ``instance_id`` (task's mean over its
    #: scored repeats); ``None`` for a task with no discriminating sample or no
    #: scorable reconstruction.
    per_task: dict[str, float | None]
    #: How many tasks contributed a non-null value (the mean's denominator).
    sampled_task_count: int
    #: Tasks whose rows could not be joined to a mutant (id not in the suite).
    unmatched_task_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "split_role": self.split_role,
            "mean_attractor_pull": self.mean_attractor_pull,
            "sampled_task_count": self.sampled_task_count,
            "task_count": len(self.per_task),
            "per_task": self.per_task,
            "unmatched_task_ids": list(self.unmatched_task_ids),
        }


@dataclass(frozen=True, slots=True)
class RecomputeResult:
    """The offline recompute over every arm in a rollout sidecar."""

    outputs_path: str
    #: The reported arm (default ``official_best``) -- the headline number.
    reported_role: str
    arms: dict[str, ArmAttractor] = field(default_factory=dict)

    @property
    def reported(self) -> ArmAttractor | None:
        """The reported arm's recompute, or ``None`` when it is absent."""
        return self.arms.get(self.reported_role)

    def as_dict(self) -> dict[str, object]:
        return {
            "tool": "whetstone.tools.recompute_attractor",
            "outputs_path": self.outputs_path,
            "reported_role": self.reported_role,
            "reported_mean_attractor_pull": (
                self.reported.mean_attractor_pull
                if self.reported is not None else None
            ),
            "arms": {r: a.as_dict() for r, a in self.arms.items()},
        }


def _mutant_map(mutants_path: Path | None) -> dict[str, Ed1mMutant]:
    """Instance-id -> mutant, loaded directly from ``mutants.jsonl``."""
    pool = load_ed1m_mutants(mutants_path)
    return {m.mutant_id: m for m in pool}


def _score_role(
    split_role: str,
    rows: Sequence[dict[str, object]],
    mutants: dict[str, Ed1mMutant],
) -> ArmAttractor:
    """Replay the dual oracle for one arm's rows -> its attractor pull.

    Groups rows by ``instance_id``, replays each reconstruction's attractor
    (offline local execution), averages a task's non-null repeat values, then
    averages the per-task non-null values -- the same reported aggregate the
    live run computed.
    """
    per_task_vals: dict[str, list[float]] = {}
    per_task: dict[str, float | None] = {}
    unmatched: list[str] = []
    for row in rows:
        instance_id = str(row.get("instance_id", ""))
        per_task.setdefault(instance_id, None)
        mutant = mutants.get(instance_id)
        if mutant is None:
            if instance_id not in unmatched:
                unmatched.append(instance_id)
            continue
        raw_output = row.get("output_text")
        decoder = extract_decoder_text(
            raw_output if isinstance(raw_output, str) else None
        )
        if decoder is None:
            continue
        score = score_ed1m_reconstruction(
            reconstruction=decoder, mutant=mutant
        )
        if score.attractor_pull is not None:
            per_task_vals.setdefault(instance_id, []).append(
                score.attractor_pull
            )
    for instance_id, vals in per_task_vals.items():
        per_task[instance_id] = sum(vals) / len(vals) if vals else None
    sampled = [v for v in per_task.values() if v is not None]
    mean = sum(sampled) / len(sampled) if sampled else None
    return ArmAttractor(
        split_role=split_role,
        mean_attractor_pull=mean,
        per_task=per_task,
        sampled_task_count=len(sampled),
        unmatched_task_ids=tuple(unmatched),
    )


def recompute_attractor(
    outputs_path: Path,
    *,
    mutants_path: Path | None = None,
    reported_role: str = DEFAULT_REPORTED_ROLE,
) -> RecomputeResult:
    """Recompute attractor pull for every arm in a rollout sidecar (offline).

    Reads ``outputs_path`` (a persisted ``rollout_outputs`` jsonl), keeps only
    ed1m rows (``env == "ed1m"``, or all rows when the env field is absent),
    and replays the dual oracle per arm with NO live calls. ``mutants_path``
    defaults to the canonical ``mutants.jsonl``.
    """
    by_role: dict[str, list[dict[str, object]]] = {}
    with outputs_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            env = row.get("env")
            if env is not None and env != "ed1m":
                continue
            role = str(row.get("split_role", "unknown"))
            by_role.setdefault(role, []).append(row)
    mutants = _mutant_map(mutants_path)
    arms = {
        role: _score_role(role, rows, mutants)
        for role, rows in by_role.items()
    }
    return RecomputeResult(
        outputs_path=str(outputs_path),
        reported_role=reported_role,
        arms=arms,
    )


def write_result(result: RecomputeResult, out_path: Path) -> Path:
    """Persist the recompute record to ``out_path`` (never the input)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return out_path


def format_result(result: RecomputeResult) -> str:
    """A concise human-readable summary of the recompute (for stdout)."""
    lines = [
        f"attractor recompute (offline, read-only) <- {result.outputs_path}",
    ]
    reported = result.reported
    if reported is not None:
        mean = reported.mean_attractor_pull
        lines.append(
            f"  REPORTED [{result.reported_role}] "
            f"mean_attractor_pull="
            f"{'n/a' if mean is None else f'{mean:.4f}'} "
            f"(sampled_tasks={reported.sampled_task_count}"
            f"/{len(reported.per_task)})"
        )
    else:
        lines.append(
            f"  REPORTED [{result.reported_role}] arm not present in sidecar"
        )
    for role in sorted(result.arms):
        arm = result.arms[role]
        mean = arm.mean_attractor_pull
        note = (
            f" unmatched={len(arm.unmatched_task_ids)}"
            if arm.unmatched_task_ids else ""
        )
        lines.append(
            f"  {role}: mean_attractor_pull="
            f"{'n/a' if mean is None else f'{mean:.4f}'} "
            f"tasks={len(arm.per_task)} sampled={arm.sampled_task_count}{note}"
        )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_REPORTED_ROLE",
    "ArmAttractor",
    "RecomputeResult",
    "extract_decoder_text",
    "format_result",
    "recompute_attractor",
    "write_result",
]
