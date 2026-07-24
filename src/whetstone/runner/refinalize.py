"""Refinalize a cell line: recompute its status from persisted evidence.

A cell that COMPLETED every planned phase must never be ``status=halted``
(halted means work was actually cut short). An early runner defect stamped
``halted`` on a cell that finished all its observations + stats merely because
the total elapsed crept past the wall budget (the c11 case: 2000/2000
observations done, then halted).

This module is the minimal correction path. It reads an existing
``cells.jsonl`` line, decides -- purely from that line's PERSISTED evidence --
whether the recorded status is wrong, and if so APPENDS a corrected line with a
provenance note (``refinalized``). The original line is preserved (append-only
ledger); the corrected line supersedes it for the resumability key.

Recomputation is evidence-only (no new provider calls): the corrected status is
derived from the persisted ``delta`` + ``delta_ci95`` via the same status rule
the live cell uses, but ONLY when the persisted evidence shows every phase
completed (``best_official`` is present -- the best-candidate official eval
ran -- so nothing was cut short).
"""

from __future__ import annotations

from dataclasses import dataclass

from whetstone.runner.ledger import CellRecord, Ledger

__all__ = [
    "REFINALIZED_NOTE",
    "RefinalizeOutcome",
    "recompute_status",
    "refinalize_cell",
]

#: The provenance note appended to a corrected cell line.
REFINALIZED_NOTE = "refinalized"


@dataclass(slots=True)
class RefinalizeOutcome:
    """The result of a refinalize attempt over one cell line."""

    original: CellRecord
    corrected: CellRecord | None
    changed: bool
    reason: str


def _ci_excludes_zero(pair: tuple[float, float] | None) -> bool:
    """True when the persisted CI lies strictly on one side of 0."""
    if pair is None:
        return False
    low, high = pair
    return low > 0.0 or high < 0.0


def _status_from(
    delta: float | None, delta_ci95: tuple[float, float] | None
) -> str:
    """The sharpened status from a paired delta + its persisted CI (live rule).

    ``improved`` REQUIRES ``delta > 0`` AND the paired CI excluding 0;
    ``delta > 0`` with a CI spanning 0 is ``inconclusive``; ``delta <= 0`` is
    ``no-improvement``. Duplicated from the cell path deliberately (a pure,
    stable rule over persisted numbers) so refinalize imports no live cell
    machinery and makes no provider call.
    """
    if delta is None or delta <= 0:
        return "no-improvement"
    if _ci_excludes_zero(delta_ci95):
        return "improved"
    return "inconclusive"


#: Terminal statistical statuses that are only valid on a COMPLETE official
#: measurement (both arms resolved). Emitting any of these when an official arm
#: never resolved (``baseline_official``/``best_official`` is None) is a
#: certified-looking verdict from a partial vector -- the c18:a1 defect.
_STATISTICAL_STATUSES: frozenset[str] = frozenset(
    {"improved", "inconclusive", "no-improvement"}
)


def recompute_status(record: CellRecord) -> tuple[str, str]:
    """Recompute a cell's correct status from its persisted evidence.

    Returns ``(status, reason)``. Two corrections are made, both evidence-only
    (no provider calls):

    * A ``halted`` cell whose evidence shows every phase completed
      (``best_official`` present) is corrected to its statistical status: the
      best-candidate official eval ran, so no work was cut short and ``halted``
      is wrong.
    * A cell stamped a terminal STATISTICAL status (``improved`` /
      ``inconclusive`` / ``no-improvement``) while an official arm never
      resolved (``baseline_official`` or ``best_official`` is None) is
      corrected to ``incomplete-arm``: that verdict was emitted off a partial
      official vector (the c18:a1 defect -- naive=None yet no-improvement +
      headroom emitted). It is not a certified result.

    Any other cell keeps its recorded status (no change).
    """
    if record.status in _STATISTICAL_STATUSES and (
        record.baseline_official is None or record.best_official is None
    ):
        which = []
        if record.baseline_official is None:
            which.append("naive")
        if record.best_official is None:
            which.append("best")
        return "incomplete-arm", (
            f"terminal statistical status {record.status!r} emitted on an "
            f"INCOMPLETE official arm ({', '.join(which)}=None): a certified "
            "verdict off a partial vector; superseded by 'incomplete-arm'"
        )
    if record.status != "halted":
        return record.status, "not halted; unchanged"
    if record.best_official is None:
        # A genuinely cut-short cell (optimize/best never ran) -- keep halted.
        return "halted", (
            "halted with no best_official: work was cut short; unchanged"
        )
    corrected = _status_from(record.delta, record.delta_ci95)
    return corrected, (
        f"halted but every phase completed (best_official="
        f"{record.best_official!r}); recomputed to {corrected!r} from "
        f"persisted delta={record.delta!r} delta_ci95={record.delta_ci95!r}"
    )


def refinalize_cell(
    ledger: Ledger, *, optimizer: str, env: str, attempt: int
) -> RefinalizeOutcome:
    """Recompute + (if changed) append a corrected line for one cell.

    Reads the latest line for ``(optimizer, env, attempt)``, recomputes
    its status from persisted evidence, and -- when the status changes --
    APPENDS a corrected line (original preserved) carrying the ``refinalized``
    provenance note. Returns the outcome either way.
    """
    ledger.load()
    original: CellRecord | None = None
    for candidate in reversed(ledger.cells()):
        if (
            candidate.optimizer == optimizer
            and candidate.env == env
            and candidate.attempt == attempt
        ):
            original = candidate
            break
    if original is None:
        raise ValueError(
            f"no cell line for ({optimizer!r}, {env!r}, attempt={attempt})"
        )

    new_status, reason = recompute_status(original)
    if new_status == original.status:
        return RefinalizeOutcome(
            original=original, corrected=None, changed=False, reason=reason
        )

    note_parts = [REFINALIZED_NOTE, reason]
    if original.escalation_note:
        note_parts.append(f"original note: {original.escalation_note}")
    update: dict[str, object] = {
        "status": new_status,
        "escalation_note": "; ".join(note_parts),
    }
    # A correction to ``incomplete-arm`` must also STRIP the certified-looking
    # headroom / no-headroom determination the bad line carried (they were
    # emitted off a partial official vector). The superseding line records no
    # such verdict.
    if new_status == "incomplete-arm":
        update["headroom_delta"] = None
        update["headroom_ci95"] = None
        update["no_demonstrable_headroom"] = None
    corrected = original.model_copy(update=update)
    ledger.append_cell(corrected)
    return RefinalizeOutcome(
        original=original, corrected=corrected, changed=True, reason=reason
    )
