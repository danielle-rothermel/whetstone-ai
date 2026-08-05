"""Refinalize a cell line: recompute its status from persisted evidence.

Two recorded statuses can be wrong in ways the persisted line itself proves,
and this module is the minimal correction path for both. It reads an existing
``cells.jsonl`` line, decides -- purely from that line's persisted evidence --
whether the recorded status is wrong, and if so appends a corrected line
carrying the ``refinalized`` provenance note. The original line is preserved,
because the ledger is append-only; the corrected line supersedes it for the
resumability key.

Recomputation is evidence-only and makes no provider call: the corrected status
is derived from the persisted ``delta`` and ``delta_ci95`` through the same
status rule the live cell uses, and only when the persisted evidence shows
every phase completed.
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

#: Persisted-format contract: the provenance note prefix on a corrected line.
REFINALIZED_NOTE = "refinalized"


@dataclass(frozen=True, slots=True)
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
    """The status implied by a paired delta and its persisted interval.

    ``improved`` requires ``delta > 0`` and the paired CI excluding 0; a
    positive delta whose CI spans 0 is ``inconclusive``; a non-positive delta
    is ``no-improvement``. The rule is duplicated from the cell path
    deliberately -- it is a pure, stable function of persisted numbers -- so
    refinalize imports no live cell machinery and can make no provider call.
    """
    if delta is None or delta <= 0:
        return "no-improvement"
    if _ci_excludes_zero(delta_ci95):
        return "improved"
    return "inconclusive"


#: Terminal statistical statuses that are only valid on a complete official
#: measurement, meaning both arms resolved. Emitting any of these when an
#: official arm never resolved is a certified-looking verdict read off a
#: partial vector.
_STATISTICAL_STATUSES: frozenset[str] = frozenset(
    {"improved", "inconclusive", "no-improvement"}
)


def recompute_status(record: CellRecord) -> tuple[str, str]:
    """Recompute a cell's correct status from its persisted evidence.

    Returns ``(status, reason)``. Two corrections are made, both evidence-only:

    * A ``halted`` cell whose evidence shows every phase completed
      (``best_official`` is present) is corrected to its statistical status:
      the best-candidate official evaluation ran, so no work was cut short and
      ``halted`` -- which means work was actually cut short -- is wrong.
    * A cell stamped a terminal statistical status while an official arm never
      resolved (``baseline_official`` or ``best_official`` is ``None``) is
      corrected to ``incomplete-arm``. That verdict was emitted off a partial
      official vector and is not a certified result.

    Any other cell keeps its recorded status.
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
            f"incomplete official arm ({', '.join(which)}=None): a certified "
            "verdict off a partial vector; superseded by 'incomplete-arm'"
        )
    if record.status != "halted":
        return record.status, "not halted; unchanged"
    if record.best_official is None:
        # A genuinely cut-short cell: the best-candidate arm never ran.
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
    """Recompute and, when it changed, append a corrected line for one cell.

    Reads the latest line for ``(optimizer, env, attempt)``, recomputes its
    status from persisted evidence, and when the status changes appends a
    corrected line -- the original is preserved -- carrying the ``refinalized``
    provenance note. Returns the outcome either way.
    """
    ledger.load()
    original = ledger.for_attempt(optimizer, env, attempt)
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
    # A correction to ``incomplete-arm`` also strips the certified-looking
    # headroom determination the superseded line carried, because it was
    # emitted off a partial official vector. The superseding line is not a
    # completed cell, so it publishes no viewer directory either.
    if new_status == "incomplete-arm":
        update["headroom_delta"] = None
        update["headroom_ci95"] = None
        update["artifacts"] = original.artifacts.model_copy(
            update={"viewer_publication": None}
        )
    corrected = original.model_copy(update=update)
    # Revalidate through the wire form: model_copy bypasses validation, and a
    # corrected line must satisfy exactly the contract a fresh line does.
    corrected = CellRecord.model_validate(
        corrected.model_dump(mode="json", by_alias=True)
    )
    ledger.append_cell(corrected)
    return RefinalizeOutcome(
        original=original, corrected=corrected, changed=True, reason=reason
    )
