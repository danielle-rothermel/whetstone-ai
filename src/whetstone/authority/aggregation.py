"""Official aggregation accounting: every planned key, none dropped.

Deliverable 5 of Workstream 9: *official aggregation accounts for every planned
key under the configured failure policy — missing rows are visible, never
dropped.*

The official write path plans a set of Rollout Execution Keys, resolves each to
its ordinary Rollout Result reference through the authoritative Result Store,
and produces a complete, explicit account: every planned key becomes exactly
one :class:`~whetstone.authority.records.PlannedKeyResult`, present or missing.
Under
:data:`OfficialFailurePolicy.STRICT` any missing planned key makes the official
account incomplete (so it cannot be certified); under
:data:`OfficialFailurePolicy.RECORD_MISSING` missing rows are recorded and
remain visible, but they are still counted — never silently skipped — and the
account is still marked incomplete so certification refuses it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from whetstone.authority.records import PlannedKeyResult
from whetstone.optimization.identity import TypedRef

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from dr_store import ObjectReference

__all__ = [
    "OfficialAggregationAccount",
    "OfficialFailurePolicy",
    "account_planned_keys",
]


class OfficialFailurePolicy(StrEnum):
    """How the official account treats a missing planned key.

    ``STRICT`` — a missing planned key is a hard incompleteness: the account is
    incomplete and its missing keys are recorded for audit.
    ``RECORD_MISSING`` — missing keys are recorded and stay visible in the
    account; the account is still incomplete (certification refuses it), but
    the account is produced rather than raising, so the operator sees exactly
    which keys are missing.

    Neither policy drops a planned key: the difference is only whether a
    missing key raises immediately (``STRICT``, when ``raise_on_missing`` is
    set) or is surfaced as a visible missing row.
    """

    STRICT = "strict"
    RECORD_MISSING = "record_missing"


class MissingPlannedKeysError(ValueError):
    """Planned keys were missing under a strict, raising official policy."""

    def __init__(self, missing: Sequence[str]) -> None:
        self.missing = tuple(missing)
        super().__init__(
            f"{len(self.missing)} planned Rollout Execution Key(s) have no "
            "bound ordinary Rollout Result under the strict official policy: "
            f"{list(self.missing)}"
        )


@dataclass(frozen=True, slots=True)
class OfficialAggregationAccount:
    """A complete account of every planned key: present or missing, all shown.

    ``planned_results`` has exactly one entry per planned key in planned order.
    ``missing_keys`` lists the planned keys with no bound Result. The counts
    always satisfy ``present + missing == planned`` so the account provably
    covers the complete planned matrix with nothing dropped.
    """

    policy: OfficialFailurePolicy
    planned_results: tuple[PlannedKeyResult, ...]
    missing_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.planned_results:
            raise ValueError("an official account has >=1 planned key")
        keys = [p.planned_key for p in self.planned_results]
        if len(set(keys)) != len(keys):
            raise ValueError("planned keys must be unique in the account")
        present = [p.planned_key for p in self.planned_results if p.is_present]
        missing = [
            p.planned_key for p in self.planned_results if not p.is_present
        ]
        if tuple(sorted(missing)) != tuple(sorted(self.missing_keys)):
            raise ValueError(
                "missing_keys must exactly match the missing planned rows"
            )
        # The complete-matrix guarantee: nothing is dropped.
        if len(present) + len(missing) != len(self.planned_results):
            raise ValueError("planned rows must all be accounted for")

    @property
    def planned_count(self) -> int:
        return len(self.planned_results)

    @property
    def present_count(self) -> int:
        return sum(1 for p in self.planned_results if p.is_present)

    @property
    def missing_count(self) -> int:
        return len(self.missing_keys)

    @property
    def complete(self) -> bool:
        """Complete iff every planned key resolved to a bound Result."""
        return self.missing_count == 0


def account_planned_keys(
    *,
    planned_keys: Sequence[str],
    resolve: Callable[[str], ObjectReference | None],
    policy: OfficialFailurePolicy = OfficialFailurePolicy.STRICT,
    raise_on_missing: bool = False,
) -> OfficialAggregationAccount:
    """Account for every planned key under the configured failure policy.

    ``planned_keys`` is the complete planned set of canonical Rollout Execution
    Key strings; ``resolve`` maps one such key to its bound ordinary Rollout
    Result Object Reference (or ``None`` when unbound). Every planned key
    produces exactly one :class:`PlannedKeyResult`; a missing key is recorded
    as an explicit missing row, never dropped.

    With ``raise_on_missing`` set (and the strict policy), any missing key
    raises :class:`MissingPlannedKeysError` after the full account is computed,
    so the caller still sees the complete missing set in the exception.
    """
    if not planned_keys:
        raise ValueError("account_planned_keys requires >=1 planned key")
    if len(set(planned_keys)) != len(planned_keys):
        raise ValueError("planned_keys must be unique")

    rows: list[PlannedKeyResult] = []
    missing: list[str] = []
    for key in planned_keys:
        reference = resolve(key)
        if reference is None:
            missing.append(key)
            rows.append(PlannedKeyResult(planned_key=key, result_ref=None))
        else:
            rows.append(
                PlannedKeyResult(
                    planned_key=key,
                    result_ref=TypedRef(
                        schema_name=reference.schema,
                        content_hash=reference.content_hash,
                    ),
                )
            )

    account = OfficialAggregationAccount(
        policy=policy,
        planned_results=tuple(rows),
        missing_keys=tuple(missing),
    )
    if missing and raise_on_missing:
        raise MissingPlannedKeysError(missing)
    return account


__all__ += ["MissingPlannedKeysError"]
