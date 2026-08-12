from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from whetstone.coordination.official.records import PlannedKeyResult
from whetstone.core.identity import TypedRef

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from dr_store import ObjectReference

__all__ = [
    "OfficialAggregationAccount",
    "OfficialFailurePolicy",
    "account_planned_keys",
]


class OfficialFailurePolicy(StrEnum):
    STRICT = "strict"
    RECORD_MISSING = "record_missing"


class MissingPlannedKeysError(ValueError):
    def __init__(self, missing: Sequence[str]) -> None:
        self.missing = tuple(missing)
        super().__init__(
            f"{len(self.missing)} planned Generation Execution Key(s) "
            "have no bound ordinary Generation Result under the strict "
            "official policy: "
            f"{list(self.missing)}"
        )


@dataclass(frozen=True, slots=True)
class OfficialAggregationAccount:
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
        return self.missing_count == 0


def account_planned_keys(
    *,
    planned_keys: Sequence[str],
    resolve: Callable[[str], ObjectReference | None],
    policy: OfficialFailurePolicy = OfficialFailurePolicy.STRICT,
    raise_on_missing: bool = False,
) -> OfficialAggregationAccount:
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
