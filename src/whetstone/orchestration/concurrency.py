"""Provider Concurrency Control configuration.

Whetstone declares a deliberately coarse, conservative cross-worker cap by
configuring dr-platform's generic Stage Control selector-capacity mechanism.
The concurrency unit is one dr-providers Provider Quota Identity, exactly
``(provider, protocol, model)``. dr-platform enforces label/selector matching
generically without provider-specific code; **Whetstone alone** derives the
collision-free labels (:mod:`whetstone.orchestration.labels`) and owns the
capacity policy.

Two configuration duties, both mandatory:

* **The default empty-selector capacity is mandatory Stage configuration.**
  Without it, admission leaves matching work ``READY``/unadmitted even when
  provider-specific controls exist. :func:`configure_provider_concurrency`
  always writes it.

* **One exact-label capacity per Provider Quota Identity.** For each applicable
  ``(provider, protocol, model)`` the function configures a control whose
  selector is that quota's collision-free label. A Work Item carrying several
  quotas' labels matches every corresponding control, and admission applies
  **all** matching controls together for the full ADMITTED Stage lifetime,
  including backoff between semantic attempts.

Everything here is generic dr-platform Stage Control writes; no provider-
specific logic enters dr-platform.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from dr_platform.staging.identities import StageKey
from dr_platform.staging.operations import (
    set_selector_capacity,
    set_stage_capacity,
)

from whetstone.orchestration.labels import quota_selector
from whetstone.orchestration.pipeline import (
    ROLLOUT_EXECUTION_STAGE_KEY,
    orchestration_pipeline_identity,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from dr_platform.staging.records import StageControlRecord
    from dr_providers import ProviderQuotaIdentity
    from sqlalchemy import Engine

__all__ = [
    "ConcurrencyConfiguration",
    "QuotaCapacity",
    "configure_provider_concurrency",
]


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class QuotaCapacity:
    """One Provider Quota Identity paired with its per-label Stage capacity."""

    quota: ProviderQuotaIdentity
    capacity: int


@dataclass(frozen=True, slots=True)
class ConcurrencyConfiguration:
    """The Stage Controls written by one concurrency configuration call.

    ``default_control`` is the mandatory empty-selector control; ``per_quota``
    maps each configured quota's collision-free selector to its control.
    """

    default_control: StageControlRecord
    per_quota: Mapping[str, StageControlRecord]


def configure_provider_concurrency(
    *,
    engine: Engine,
    default_capacity: int,
    quota_capacities: tuple[QuotaCapacity, ...],
    stage_key: StageKey | str = ROLLOUT_EXECUTION_STAGE_KEY,
    clock: Callable[[], datetime] = _utc_now,
) -> ConcurrencyConfiguration:
    """Configure the mandatory default plus every exact-label capacity.

    Writes, on the rollout-execution Stage of the Orchestration Pipeline:

    * the mandatory default empty-selector control at ``default_capacity``
      (without it, matching work would remain ``READY``/unadmitted), and
    * one exact-label control per :class:`QuotaCapacity`, each keyed by that
      quota's collision-free selector.

    dr-platform later enforces all matching controls together for the full
    admitted Stage lifetime. Returns the written controls.
    """
    pipeline = orchestration_pipeline_identity()
    normalized_stage_key = (
        stage_key if isinstance(stage_key, StageKey) else StageKey(stage_key)
    )
    default_control = set_stage_capacity(
        pipeline=pipeline,
        stage_key=normalized_stage_key,
        capacity=default_capacity,
        engine=engine,
        clock=clock,
    )
    per_quota: dict[str, StageControlRecord] = {}
    for quota_capacity in quota_capacities:
        selector = dict(quota_selector(quota_capacity.quota))
        control = set_selector_capacity(
            pipeline=pipeline,
            stage_key=normalized_stage_key,
            labels=selector,
            capacity=quota_capacity.capacity,
            engine=engine,
            clock=clock,
        )
        # Key by the selector's single collision-free label value so callers
        # can retrieve one quota's control unambiguously.
        (label_value,) = selector.values()
        per_quota[label_value] = control
    return ConcurrencyConfiguration(
        default_control=default_control,
        per_quota=per_quota,
    )
