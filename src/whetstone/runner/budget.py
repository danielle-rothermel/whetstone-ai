"""Budget guards and the OpenRouter credits snapshot.

Two hard guards bound a validation run's spend:

* **Reserve guard**: a canonical cell refuses to START when remaining credits
  are below the reserve (``$18.60``). Below the reserve, only reruns proceed.
* **Per-cell stop-loss**: a cell whose spend tracks above ``2x`` the expected
  per-cell cost halts (``status=halted``).

The credits snapshot models ``GET /api/v1/credits``. The fetcher is injected --
a scripted callable in tests, a real HTTP GET in a live run -- so no network
call is made by importing this module. Reading a credits number is not a paid
LLM call; it is budget plumbing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEFAULT_EXPECTED_CELL_USD",
    "OPENROUTER_CREDITS_URL",
    "RESERVE_USD",
    "STOP_LOSS_MULTIPLIER",
    "BudgetGuard",
    "CreditsFetcher",
    "CreditsSnapshot",
    "ReserveError",
    "StopLossError",
    "credits_from_payload",
    "openrouter_credits_fetcher",
]

#: The OpenRouter credits endpoint the live fetcher GETs (budget plumbing, not
#: a paid LLM call).
OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"

#: The reserve floor; below it only reruns run.
RESERVE_USD = 18.60

#: Expected canonical per-cell cost.
DEFAULT_EXPECTED_CELL_USD = 2.0

#: A cell that spends above this multiple of expected is halted.
STOP_LOSS_MULTIPLIER = 2.0


class ReserveError(Exception):
    """A canonical cell was refused because remaining < reserve."""


class StopLossError(Exception):
    """A cell exceeded its per-cell stop-loss and must halt."""


@dataclass(frozen=True, slots=True)
class CreditsSnapshot:
    """An OpenRouter ``GET /api/v1/credits`` snapshot."""

    total_credits: float | None
    total_usage: float | None
    at: str = ""

    @property
    def remaining_usd(self) -> float | None:
        if self.total_credits is None or self.total_usage is None:
            return None
        return self.total_credits - self.total_usage


def credits_from_payload(payload: dict[str, Any]) -> CreditsSnapshot:
    """Parse an OpenRouter credits JSON body into a snapshot.

    OpenRouter returns ``{"data": {"total_credits": .., "total_usage": ..}}``;
    a flat body is also accepted.
    """
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        data = payload
    total_credits = data.get("total_credits")
    total_usage = data.get("total_usage")
    return CreditsSnapshot(
        total_credits=(
            float(total_credits) if total_credits is not None else None
        ),
        total_usage=(float(total_usage) if total_usage is not None else None),
        at=str(payload.get("at", "")),
    )


CreditsFetcher = Callable[[], CreditsSnapshot | None]


def openrouter_credits_fetcher(
    api_key_env: str = "OPENROUTER_API_KEY",
    *,
    url: str = OPENROUTER_CREDITS_URL,
) -> CreditsFetcher:  # pragma: no cover - live only
    """Build a live ``GET /api/v1/credits`` fetcher for the CLI live path.

    Reads the OpenRouter key from ``api_key_env`` and returns a zero-arg
    callable that GETs the credits endpoint and parses it into a
    :class:`CreditsSnapshot`. Returns ``None`` when the key is absent, so a
    lane without credentials degrades to no-spend-recording rather than
    raising. Reading credits is budget plumbing, not a paid LLM call.
    """
    import os

    def fetch() -> CreditsSnapshot | None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            return None
        import httpx

        response = httpx.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )
        response.raise_for_status()
        return credits_from_payload(response.json())

    return fetch


@dataclass(frozen=True, slots=True)
class BudgetGuard:
    """Enforces the reserve and the per-cell stop-loss.

    ``remaining_usd`` at cell start may be ``None`` when credits are
    unavailable -- for example a plan-lane cell that makes no OpenRouter call.
    ``expected_cell_usd`` and ``reserve_usd`` are the guard thresholds.
    """

    reserve_usd: float = RESERVE_USD
    expected_cell_usd: float = DEFAULT_EXPECTED_CELL_USD
    stop_loss_multiplier: float = STOP_LOSS_MULTIPLIER

    @property
    def stop_loss_usd(self) -> float:
        return self.expected_cell_usd * self.stop_loss_multiplier

    def check_start(
        self,
        *,
        canonical: bool,
        remaining_usd: float | None,
        is_rerun: bool = False,
    ) -> None:
        """Refuse to start a canonical cell below the reserve.

        A rerun is permitted below reserve; a fresh canonical cell is not.
        When remaining is unavailable (no OpenRouter lane), the reserve does
        not gate.
        """
        if not canonical or is_rerun or remaining_usd is None:
            return
        if remaining_usd < self.reserve_usd:
            raise ReserveError(
                f"remaining ${remaining_usd:.2f} < reserve "
                f"${self.reserve_usd:.2f}; only reruns may proceed"
            )

    def check_stop_loss(self, spent_usd: float) -> None:
        """Halt a cell whose spend crossed ``multiplier x`` expected."""
        if spent_usd > self.stop_loss_usd:
            raise StopLossError(
                f"cell spent ${spent_usd:.2f} > stop-loss "
                f"${self.stop_loss_usd:.2f} "
                f"({self.stop_loss_multiplier}x expected "
                f"${self.expected_cell_usd:.2f})"
            )

    def would_halt(self, spent_usd: float) -> bool:
        return spent_usd > self.stop_loss_usd
