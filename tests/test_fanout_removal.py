"""The bespoke subprocess scheduler is gone, with no references left behind.

Rollout rows run on a dr-exec worker pool. These checks fail loudly if any
part of the retired fanout scheduler, its process worker, or its guardian
protocol is reintroduced or referenced.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SEARCH_ROOTS = ("src", "tests", "scripts")

RETIRED_MODULES = (
    "whetstone.execution.fanout",
    "whetstone.execution.process_worker",
    "whetstone.execution.process_guardian",
)

#: The retired scheduler-evidence fields (``concurrency_halved``,
#: ``guard_timeouts``) are pinned as absent by the evidence golden test,
#: which necessarily names them, so they are not searched for here.
RETIRED_SYMBOLS = (
    "CallSpec",
    "FanoutResult",
    "FanoutStatus",
    "PoolOutcome",
    "ProcessCancellationError",
    "ProcessJob",
    "ProcessWorkerError",
    "cancellation_barrier",
    "process_guardian",
    "process_worker",
    "run_call_pool",
)

#: This file names every retired symbol, so it can never be its own evidence.
SELF = Path(__file__).resolve()


def _python_sources() -> tuple[Path, ...]:
    return tuple(
        path
        for root in SEARCH_ROOTS
        for path in (REPO_ROOT / root).rglob("*.py")
        if path.resolve() != SELF
    )


@pytest.mark.parametrize("module_name", RETIRED_MODULES)
def test_retired_module_is_not_importable(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("symbol", RETIRED_SYMBOLS)
def test_retired_symbol_has_no_references(symbol: str) -> None:
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _python_sources()
        if symbol in path.read_text()
    ]
    assert not offenders, f"{symbol!r} still referenced in {offenders}"


def test_fanout_symbols_are_not_re_exported_from_execution() -> None:
    import whetstone.execution as execution

    assert not {"run_call_pool", "CallSpec", "FanoutStatus"} & set(
        execution.__all__
    )
    assert "DEFAULT_CONCURRENCY" not in execution.__all__
