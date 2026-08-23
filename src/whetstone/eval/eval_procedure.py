from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from whetstone.eval.protocol import EvalTaskView

__all__ = [
    "SEED_AWARE_ATTRIBUTE",
    "EvalProcedureRunner",
    "SeedAwareEvalProcedureRunner",
    "accepts_seed_index",
]


@runtime_checkable
class EvalProcedureRunner(Protocol):
    """Runs one whetstone.eval/v1 node: upstream outputs -> score + submission."""

    def run_eval_node(
        self,
        *,
        node_id: str,
        node_inputs: Mapping[str, object],
        evaluation_procedure_config_hash: str,
        task: EvalTaskView,
    ) -> tuple[float | None, object | None, dict[str, object]]: ...


#: A runner sets this attribute to ``True`` to declare that its
#: ``run_eval_node`` also takes ``seed_index``. It is an explicit flag rather
#: than a ``runtime_checkable`` Protocol on purpose: ``isinstance`` against a
#: Protocol matches on *method names only* and ignores signatures, so every
#: ordinary ``EvalProcedureRunner`` would match and then be called with an
#: argument it does not accept.
SEED_AWARE_ATTRIBUTE = "accepts_seed_index"


@runtime_checkable
class SeedAwareEvalProcedureRunner(Protocol):
    """An ``EvalProcedureRunner`` that also reads which repeat it is scoring.

    A real eval procedure scores one generation and must not vary with the
    repeat that produced it, so ``EvalProcedureRunner`` deliberately does not
    carry ``seed_index``. Test doubles need the opposite: without per-repeat
    variation a repeat-mean is indistinguishable from repeat 0, and a test
    asserting the mean proves nothing.

    Implementations must set ``accepts_seed_index = True``; that flag, not
    structural typing, is what the graph rollout dispatches on.
    """

    accepts_seed_index: bool

    def run_eval_node(
        self,
        *,
        node_id: str,
        node_inputs: Mapping[str, object],
        evaluation_procedure_config_hash: str,
        task: EvalTaskView,
        seed_index: int,
    ) -> tuple[float | None, object | None, dict[str, object]]: ...


def accepts_seed_index(runner: object) -> bool:
    """Whether ``runner`` opted into receiving the row's ``seed_index``."""

    return getattr(runner, SEED_AWARE_ATTRIBUTE, False) is True
