from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from whetstone.evaluation.protocol import EvaluationTaskView

__all__ = ["EvalProcedureRunner"]


@runtime_checkable
class EvalProcedureRunner(Protocol):
    """Runs one whetstone.eval/v1 node: upstream outputs -> score + submission."""

    def run_eval_node(
        self,
        *,
        node_id: str,
        node_inputs: Mapping[str, object],
        evaluation_procedure_config_hash: str,
        task: EvaluationTaskView,
    ) -> tuple[float | None, object | None, dict[str, object]]: ...
