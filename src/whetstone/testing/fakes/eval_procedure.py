from __future__ import annotations

from collections.abc import Mapping

from whetstone.eval.protocol import EvalTaskView
from whetstone.testing.toy.scoring import score_generation

__all__ = [
    "FakeEvalProcedureRunner",
    "RepeatVaryingEvalProcedureRunner",
]


class FakeEvalProcedureRunner:
    """Stub eval-node runner for toy graph exercises."""

    def run_eval_node(
        self,
        *,
        node_id: str,
        node_inputs: Mapping[str, object],
        evaluation_procedure_config_hash: str,
        task: EvalTaskView,
    ) -> tuple[float | None, object | None, dict[str, object]]:
        _ = (node_id, evaluation_procedure_config_hash)
        generation = node_inputs.get("provider_generation")
        text = generation if isinstance(generation, str) else str(generation or "")
        gold = ""
        if hasattr(task, "gold"):
            raw_gold = getattr(task, "gold")
            gold = raw_gold if isinstance(raw_gold, str) else ""
        score = score_generation(
            generation=text,
            gold=gold,
            task_id=task.task_id,
        )
        return score, {"text": text}, {}


class RepeatVaryingEvalProcedureRunner:
    """Toy eval-node runner whose score differs on every repeat of a task.

    Satisfies ``SeedAwareEvalProcedureRunner``, so the graph rollout hands it
    the row's ``seed_index``. Use it only where a test must tell a repeat-mean
    apart from repeat 0, the max, or the sum -- with the default
    ``FakeEvalProcedureRunner`` every repeat of a task scores identically and
    such an assertion holds no matter which reduction the code uses.
    """

    accepts_seed_index = True

    def run_eval_node(
        self,
        *,
        node_id: str,
        node_inputs: Mapping[str, object],
        evaluation_procedure_config_hash: str,
        task: EvalTaskView,
        seed_index: int,
    ) -> tuple[float | None, object | None, dict[str, object]]:
        _ = (node_id, evaluation_procedure_config_hash)
        generation = node_inputs.get("provider_generation")
        text = generation if isinstance(generation, str) else str(generation or "")
        gold = ""
        if hasattr(task, "gold"):
            raw_gold = getattr(task, "gold")
            gold = raw_gold if isinstance(raw_gold, str) else ""
        score = score_generation(
            generation=text,
            gold=gold,
            task_id=task.task_id,
            seed_index=seed_index,
        )
        return score, {"text": text}, {}
