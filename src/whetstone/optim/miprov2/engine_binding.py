"""Bind MIPROv2's per-effect Eval Config derivations to a real eval engine.

MIPROv2 evaluates different task subsets at different points in its search:
one task for a bootstrap generation, a minibatch for a sampled trial, the
full validation set for a baseline or promotion. Each subset is its own
Eval Config, derived from the run's source config, so every evaluation is
attributable to the exact sampling it ran under.

:class:`EngineEvalBindingResolver` produces those derivations from the
engine that will actually run them, via ``engine.for_task_ids``. Deriving
the binding from the same object that executes it is what keeps the
recorded Eval Config honest: nothing here invents a sampling the engine
would not reproduce.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from whetstone.eval.protocol import EvalEngine
from whetstone.optim.miprov2.eval_config import (
    EvalBinding,
    EvalBindingRequest,
    derive_eval_config_reference,
)

if TYPE_CHECKING:
    from whetstone.experiment.sampling import EvalSplit

__all__ = ["EngineEvalBindingResolver", "engine_for_task_hashes"]


def engine_for_task_hashes(
    engine: EvalEngine,
    task_hashes: tuple[str, ...],
) -> EvalEngine:
    """Narrow ``engine`` to exactly ``task_hashes``, in that order.

    The engine addresses tasks by task id while MIPROv2 addresses them by
    task hash, because a hash is the identity that survives persistence.
    This translates between the two against the engine's own split, so an
    unknown or duplicated hash fails here rather than silently evaluating
    the wrong rows.
    """

    if not task_hashes:
        raise ValueError("an Eval Config derivation requires at least one task")
    if len(set(task_hashes)) != len(task_hashes):
        raise ValueError("Eval Config derivation tasks must be unique")
    split = _split_of(engine)
    id_by_hash = dict(
        zip(
            split.task_set.task_hashes,
            (task.task_id for task in split.tasks),
            strict=True,
        )
    )
    try:
        task_ids = tuple(id_by_hash[task_hash] for task_hash in task_hashes)
    except KeyError as exc:
        raise ValueError(
            f"engine sampling has no task with hash {exc.args[0]!r}"
        ) from None
    return engine.for_task_ids(task_ids)


def _split_of(engine: EvalEngine) -> EvalSplit:
    return engine.sampling_split


class EngineEvalBindingResolver:
    """Derive MIPROv2 Eval Config bindings from a bound eval engine."""

    def __init__(self, *, engine: EvalEngine) -> None:
        self._engine = engine

    @property
    def engine(self) -> EvalEngine:
        return self._engine

    def engine_for(self, request: EvalBindingRequest) -> EvalEngine:
        """The engine narrowed to the request's exact ordered task subset."""

        return engine_for_task_hashes(self._engine, request.task_batch_hashes)

    def resolve(self, request: EvalBindingRequest) -> EvalBinding:
        subset = self.engine_for(request)
        split = _split_of(subset)
        if split.seed_plan.num_seeds != request.num_seeds:
            raise ValueError(
                "engine sampling repeats "
                f"({split.seed_plan.num_seeds}) do not match the requested "
                f"num_seeds ({request.num_seeds})"
            )
        engine_task_model = subset.task_model_identity_hash()
        requested_task_model = request.execution_policy.task_model_identity_hash
        if engine_task_model != requested_task_model:
            raise ValueError(
                "engine task-model route "
                f"({engine_task_model}) does not match the requested "
                f"task_model_identity_hash ({requested_task_model})"
            )
        # The binding records the derivation of the *source* config under the
        # subset sampling, which is the identity MIPROv2 persists. The
        # subset engine's own eval_config_ref is derived the same way from
        # the same sampling, so the two agree whenever the request's source
        # config is the engine's -- and disagreeing is a real conflict.
        eval_config = derive_eval_config_reference(
            request.source_eval_config,
            split.sampling_config,
        )
        return EvalBinding(
            request=request,
            task_set=split.task_set,
            seed_plan=split.seed_plan,
            sampling_config=split.sampling_config,
            eval_config=eval_config,
        )
