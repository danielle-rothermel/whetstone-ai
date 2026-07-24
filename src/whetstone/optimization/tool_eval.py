"""The tool-side evaluation service shared by the tool-using optimizers.

GEPA and Codex both reach measurement through Tool Calls rather than
Evaluation Intents. Both therefore need the same execution-boundary machinery:

* a :class:`ToolEvaluator` protocol -- Whetstone's internal-role evaluation of
  one accepted Tool Call: validate, materialize the chosen base + template at
  the fixed ratio, bind the tool's ordinary Eval Config under the *internal*
  Evaluation Role, plan/execute the Rollouts the Eval Config sizes, aggregate,
  and derive the tool's Reward. It returns typed evaluation evidence (Rollout
  Result references + aggregates); it never manufactures a score and never
  touches the store.

* :class:`EvaluatingToolExecutor` -- the ``ToolExecutor`` implementation that
  constructs a :class:`RuntimeToolHandle` **only at the execution boundary**.
  The handle drives the authoritative Tool Call Store: ``accept_or_refuse``
  first (so a capacity/budget/validation refusal is a typed non-execution
  outcome that never masquerades as a measurement), then -- only for an
  accepted call -- the :class:`ToolEvaluator` produces the internal evaluation,
  the Reward Policy scalarizes it, and a completed Tool Result is returned.
  Every Tool Result names its terminal Tool Call Store Entry via the store; the
  harness records both as Step evidence.

The Reward is produced only through :func:`apply_reward_policy`, so a refused
call carries no Reward and official evaluation still computes none.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization.harness import ToolExecutor
from whetstone.optimization.identity import TypedRef
from whetstone.optimization.reward import RewardPolicy, apply_reward_policy
from whetstone.optimization.tool_store import ToolCallState, ToolCallStore
from whetstone.optimization.tools import (
    RefusalClass,
    RuntimeToolHandle,
    ToolCall,
    ToolConfig,
    ToolRefusal,
    ToolResult,
)

__all__ = [
    "EvaluatingToolExecutor",
    "ToolEvaluation",
    "ToolEvaluator",
    "ToolValidationError",
]


class ToolValidationError(ValueError):
    """A Tool Call's arguments failed validation before any measurement.

    Raised by a :class:`ToolEvaluator` when the call arguments are ill-formed
    (missing base/template, off-surface field, disallowed Model Route). The
    executor converts it into a typed VALIDATION Tool Refusal so it can never
    be mistaken for a measured result.
    """


@dataclass(frozen=True, slots=True)
class ToolEvaluation:
    """The internal-role evaluation evidence a :class:`ToolEvaluator` returns.

    * ``rollout_refs`` -- the Rollout Result references the Eval Config sized
      (internal_task_count times internal_repeat_count for Codex; 3 for a GEPA
      minibatch; 24 for a GEPA subset). Referenced, never duplicated.
    * ``aggregates`` -- the named internal aggregate/objective values the
      Reward Policy scalarizes; a value may be ``None`` to exercise the
      policy's
      missing-data rule.
    * ``objective_values`` -- the per-objective values GEPA compares/frontiers
      on (a subset of/derived from ``aggregates``); carried for the caller.
    * ``eval_config_hash`` -- the exact internal Eval Config identity the
      evaluation bound; recorded as tool evidence.
    * ``source_eval_config_hash`` -- the Tool Config's source Eval Config
      identity from which an exact task subset may be derived.
    * ``extra_output`` -- any additional typed output the tool exposes to the
      caller (e.g. a GEPA bounded Rollout Trace projection ref).
    """

    rollout_refs: tuple[TypedRef, ...]
    aggregates: dict[str, float | None]
    eval_config_hash: str
    source_eval_config_hash: str
    objective_values: dict[str, float] = field(default_factory=dict)
    extra_output: dict[str, object] = field(default_factory=dict)


class ToolEvaluator(Protocol):
    """Whetstone's internal-role evaluation of one accepted Tool Call.

    Given the accepted :class:`ToolCall` and its :class:`ToolConfig`, resolves
    the internal evaluation under the Tool Config's ordinary Eval Config bound
    through the ``internal`` Evaluation Role and returns a
    :class:`ToolEvaluation`. It is called ONLY after the Tool Call Store
    accepts
    the call, so it never debits capacity and never sees a refused call.
    """

    def evaluate(
        self, call: ToolCall, config: ToolConfig
    ) -> ToolEvaluation: ...


class EvaluatingToolExecutor(ToolExecutor):
    """A :class:`ToolExecutor` that measures accepted calls, refuses the rest.

    Constructs the :class:`RuntimeToolHandle` at the execution boundary only.
    The bound callable's order is load-bearing:

    1. ``store.accept_or_refuse(call, config)`` -- the authoritative Tool Call
       Store decides acceptance/refusal and debits capacity **exactly once** on
       acceptance. A capacity refusal (or a pre-accept validation refusal)
       yields a refused Tool Result with NO evaluation evidence and NO Reward.
    2. only for an accepted call, the :class:`ToolEvaluator` runs the
       internal-role evaluation and the Reward Policy scalarizes it into the
       Tool Result's Reward.

    The completed Tool Result is transitioned in the store by the harness
    (which
    records the terminal Entry as evidence), so this executor leaves the entry
    ``accepted`` and returns the result for the harness to complete.
    """

    def __init__(
        self,
        evaluator: ToolEvaluator,
        reward_policy: RewardPolicy,
    ) -> None:
        self._evaluator = evaluator
        self._reward_policy = reward_policy

    def runtime_handle(
        self, config: ToolConfig, store: ToolCallStore
    ) -> RuntimeToolHandle:
        if config.reward_policy_ref != self._reward_policy.identity_hash():
            raise ValueError(
                "Tool Config reward_policy_ref does not match the executor's "
                "Reward Policy Identity Hash"
            )

        def execute(call: ToolCall) -> ToolResult:
            existing = store.get(call.tool_config_hash, call.call_id)
            if existing is not None:
                entry = store.accept_or_refuse(call, config)
                if entry.state is ToolCallState.COMPLETED:
                    return store.load_completed_result(entry)
                if entry.state is ToolCallState.REFUSED:
                    assert entry.refusal is not None
                    return _refused_result(call, config, entry.refusal)

            # Pre-acceptance argument validation is a typed VALIDATION refusal
            # that never spends a capacity slot and never becomes a
            # measurement. It is recorded durably in the Tool Call Store so the
            # refusal is inspectable evidence with a refused terminal Entry.
            try:
                self._validate_args(call)
            except ToolValidationError as exc:
                refusal = ToolRefusal(
                    refusal_class=RefusalClass.VALIDATION, reason=str(exc)
                )
                store.refuse(call, config, refusal=refusal)
                return _refused_result(call, config, refusal)

            entry = store.accept_or_refuse(call, config)
            if entry.state is ToolCallState.REFUSED:
                # Capacity (or store-side) refusal: no evidence, no Reward.
                assert entry.refusal is not None
                return ToolResult(
                    call_id=call.call_id,
                    tool_config_ref=config.tool_definition_ref,
                    tool_config_hash=config.identity_hash(),
                    store_namespace=config.store_namespace,
                    refusal=entry.refusal,
                    provenance_ordinal=None,
                )

            evaluation = self._evaluator.evaluate(call, config)
            if (
                evaluation.source_eval_config_hash
                != config.eval_config_identity_hash
            ):
                raise ValueError(
                    "the tool's internal evaluation was not derived from "
                    "the Tool Config's internal-role Eval Config"
                )
            reward = apply_reward_policy(
                self._reward_policy,
                aggregates=evaluation.aggregates,
                evidence_role=EvaluationRole.INTERNAL,
            )
            output: dict[str, object] = {
                "rollout_refs": [
                    ref.model_dump(mode="json")
                    for ref in evaluation.rollout_refs
                ],
                "rollout_ref_count": len(evaluation.rollout_refs),
                "objective_values": evaluation.objective_values,
                "eval_config_hash": evaluation.eval_config_hash,
                "source_eval_config_hash": (
                    evaluation.source_eval_config_hash
                ),
                "internal_role": EvaluationRole.INTERNAL.value,
                **evaluation.extra_output,
            }
            result = ToolResult(
                call_id=call.call_id,
                tool_config_ref=config.tool_definition_ref,
                tool_config_hash=config.identity_hash(),
                store_namespace=config.store_namespace,
                output=output,
                evaluation_evidence_refs=evaluation.rollout_refs,
                reward=reward.record_content(),
                provenance_ordinal=entry.capacity_debit_ordinal,
            )
            store.persist_and_complete(result)
            return result

        return RuntimeToolHandle(config, execute)

    @staticmethod
    def _validate_args(call: ToolCall) -> None:
        template = call.args.get("template")
        if not isinstance(template, str) or template == "":
            raise ToolValidationError(
                "evaluate call requires a non-empty encoder 'template'"
            )
        route = call.args.get("model_route")
        if not isinstance(route, str) or route == "":
            raise ToolValidationError(
                "evaluate call requires a non-empty 'model_route'"
            )


def _refused_result(
    call: ToolCall,
    config: ToolConfig,
    refusal: ToolRefusal,
) -> ToolResult:
    return ToolResult(
        call_id=call.call_id,
        tool_config_ref=config.tool_definition_ref,
        tool_config_hash=config.identity_hash(),
        store_namespace=config.store_namespace,
        refusal=refusal,
    )
