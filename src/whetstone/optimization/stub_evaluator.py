"""A deterministic stub ToolEvaluator for tests and the live smoke test.

This produces internal-role evaluation evidence *without* running a real graph:
it derives stable objective values from the call's template text so tests (and
the live Codex smoke test, which runs this in the MCP-server subprocess) get
reproducible measurements and Rewards. It is NOT a real evaluator -- it never
executes Rollouts -- but it exercises the exact tool-evaluation contract
(internal Eval Config binding, Rollout-ref shaping, aggregate emission) so the
Reward Policy and Tool Result plumbing are proven end to end.

The module-level :func:`make_stub_evaluator` factory is the seam the launched
MCP server reaches via ``WS_MCP_EVALUATOR=whetstone.optimization.stub_evaluator
:make_stub_evaluator``.
"""

from __future__ import annotations

import hashlib

from whetstone.optimization.identity import TypedRef, typed_ref_for_record
from whetstone.optimization.tool_eval import ToolEvaluation
from whetstone.optimization.tools import ToolCall, ToolConfig

__all__ = ["StubToolEvaluator", "make_stub_evaluator"]

_ROLLOUT_SCHEMA = "whetstone.test.stub_rollout_result"


def _stable_unit(text: str, salt: str) -> float:
    digest = hashlib.sha256((salt + "::" + text).encode()).digest()
    return int.from_bytes(digest[:6], "big") / float(1 << 48)


class StubToolEvaluator:
    """Deterministic internal-role evaluation derived from the template text.

    ``rollout_count`` is the number of Rollout refs to emit per call (20 for
    the Codex internal Eval Config; a GEPA minibatch would use 3, a subset 24).
    The refs are content-addressed stand-ins; the objective/aggregate values
    are stable functions of the template so a replay reproduces them exactly.
    """

    def __init__(self, *, rollout_count: int = 20) -> None:
        self._rollout_count = rollout_count

    def evaluate(
        self, call: ToolCall, config: ToolConfig
    ) -> ToolEvaluation:
        template = str(call.args.get("template", ""))
        route = str(call.args.get("model_route", ""))
        pass_rate = _stable_unit(template + route, "pass")
        # Compression length: a small integer-ish value that shrinks with a
        # longer, more specific template (stable, bounded).
        compression = 1.0 + _stable_unit(template, "compress")

        rollout_refs: list[TypedRef] = []
        for index in range(self._rollout_count):
            record: dict[str, str | int] = {
                "schema": _ROLLOUT_SCHEMA,
                "call_id": call.call_id,
                "task_index": index,
                "template": template,
                "route": route,
            }
            rollout_refs.append(
                typed_ref_for_record(_ROLLOUT_SCHEMA, record)
            )

        return ToolEvaluation(
            rollout_refs=tuple(rollout_refs),
            aggregates={"pass_rate": pass_rate, "compression": compression},
            objective_values={
                "correctness": pass_rate,
                "compression": compression,
            },
            eval_config_hash=config.eval_config_identity_hash,
        )


def make_stub_evaluator() -> StubToolEvaluator:
    """Factory the launched MCP server reaches via ``WS_MCP_EVALUATOR``."""
    return StubToolEvaluator(rollout_count=20)
