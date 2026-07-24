"""Eval identity partition validation.

The composite dr-code Eval Config binds three component Configs: Sampling,
Evaluation Procedure, and Aggregation. Whetstone additionally requires that
the Eval Config's *Evaluation Procedure Config identity* exactly match the
Evaluation Procedure Config reference assigned as a static Variable on the
Eval Node in the Graph Config.

That match is the seam that produces the settled identity partition:

* Changing the Evaluation Procedure Config changes the Eval Node's static
  Variable (hence ``graph_hash``) *and* the composite ``eval_config_hash``.
* Changing only Sampling or only Aggregation changes only
  ``eval_config_hash`` — the Graph Config, and thus ``graph_hash``, is
  untouched.

Whetstone owns no parallel Eval Config identity; ``eval_config_hash`` is
exactly ``EvalConfig.config_identity_hash`` from dr-code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dr_graph import GraphConfig

from whetstone.graph.nodes import EVAL_NODE_TYPE, eval_node_procedure_hash

if TYPE_CHECKING:
    from dr_code.eval import EvalConfig


class EvalIdentityMismatchError(ValueError):
    """The composite Eval Config's Procedure identity does not match the
    Eval Node's statically assigned Evaluation Procedure Config reference."""


class EvalNodeError(ValueError):
    """The Graph Config does not contain exactly one Eval Node, or the Eval
    Node is malformed."""


def eval_node_procedure_hashes(graph: GraphConfig) -> list[str]:
    """Return the Evaluation Procedure Config Identity Hash of every Eval
    Node in ``graph`` (in graph node order)."""
    return [
        eval_node_procedure_hash(node.variables)
        for node in graph.nodes
        if node.node_type == EVAL_NODE_TYPE
    ]


def sole_eval_node_procedure_hash(graph: GraphConfig) -> str:
    """Return the single Eval Node's Evaluation Procedure Config Identity
    Hash, requiring exactly one Eval Node in the Graph Config."""
    hashes = eval_node_procedure_hashes(graph)
    if len(hashes) != 1:
        raise EvalNodeError(
            f"expected exactly one {EVAL_NODE_TYPE} node, found {len(hashes)}"
        )
    return hashes[0]


def eval_config_hash(eval_config: EvalConfig) -> str:
    """The composite Eval Config Identity Hash (``eval_config_hash``).

    This is dr-code's own ``config_identity_hash``; Whetstone adds no second
    identity for the Eval Config.
    """
    return eval_config.config_identity_hash


def validate_eval_identity_partition(
    graph: GraphConfig,
    eval_config: EvalConfig,
) -> None:
    """Validate the Eval identity partition invariant.

    The composite Eval Config's Evaluation Procedure Config identity MUST
    exactly match the Eval Node / Graph Config reference. Raises
    :class:`EvalIdentityMismatchError` on any mismatch.
    """
    node_procedure_hash = sole_eval_node_procedure_hash(graph)
    config_procedure_hash = eval_config.evaluation_procedure_config_hash
    if node_procedure_hash != config_procedure_hash:
        raise EvalIdentityMismatchError(
            "Eval Node Evaluation Procedure Config identity "
            f"{node_procedure_hash!r} does not match composite Eval Config "
            f"Procedure identity {config_procedure_hash!r}"
        )


__all__ = [
    "EvalIdentityMismatchError",
    "EvalNodeError",
    "eval_config_hash",
    "eval_node_procedure_hashes",
    "sole_eval_node_procedure_hash",
    "validate_eval_identity_partition",
]
