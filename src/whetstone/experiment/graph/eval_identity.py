from __future__ import annotations

from typing import TYPE_CHECKING

from dr_graph import GraphConfig

from whetstone.experiment.graph.nodes import (
    EVAL_NODE_TYPE,
    eval_node_procedure_hash,
)

if TYPE_CHECKING:
    from whetstone.evaluation import EvalConfig


class EvalIdentityMismatchError(ValueError):
    pass


class EvalNodeError(ValueError):
    pass


def eval_node_procedure_hashes(graph: GraphConfig) -> list[str]:
    return [
        eval_node_procedure_hash(node.variables)
        for node in graph.nodes
        if node.node_type == EVAL_NODE_TYPE
    ]


def sole_eval_node_procedure_hash(graph: GraphConfig) -> str:
    hashes = eval_node_procedure_hashes(graph)
    if len(hashes) != 1:
        raise EvalNodeError(
            f"expected exactly one {EVAL_NODE_TYPE} node, found {len(hashes)}"
        )
    return hashes[0]


def eval_config_hash(eval_config: EvalConfig) -> str:
    return eval_config.config_hash


def validate_eval_identity_partition(
    graph: GraphConfig,
    eval_config: EvalConfig,
) -> None:
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
