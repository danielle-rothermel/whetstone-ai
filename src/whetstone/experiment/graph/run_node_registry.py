from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dr_graph import NodeConfig, RunNode

from whetstone.experiment.graph.llm_call_run_node import (
    EvalRunNodeDeps,
    LlmCallRunNodeDeps,
    build_eval_run_node,
    build_llm_call_run_node,
)
from whetstone.experiment.graph.nodes import EVAL_NODE_TYPE, LLM_CALL_NODE_TYPE

__all__ = ["RunNodeRegistry", "build_run_node"]


class RunNodeRegistry:
    __slots__ = ("_handlers",)

    def __init__(self) -> None:
        self._handlers: dict[str, RunNode] = {}

    def register(self, node_type: str, handler: RunNode) -> None:
        if not node_type:
            raise ValueError("node_type must be non-empty")
        self._handlers[node_type] = handler

    def dispatch(
        self, node: NodeConfig, node_inputs: Mapping[str, Any]
    ) -> Any:
        handler = self._handlers.get(node.node_type)
        if handler is None:
            raise ValueError(
                f"no RunNode handler registered for node type {node.node_type!r}"
            )
        return handler(node, node_inputs)


def build_run_node(
    *,
    llm_deps: LlmCallRunNodeDeps,
    eval_deps: EvalRunNodeDeps,
) -> RunNode:
    """Build one dispatching RunNode for the canonical LLM and Eval node types."""
    registry = RunNodeRegistry()
    registry.register(LLM_CALL_NODE_TYPE, build_llm_call_run_node(llm_deps))
    registry.register(EVAL_NODE_TYPE, build_eval_run_node(eval_deps))

    def run_node(node: NodeConfig, node_inputs: Mapping[str, Any]) -> Any:
        return registry.dispatch(node, node_inputs)

    return run_node
