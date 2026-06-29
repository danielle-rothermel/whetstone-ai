from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from dr_dspy.graph.models import (
    BindingSource,
    GraphRunResult,
    GraphRunStatus,
    GraphSpec,
    InputResolutionError,
    NodeExecutionError,
    NodeOutcome,
    NodeOutcomeStatus,
    NodeOutput,
    NodeSpec,
    TerminalError,
)

type RunNode = Callable[[NodeSpec, Mapping[str, Any]], NodeOutput]


def execute_graph(
    *,
    graph: GraphSpec,
    inputs: Mapping[str, Any],
    run_node: RunNode,
) -> GraphRunResult:
    outcomes: dict[str, NodeOutcome] = {}
    execution_order: list[str] = []

    for node in graph.topological_order():
        execution_order.append(node.id)
        blocked_by = _blocked_dependencies(node, outcomes)
        if blocked_by:
            outcomes[node.id] = NodeOutcome.blocked(
                node_id=node.id,
                blocked_by=blocked_by,
            )
            continue

        try:
            node_inputs = resolve_node_inputs(
                node=node,
                task_inputs=inputs,
                outcomes=outcomes,
                graph=graph,
            )
            output = _run_node(
                node=node,
                node_inputs=node_inputs,
                run_node=run_node,
            )
        except Exception as error:
            outcomes[node.id] = NodeOutcome.from_error(
                node_id=node.id,
                error=error,
            )
            continue

        outcomes[node.id] = NodeOutcome.success(
            node_id=node.id,
            output=output,
        )

    return _build_result(
        graph=graph,
        outcomes=outcomes,
        execution_order=tuple(execution_order),
    )


def resolve_node_inputs(
    *,
    node: NodeSpec,
    task_inputs: Mapping[str, Any],
    outcomes: Mapping[str, NodeOutcome],
    graph: GraphSpec,
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for field_name, ref in node.config.input_bindings.items():
        if ref.source is BindingSource.TASK:
            if ref.field not in task_inputs:
                raise InputResolutionError(
                    f"missing task input {ref.field!r} for node {node.id!r}"
                )
            resolved[field_name] = task_inputs[ref.field]
            continue

        if ref.node_id is None:
            raise InputResolutionError(
                f"node input ref {ref.ref!r} has no node id"
            )
        upstream = outcomes.get(ref.node_id)
        if (
            upstream is None
            or upstream.status is not NodeOutcomeStatus.SUCCESS
        ):
            raise InputResolutionError(
                f"upstream node {ref.node_id!r} did not succeed"
            )
        if upstream.output is None:
            raise InputResolutionError(
                f"upstream node {ref.node_id!r} has no output"
            )
        output_field = ref.field or graph.node(ref.node_id).config.output_field
        if output_field not in upstream.output.values:
            raise InputResolutionError(
                f"upstream node {ref.node_id!r} output missing "
                f"field {output_field!r}"
            )
        resolved[field_name] = upstream.output.values[output_field]
    return resolved


def _run_node(
    *,
    node: NodeSpec,
    node_inputs: Mapping[str, Any],
    run_node: RunNode,
) -> NodeOutput:
    output = NodeOutput.model_validate(run_node(node, node_inputs))
    if node.config.output_field not in output.values:
        raise NodeExecutionError(
            f"node {node.id!r} output missing field "
            f"{node.config.output_field!r}"
        )
    return output


def _blocked_dependencies(
    node: NodeSpec,
    outcomes: Mapping[str, NodeOutcome],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            dependency
            for dependency in node.dependencies()
            if outcomes[dependency].status is not NodeOutcomeStatus.SUCCESS
        )
    )


def _build_result(
    *,
    graph: GraphSpec,
    outcomes: dict[str, NodeOutcome],
    execution_order: tuple[str, ...],
) -> GraphRunResult:
    terminal = outcomes[graph.terminal_node_id]
    terminal_output: Any | None = None
    terminal_error: TerminalError | None = None

    if terminal.status is NodeOutcomeStatus.SUCCESS:
        if terminal.output is not None:
            terminal_output = terminal.output.values[
                graph.node(graph.terminal_node_id).config.output_field
            ]
    else:
        terminal_error = TerminalError(
            node_id=terminal.node_id,
            status=terminal.status,
            error=terminal.error,
            blocked_by=terminal.blocked_by,
        )

    return GraphRunResult(
        status=_graph_status(
            terminal=terminal,
            outcomes=outcomes,
        ),
        outcomes=outcomes,
        execution_order=execution_order,
        terminal_node_id=graph.terminal_node_id,
        terminal_output=terminal_output,
        terminal_error=terminal_error,
    )


def _graph_status(
    *,
    terminal: NodeOutcome,
    outcomes: Mapping[str, NodeOutcome],
) -> GraphRunStatus:
    if terminal.status is not NodeOutcomeStatus.SUCCESS:
        if terminal.status is NodeOutcomeStatus.BLOCKED:
            return GraphRunStatus.BLOCKED
        return GraphRunStatus.ERROR
    if any(
        outcome.status is not NodeOutcomeStatus.SUCCESS
        for outcome in outcomes.values()
    ):
        return GraphRunStatus.PARTIAL
    return GraphRunStatus.SUCCESS
