from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    model_serializer,
    model_validator,
)

TASK_SOURCE = "task"
REF_SEPARATOR = "."


class NodeOp(StrEnum):
    LLM_CALL = "llm_call"


class FieldRole(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


class FieldType(StrEnum):
    STRING = "str"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    CODE = "code"
    JSON = "json"


class BindingSource(StrEnum):
    TASK = "task"
    NODE = "node"


class NodeOutcomeStatus(StrEnum):
    """Runner outcome states, not append-only node-attempt row states.

    BLOCKED means the node was not invoked because an upstream dependency did
    not succeed. Persistence wrappers should not store BLOCKED as a node
    attempt outcome; it is derivable from the graph and upstream outcomes.
    """

    SUCCESS = "success"
    ERROR = "error"
    BLOCKED = "blocked"


class GraphRunStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    BLOCKED = "blocked"
    PARTIAL = "partial"


class BindingRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: BindingSource
    field: StrictStr | None = None
    node_id: StrictStr | None = None

    @model_validator(mode="before")
    @classmethod
    def parse_ref(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        head, separator, tail = value.partition(REF_SEPARATOR)
        if not separator:
            return {
                "source": BindingSource.NODE,
                "node_id": head,
            }
        if head == TASK_SOURCE:
            return {
                "source": BindingSource.TASK,
                "field": tail,
            }
        return {
            "source": BindingSource.NODE,
            "node_id": head,
            "field": tail,
        }

    @model_validator(mode="after")
    def validate_shape(self) -> BindingRef:
        if self.source is BindingSource.TASK:
            if self.node_id is not None:
                raise ValueError("task binding refs cannot include node_id")
            if not self.field:
                raise ValueError("task binding refs require a field")
            return self
        if not self.node_id:
            raise ValueError("node binding refs require node_id")
        if self.field is not None and not self.field:
            raise ValueError("node binding refs require a non-empty field")
        return self

    @model_serializer(mode="plain")
    def serialize_ref(self) -> str:
        return self.ref

    @property
    def ref(self) -> str:
        if self.source is BindingSource.TASK:
            return f"{TASK_SOURCE}{REF_SEPARATOR}{self.field}"
        if self.field is None:
            return str(self.node_id)
        return f"{self.node_id}{REF_SEPARATOR}{self.field}"

    @property
    def dependency_node_id(self) -> str | None:
        if self.source is BindingSource.NODE:
            return self.node_id
        return None


class FieldSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: StrictStr
    role: FieldRole
    type_name: FieldType = FieldType.STRING
    description: StrictStr | None = None


class NodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fields: tuple[FieldSpec, ...] = ()
    input_bindings: dict[str, BindingRef] = Field(default_factory=dict)
    output_field: StrictStr
    # Included in graph_digest via GraphSpec.model_dump; keep payloads small
    # until schema freeze adds explicit size/type constraints.
    parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def input_fields(self) -> tuple[FieldSpec, ...]:
        return tuple(
            field
            for field in self.fields
            if field.role is FieldRole.INPUT
        )

    def output_fields(self) -> tuple[FieldSpec, ...]:
        return tuple(
            field for field in self.fields if field.role is FieldRole.OUTPUT
        )

    @model_validator(mode="after")
    def validate_fields(self) -> NodeConfig:
        if not self.fields:
            raise ValueError("node config must declare at least one field")

        field_names = [field.name for field in self.fields]
        if len(field_names) != len(set(field_names)):
            raise ValueError("duplicate field names in node config")

        output_names = {field.name for field in self.output_fields()}
        if self.output_field not in output_names:
            raise ValueError(
                f"output_field {self.output_field!r} is not an output field"
            )

        input_names = {field.name for field in self.input_fields()}
        for field_name in self.input_bindings:
            if field_name not in input_names:
                raise ValueError(
                    f"input binding {field_name!r} is not an input field"
                )
        return self


class NodeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: StrictStr
    config: NodeConfig
    op: NodeOp = NodeOp.LLM_CALL

    def dependencies(self) -> set[str]:
        return {
            node_id
            for ref in self.config.input_bindings.values()
            if (node_id := ref.dependency_node_id) is not None
        }


class GraphSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: tuple[NodeSpec, ...]
    terminal_node_id: StrictStr
    task_fields: tuple[StrictStr, ...] = ()

    def node_ids(self) -> list[str]:
        return [node.id for node in self.nodes]

    def node(self, node_id: str) -> NodeSpec:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def topological_order(self) -> tuple[NodeSpec, ...]:
        return topological_order(self.nodes)

    @model_validator(mode="after")
    def validate_graph(self) -> GraphSpec:
        validate_graph_spec(self)
        return self


class NodeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


class NodeError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error_type: StrictStr
    message: StrictStr
    failure_class: StrictStr | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_exception(cls, error: BaseException) -> NodeError:
        return cls(
            error_type=_exception_error_type(error),
            message=str(error),
            failure_class=_exception_failure_class(error),
            metadata=_exception_metadata(error),
        )


class NodeOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: StrictStr
    status: NodeOutcomeStatus
    output: NodeOutput | None = None
    error: NodeError | None = None
    blocked_by: tuple[StrictStr, ...] = ()

    @classmethod
    def success(cls, *, node_id: str, output: NodeOutput) -> NodeOutcome:
        return cls(
            node_id=node_id,
            status=NodeOutcomeStatus.SUCCESS,
            output=output,
        )

    @classmethod
    def from_error(
        cls,
        *,
        node_id: str,
        error: BaseException,
    ) -> NodeOutcome:
        return cls(
            node_id=node_id,
            status=NodeOutcomeStatus.ERROR,
            error=NodeError.from_exception(error),
        )

    @classmethod
    def blocked(
        cls,
        *,
        node_id: str,
        blocked_by: tuple[str, ...],
    ) -> NodeOutcome:
        return cls(
            node_id=node_id,
            status=NodeOutcomeStatus.BLOCKED,
            blocked_by=blocked_by,
        )

    @model_validator(mode="after")
    def validate_outcome(self) -> NodeOutcome:
        if self.status is NodeOutcomeStatus.SUCCESS:
            if self.output is None:
                raise ValueError("successful node outcomes require output")
            if self.error is not None:
                raise ValueError(
                    "successful node outcomes cannot include error"
                )
            if self.blocked_by:
                raise ValueError(
                    "successful node outcomes cannot include blocked_by"
                )
            return self
        if self.status is NodeOutcomeStatus.ERROR:
            if self.error is None:
                raise ValueError("error node outcomes require error")
            if self.output is not None:
                raise ValueError("error node outcomes cannot include output")
            if self.blocked_by:
                raise ValueError(
                    "error node outcomes cannot include blocked_by"
                )
            return self
        if not self.blocked_by:
            raise ValueError("blocked node outcomes require blocked_by")
        if self.output is not None:
            raise ValueError("blocked node outcomes cannot include output")
        if self.error is not None:
            raise ValueError("blocked node outcomes cannot include error")
        return self


class TerminalError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: StrictStr
    status: NodeOutcomeStatus
    error: NodeError | None = None
    blocked_by: tuple[StrictStr, ...] = ()

    @model_validator(mode="after")
    def validate_terminal_error(self) -> TerminalError:
        if self.status is NodeOutcomeStatus.ERROR:
            if self.error is None:
                raise ValueError("error terminal outcomes require error")
            if self.blocked_by:
                raise ValueError(
                    "error terminal outcomes cannot include blocked_by"
                )
            return self
        if self.status is NodeOutcomeStatus.BLOCKED:
            if not self.blocked_by:
                raise ValueError(
                    "blocked terminal outcomes require blocked_by"
                )
            if self.error is not None:
                raise ValueError(
                    "blocked terminal outcomes cannot include error"
                )
            return self
        if self.error is not None:
            raise ValueError(
                "successful terminal outcomes cannot include error"
            )
        if self.blocked_by:
            raise ValueError(
                "successful terminal outcomes cannot include blocked_by"
            )
        return self


class GraphRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: GraphRunStatus
    outcomes: dict[str, NodeOutcome]
    execution_order: tuple[StrictStr, ...]
    terminal_node_id: StrictStr
    terminal_output: Any | None = None
    terminal_error: TerminalError | None = None


class GraphExecutionError(Exception):
    """Base exception for pure graph execution errors."""


class GraphValidationError(GraphExecutionError, ValueError):
    pass


class InputResolutionError(GraphExecutionError):
    pass


class NodeExecutionError(GraphExecutionError):
    pass


def validate_graph_spec(graph: GraphSpec) -> None:
    if not graph.nodes:
        raise GraphValidationError("graph must have at least one node")
    node_ids = graph.node_ids()
    if len(node_ids) != len(set(node_ids)):
        raise GraphValidationError("duplicate node ids")
    nodes_by_id = {node.id: node for node in graph.nodes}
    if graph.terminal_node_id not in nodes_by_id:
        raise GraphValidationError(
            f"terminal_node_id {graph.terminal_node_id!r} not in graph"
        )
    for node in graph.nodes:
        for ref in node.config.input_bindings.values():
            validate_binding_ref(ref, nodes_by_id)
    validate_task_binding_fields(graph)
    validate_acyclic_graph(graph.nodes)


def validate_task_binding_fields(graph: GraphSpec) -> None:
    task_binding_fields = {
        ref.field
        for node in graph.nodes
        for ref in node.config.input_bindings.values()
        if ref.source is BindingSource.TASK and ref.field is not None
    }
    if not task_binding_fields:
        return
    if not graph.task_fields:
        raise GraphValidationError(
            "graph uses task bindings but declares no task_fields; "
            "set task_fields to the allowed task input names"
        )
    allowed = set(graph.task_fields)
    unknown = sorted(task_binding_fields - allowed)
    if unknown:
        unknown_list = ", ".join(repr(field) for field in unknown)
        raise GraphValidationError(
            f"task binding field(s) {unknown_list} not in task_fields"
        )


def validate_binding_ref(
    ref: BindingRef,
    nodes_by_id: dict[str, NodeSpec],
) -> None:
    if ref.source is BindingSource.TASK:
        return
    if ref.node_id not in nodes_by_id:
        raise GraphValidationError(
            f"ref {ref.ref!r} points at unknown node {ref.node_id!r}"
        )
    if ref.field is None:
        return
    source_node = nodes_by_id[ref.node_id]
    output_names = {
        field.name for field in source_node.config.output_fields()
    }
    if ref.field not in output_names:
        raise GraphValidationError(
            f"ref {ref.ref!r} points at unknown field {ref.field!r} "
            f"on node {ref.node_id!r}"
        )


def validate_acyclic_graph(nodes: tuple[NodeSpec, ...]) -> None:
    topological_order(nodes)


def topological_order(nodes: tuple[NodeSpec, ...]) -> tuple[NodeSpec, ...]:
    node_ids = {node.id for node in nodes}
    by_id = {node.id: node for node in nodes}
    done: set[str] = set()
    remaining = set(node_ids)
    ordered: list[NodeSpec] = []
    while remaining:
        ready = sorted(
            node_id
            for node_id in remaining
            if by_id[node_id].dependencies() <= done
        )
        if not ready:
            stuck = ", ".join(sorted(remaining))
            raise GraphValidationError(f"graph has a cycle among: {stuck}")
        ordered.extend(by_id[node_id] for node_id in ready)
        done.update(ready)
        remaining.difference_update(ready)
    return tuple(ordered)


def _exception_failure_class(error: BaseException) -> str | None:
    failure_class = getattr(error, "failure_class", None)
    if isinstance(failure_class, StrEnum):
        return failure_class.value
    if isinstance(failure_class, str):
        return failure_class
    failure_class = getattr(type(error), "failure_class", None)
    if isinstance(failure_class, StrEnum):
        return failure_class.value
    if isinstance(failure_class, str):
        return failure_class
    return None


def _exception_error_type(error: BaseException) -> str:
    error_type = getattr(error, "error_type", None)
    if isinstance(error_type, str):
        return error_type
    return f"{type(error).__module__}.{type(error).__qualname__}"


def _exception_metadata(error: BaseException) -> dict[str, Any]:
    metadata = getattr(error, "metadata", None)
    result = dict(metadata) if isinstance(metadata, dict) else {}
    if getattr(error, "underlying", None) is not None:
        from dr_dspy.eval_failures.policy import (
            underlying_exception_type_name,
        )

        result.setdefault(
            "underlying_exception_type",
            underlying_exception_type_name(error),
        )
    return result
