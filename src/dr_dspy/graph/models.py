from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictStr,
    model_serializer,
    model_validator,
)

from dr_dspy.eval_failures.exceptions import EvalFailureError
from dr_dspy.eval_failures.types import FailureClass

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
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class GraphRunStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
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
        field_names = [field.name for field in self.fields]
        if len(field_names) != len(set(field_names)):
            raise ValueError("duplicate field names in node config")

        output_names = {field.name for field in self.output_fields()}
        if self.fields and self.output_field not in output_names:
            raise ValueError(
                f"output_field {self.output_field!r} is not an output field"
            )

        input_names = {field.name for field in self.input_fields()}
        for field_name in self.input_bindings:
            if input_names and field_name not in input_names:
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

    def node_ids(self) -> list[str]:
        return [node.id for node in self.nodes]

    def node(self, node_id: str) -> NodeSpec:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def topological_order(self) -> tuple[NodeSpec, ...]:
        validate_acyclic_graph(self.nodes)
        by_id = {node.id: node for node in self.nodes}
        done: set[str] = set()
        ordered: list[NodeSpec] = []
        while len(ordered) < len(self.nodes):
            ready = sorted(
                node_id
                for node_id, node in by_id.items()
                if node_id not in done and node.dependencies() <= done
            )
            if not ready:
                stuck = ", ".join(
                    sorted(node_id for node_id in by_id if node_id not in done)
                )
                raise GraphValidationError(f"graph has a cycle among: {stuck}")
            for node_id in ready:
                ordered.append(by_id[node_id])
                done.add(node_id)
        return tuple(ordered)

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
    failure_class: FailureClass | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_exception(cls, error: BaseException) -> NodeError:
        error_type = f"{type(error).__module__}.{type(error).__qualname__}"
        failure_class: FailureClass | None = None
        metadata: dict[str, Any] = {}
        if isinstance(error, EvalFailureError):
            failure_class = type(error).failure_class
            metadata = dict(error.metadata)
        return cls(
            error_type=error_type,
            message=str(error),
            failure_class=failure_class,
            metadata=metadata,
        )


class NodeOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: StrictStr
    status: NodeOutcomeStatus
    output: NodeOutput | None = None
    error: NodeError | None = None
    blocked_by: tuple[StrictStr, ...] = ()
    _exception: BaseException | None = PrivateAttr(default=None)

    @classmethod
    def success(cls, *, node_id: str, output: NodeOutput) -> NodeOutcome:
        return cls(
            node_id=node_id,
            status=NodeOutcomeStatus.SUCCESS,
            output=output,
        )

    @classmethod
    def failed(
        cls,
        *,
        node_id: str,
        error: BaseException,
    ) -> NodeOutcome:
        outcome = cls(
            node_id=node_id,
            status=NodeOutcomeStatus.FAILED,
            error=NodeError.from_exception(error),
        )
        outcome._exception = error
        return outcome

    @classmethod
    def skipped(
        cls,
        *,
        node_id: str,
        blocked_by: tuple[str, ...],
    ) -> NodeOutcome:
        return cls(
            node_id=node_id,
            status=NodeOutcomeStatus.SKIPPED,
            blocked_by=blocked_by,
        )

    @property
    def exception(self) -> BaseException | None:
        return self._exception

    @model_validator(mode="after")
    def validate_outcome(self) -> NodeOutcome:
        if self.status is NodeOutcomeStatus.SUCCESS and self.output is None:
            raise ValueError("successful node outcomes require output")
        if self.status is NodeOutcomeStatus.FAILED and self.error is None:
            raise ValueError("failed node outcomes require error")
        if (
            self.status is NodeOutcomeStatus.SKIPPED
            and not self.blocked_by
        ):
            raise ValueError("skipped node outcomes require blocked_by")
        return self


class TerminalError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: StrictStr
    status: NodeOutcomeStatus
    error: NodeError | None = None
    blocked_by: tuple[StrictStr, ...] = ()


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
    id_set = set(node_ids)
    if graph.terminal_node_id not in id_set:
        raise GraphValidationError(
            f"terminal_node_id {graph.terminal_node_id!r} not in graph"
        )
    for node in graph.nodes:
        for ref in node.config.input_bindings.values():
            validate_binding_ref(ref, id_set)
    validate_acyclic_graph(graph.nodes)


def validate_binding_ref(ref: BindingRef, node_ids: set[str]) -> None:
    if ref.source is BindingSource.TASK:
        return
    if ref.node_id not in node_ids:
        raise GraphValidationError(
            f"ref {ref.ref!r} points at unknown node {ref.node_id!r}"
        )


def validate_acyclic_graph(nodes: tuple[NodeSpec, ...]) -> None:
    node_ids = {node.id for node in nodes}
    done: set[str] = set()
    remaining = set(node_ids)
    while remaining:
        ready = {
            node.id
            for node in nodes
            if node.id in remaining and node.dependencies() <= done
        }
        if not ready:
            stuck = ", ".join(sorted(remaining))
            raise GraphValidationError(f"graph has a cycle among: {stuck}")
        done.update(ready)
        remaining.difference_update(ready)
