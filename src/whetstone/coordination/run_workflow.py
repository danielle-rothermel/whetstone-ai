"""Stable parent DBOS workflow boundary for one optimization run.

One optimization run executes inside exactly one parent workflow whose ID
derives from the run request's ``identity_hash()``, so a recovered process
resumes the same run rather than starting a second one. The parent resolves
its controller from a registry keyed by the controller's ``runtime_hash``.

The boundary carries references, not records. DBOS checkpoints workflow
arguments and return values by pickling them, so the parent takes a request
of plain strings and returns the terminal ``OptimizationResult``'s exact
``TypedRef``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

try:
    from dbos import DBOS, SetWorkflowID
except ImportError as exc:
    raise ImportError(
        "DBOS coordination requires the optional dbos extra: "
        "pip install 'whetstone-ai[dbos]'"
    ) from exc
from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

from whetstone.core.identity import (
    TypedRef,
    compute_identity_hash,
    require_full_hash,
)
from whetstone.optimization.contracts import OPTIMIZATION_RESULT_SCHEMA

RUN_WORKFLOW_SCHEMA = "whetstone.coordination.parent_run"
RUN_WORKFLOW_SCHEMA_VERSION = 1

RUN_WORKFLOW_ID_PREFIX = "whetstone-run-"


class RunWorkflowError(RuntimeError):
    """The configured parent run-workflow boundary is invalid."""


@runtime_checkable
class RunController(Protocol):
    """The optimizer-agnostic capability the parent workflow drives."""

    @property
    def runtime_hash(self) -> str: ...

    def drive(self, request: RunRequest) -> TypedRef: ...


class RunRequest(BaseModel):
    """Complete serializable input identifying one optimization run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    controller_identity_hash: StrictStr
    run_id: StrictStr
    control_identity_hash: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> RunRequest:
        require_full_hash(
            self.controller_identity_hash,
            field="controller_identity_hash",
        )
        require_full_hash(
            self.control_identity_hash,
            field="control_identity_hash",
        )
        if not self.run_id:
            raise ValueError("run request run_id must be non-empty")
        return self

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=RUN_WORKFLOW_SCHEMA,
            schema_version=RUN_WORKFLOW_SCHEMA_VERSION,
            payload=self.model_dump(mode="json"),
        )

    def workflow_id(self) -> str:
        return f"{RUN_WORKFLOW_ID_PREFIX}{self.identity_hash()}"


_CONTROLLERS: dict[str, RunController] = {}


def register_run_controller(controller: RunController) -> str:
    """Bind one run controller under its exact runtime identity."""
    identity_hash = controller.runtime_hash
    require_full_hash(identity_hash, field="controller_identity_hash")
    existing = _CONTROLLERS.get(identity_hash)
    if existing is not None and existing is not controller:
        raise RunWorkflowError("run controller identity is already bound")
    _CONTROLLERS[identity_hash] = controller
    return identity_hash


def _registered_controller(request: RunRequest) -> RunController:
    try:
        controller = _CONTROLLERS[request.controller_identity_hash]
    except KeyError:
        raise RunWorkflowError(
            "run controller is not registered before DBOS launch"
        ) from None
    if controller.runtime_hash != request.controller_identity_hash:
        raise RunWorkflowError("registered run controller identity drifted")
    return controller


def _validated_result_ref(reference: TypedRef) -> TypedRef:
    exact = TypedRef.model_validate(reference.model_dump(mode="json"))
    if exact.schema_name != OPTIMIZATION_RESULT_SCHEMA:
        raise RunWorkflowError(
            "a run must terminalize into an exact Optimization Result ref"
        )
    return exact


@DBOS.workflow()
def _parent_run_workflow(request: RunRequest) -> TypedRef:
    """Drive one whole optimization run inside one parent workflow body."""
    if DBOS.step_id is not None:
        raise RunWorkflowError(
            "the parent run workflow cannot execute inside a DBOS step"
        )
    controller = _registered_controller(request)
    return _validated_result_ref(controller.drive(request))


class DbosRunner:
    """Launch one stable parent workflow per optimization run."""

    def run(self, request: RunRequest) -> TypedRef:
        """Execute the run and return its terminal Optimization Result ref."""
        with SetWorkflowID(request.workflow_id()):
            reference = _parent_run_workflow(request)
        return _validated_result_ref(reference)


__all__ = [
    "RUN_WORKFLOW_ID_PREFIX",
    "RUN_WORKFLOW_SCHEMA",
    "RUN_WORKFLOW_SCHEMA_VERSION",
    "DbosRunner",
    "RunController",
    "RunRequest",
    "RunWorkflowError",
    "register_run_controller",
]
