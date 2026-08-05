"""Stable parent DBOS workflow boundary for one optimization run.

The runner owns the DBOS workflow context, and this module is where it is
entered. One optimization run executes inside exactly one parent workflow whose
ID derives from the run request's ``identity_hash()``, so a recovered process
resumes the same run rather than starting a second one. Everything the parent
body needs is bound by identity before ``DBOS.launch``: the body resolves its
controller from a registry keyed by the controller's ``runtime_identity_hash``,
so a registered identity can never detach from the capability a recovered
workflow invokes.

**One parent per run.** The runner adds exactly this one workflow per run and
never wraps individual optimizer steps in additional child workflows. The
proposal executor spawns its own child workflow per logical proposal call,
which is that layer's contract and not something this boundary multiplies.

**Why the boundary lives here.** The proposal executor refuses to run outside a
workflow body and refuses to start a child workflow from inside a step. Driving
the optimizer from this parent body satisfies both conditions by construction,
which is what lets the optimization harness stay DBOS-unaware: nothing under
``whetstone.optimization`` imports dbos.

Guarantee, stated honestly. The parent gives replay from run start: a recovered
parent re-enters the body and re-drives the controller, which resolves already
durable step results from the harness rather than re-executing them. Recovery
is not free of every re-execution -- the proposal executor's own accepted
at-least-once window still applies to a call whose provider had already served
it when the process died -- and this boundary does not close that window.

**The boundary carries references, not records.** DBOS checkpoints workflow
arguments and return values by pickling them, and the optimization schema types
are built on ``ImmutableJsonObject``, which wraps a ``mappingproxy`` and is not
picklable. So the parent takes a request of plain strings and returns the
terminal ``OptimizationResult``'s exact ``TypedRef``. That is not merely a
workaround: the result is already durable and content-addressed in the
ObjectStore before the workflow returns, so the reference is the authoritative
handle and shipping a second copy through the checkpoint would duplicate it.
Callers resolve the record through the harness.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dbos import DBOS, SetWorkflowID
from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

from whetstone.core.identity import (
    TypedRef,
    compute_identity_hash,
    require_full_hash,
)
from whetstone.optimization.contracts import OPTIMIZATION_RESULT_SCHEMA

RUN_WORKFLOW_SCHEMA = "whetstone.runner.parent_run"
RUN_WORKFLOW_SCHEMA_VERSION = 1

#: Persisted identity contract: the parent workflow ID prefix. A recovered
#: process addresses a run by this exact string plus the request identity, so
#: renaming it orphans every in-flight run.
RUN_WORKFLOW_ID_PREFIX = "whetstone-run-"


class RunWorkflowError(RuntimeError):
    """The configured parent run-workflow boundary is invalid."""


@runtime_checkable
class RunController(Protocol):
    """The optimizer-agnostic capability the parent workflow drives.

    One controller owns one optimizer's harness, adapters, and durable control
    record. ``drive`` is the whole run: bind the run, drive steps until a
    non-continuing status, and terminalize, returning the exact reference of
    the terminal result it bound. It must be replay-safe, because a recovered
    parent calls it again from the top.
    """

    @property
    def runtime_identity_hash(self) -> str: ...

    def drive(self, request: RunRequest) -> TypedRef: ...


class RunRequest(BaseModel):
    """Complete serializable input identifying one optimization run.

    ``controller_identity_hash`` binds the exact registered controller;
    ``run_id`` is the run the harness binds; ``control_identity_hash`` is the
    content hash of the controller's durable control record, which covers the
    candidates, budget, algorithm parameters, eval configs, tools, and proposer
    config.

    The request is the whole workflow identity: a changed control under the
    same ``run_id`` hashes to a different workflow, so it can never silently
    resume a run that was configured differently.

    Every field is a plain string. DBOS pickles workflow arguments, so the
    request deliberately carries hashes rather than the rich control objects
    they identify: the controller resolves the full control from its own
    durable record, and the boundary stays trivially serializable.
    """

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
    """Bind one run controller under its exact runtime identity.

    Registration happens at runner startup, before ``DBOS.launch``, so a
    recovered parent workflow always finds the controller it was built with.
    Re-registering the identical object is a no-op; binding a different object
    to an already-bound identity is refused.
    """
    identity_hash = controller.runtime_identity_hash
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
    if controller.runtime_identity_hash != request.controller_identity_hash:
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
