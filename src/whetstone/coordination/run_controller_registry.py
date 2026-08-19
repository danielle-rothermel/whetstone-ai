from __future__ import annotations

from typing import TYPE_CHECKING

from whetstone.core.identity import require_full_hash

if TYPE_CHECKING:
    from whetstone.coordination.run_workflow import RunController, RunRequest


class RunWorkflowError(RuntimeError):
    """The configured parent run-workflow boundary is invalid."""


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


def registered_controller(request: RunRequest) -> RunController:
    try:
        controller = _CONTROLLERS[request.controller_identity_hash]
    except KeyError:
        raise RunWorkflowError(
            "run controller is not registered before DBOS launch"
        ) from None
    if controller.runtime_hash != request.controller_identity_hash:
        raise RunWorkflowError("registered run controller identity drifted")
    return controller


__all__ = [
    "RunWorkflowError",
    "register_run_controller",
    "registered_controller",
]
