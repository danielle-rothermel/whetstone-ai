"""Parent run-workflow boundary tests.

The load-bearing claim these pin: the runner enters a DBOS workflow before
driving any optimizer step, exactly one parent workflow exists per run, and the
controller is bound by identity before launch.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

import pytest

_HASH = "a" * 64
_OTHER_HASH = "b" * 64


class _ReplayDbos:
    """A DBOS stand-in recording the workflow bodies the runner enters."""

    workflow_ids: ClassVar[list[str]] = []
    step_id: ClassVar[int | None] = None

    @classmethod
    def workflow(cls):
        def decorate(function):
            return function

        return decorate


class _SetWorkflowID:
    def __init__(self, workflow_id: str) -> None:
        self._workflow_id = workflow_id

    def __enter__(self):
        _ReplayDbos.workflow_ids.append(self._workflow_id)
        return self

    def __exit__(self, *_args):
        return False


def _load_module():
    """Load run_workflow with only its dbos decorator seam replaced."""
    fake_dbos = types.ModuleType("dbos")
    fake_dbos.__dict__["DBOS"] = _ReplayDbos
    fake_dbos.__dict__["SetWorkflowID"] = _SetWorkflowID
    prior = sys.modules.get("dbos")
    sys.modules["dbos"] = fake_dbos
    module_name = "_run_workflow_replay_test"
    try:
        path = Path("src/whetstone/coordination/run_workflow.py")
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)
        if prior is None:
            del sys.modules["dbos"]
        else:
            sys.modules["dbos"] = prior


@pytest.fixture(autouse=True)
def _reset_replay_state():
    _ReplayDbos.workflow_ids = []
    _ReplayDbos.step_id = None
    yield
    _ReplayDbos.workflow_ids = []
    _ReplayDbos.step_id = None


class _Controller:
    """A controller recording where it was driven from."""

    def __init__(self, result: Any, *, identity: str = _HASH) -> None:
        self.runtime_identity_hash = identity
        self._result = result
        self.drive_calls = 0
        self.observed_workflow_ids: list[str] = []

    def drive(self, request: Any) -> Any:
        self.drive_calls += 1
        self.observed_workflow_ids.append(
            _ReplayDbos.workflow_ids[-1]
            if _ReplayDbos.workflow_ids
            else "<no workflow>"
        )
        return self._result


def _request(module: Any, **overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "controller_identity_hash": _HASH,
        "run_id": "copro:c18:a0",
        "control_identity_hash": _OTHER_HASH,
    }
    fields.update(overrides)
    return module.RunRequest(**fields)


# --------------------------------------------------------------------------
# RunRequest identity
# --------------------------------------------------------------------------


def test_the_workflow_id_prefix_is_pinned() -> None:
    module = _load_module()

    assert module.RUN_WORKFLOW_ID_PREFIX == "whetstone-run-"
    assert module.RUN_WORKFLOW_SCHEMA == "whetstone.runner.parent_run"
    assert module.RUN_WORKFLOW_SCHEMA_VERSION == 1


def test_the_workflow_id_derives_from_the_request_identity() -> None:
    module = _load_module()
    request = _request(module)

    assert request.workflow_id() == (
        f"whetstone-run-{request.identity_hash()}"
    )


def test_a_changed_control_is_a_different_workflow() -> None:
    module = _load_module()
    original = _request(module)
    changed = _request(module, control_identity_hash="c" * 64)

    # A run whose control changed can never silently resume the prior run.
    assert changed.workflow_id() != original.workflow_id()


def test_a_changed_run_id_is_a_different_workflow() -> None:
    module = _load_module()

    assert (
        _request(module, run_id="copro:c18:a1").workflow_id()
        != _request(module).workflow_id()
    )


def test_the_same_request_always_hashes_to_the_same_workflow() -> None:
    module = _load_module()

    assert _request(module).workflow_id() == _request(module).workflow_id()


def test_the_request_is_serializable_across_the_dbos_boundary() -> None:
    # DBOS pickles workflow arguments, so a request carrying anything
    # unpicklable would fail only against a real database. Pin it here
    # against the real module, since a dynamically loaded one is unpicklable
    # for reasons that have nothing to do with the request's own fields.
    import pickle

    from whetstone.coordination.run_workflow import RunRequest

    request = RunRequest(
        controller_identity_hash=_HASH,
        run_id="copro:c18:a0",
        control_identity_hash=_OTHER_HASH,
    )

    assert pickle.loads(pickle.dumps(request)) == request


def test_a_truncated_identity_hash_is_refused() -> None:
    module = _load_module()

    with pytest.raises(ValueError):
        _request(module, control_identity_hash="abc")


def test_an_empty_run_id_is_refused() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="run_id must be non-empty"):
        _request(module, run_id="")


def test_extra_request_fields_are_refused() -> None:
    module = _load_module()

    with pytest.raises(ValueError):
        _request(module, surprise=1)


# --------------------------------------------------------------------------
# Registration before launch
# --------------------------------------------------------------------------


def test_an_unregistered_controller_is_refused(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_CONTROLLERS", {})

    with pytest.raises(
        module.RunWorkflowError, match="not registered before DBOS launch"
    ):
        module._parent_run_workflow(_request(module))


def test_registering_the_identical_controller_twice_is_a_no_op(
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_CONTROLLERS", {})
    controller = _Controller(None)

    first = module.register_run_controller(controller)
    second = module.register_run_controller(controller)

    assert first == second == _HASH


def test_binding_a_second_controller_to_one_identity_is_refused(
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_CONTROLLERS", {})
    module.register_run_controller(_Controller(None))

    with pytest.raises(module.RunWorkflowError, match="already bound"):
        module.register_run_controller(_Controller(None))


def test_a_controller_whose_identity_drifted_is_refused(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_CONTROLLERS", {})
    controller = _Controller(None)
    module.register_run_controller(controller)
    controller.runtime_identity_hash = "c" * 64

    with pytest.raises(module.RunWorkflowError, match="identity drifted"):
        module._parent_run_workflow(_request(module))


def test_a_controller_with_a_truncated_identity_is_refused(
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_CONTROLLERS", {})

    with pytest.raises(ValueError):
        module.register_run_controller(_Controller(None, identity="abc"))


# --------------------------------------------------------------------------
# The workflow boundary itself
# --------------------------------------------------------------------------


def test_the_parent_refuses_to_run_inside_a_step(monkeypatch) -> None:
    # The proposal executor cannot start a child workflow from inside a step,
    # so the parent must refuse that context rather than fail deeper.
    module = _load_module()
    monkeypatch.setattr(module, "_CONTROLLERS", {})
    _ReplayDbos.step_id = 3

    with pytest.raises(
        module.RunWorkflowError, match="cannot execute inside a DBOS step"
    ):
        module._parent_run_workflow(_request(module))


def test_the_runner_enters_exactly_one_workflow_per_run(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_CONTROLLERS", {})
    result = _identity_result()
    controller = _Controller(result)
    module.register_run_controller(controller)
    request = _request(module)

    module.DbosRunner().run(request)

    # Exactly one parent workflow; the runner never multiplies workflows.
    assert _ReplayDbos.workflow_ids == [request.workflow_id()]
    assert controller.drive_calls == 1


def test_the_controller_is_driven_inside_the_workflow(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_CONTROLLERS", {})
    controller = _Controller(_identity_result())
    module.register_run_controller(controller)
    request = _request(module)

    module.DbosRunner().run(request)

    # This is the settled decision the whole boundary exists for: the step
    # driving happens inside a workflow body, never outside one.
    assert controller.observed_workflow_ids == [request.workflow_id()]


def test_the_terminal_result_reference_round_trips(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_CONTROLLERS", {})
    reference = _identity_result()
    module.register_run_controller(_Controller(reference))

    returned = module.DbosRunner().run(_request(module))

    assert returned == reference


def test_a_controller_returning_a_foreign_ref_is_refused(monkeypatch) -> None:
    # The parent must return the run's terminal Optimization Result, never
    # some other durable record that happens to be at hand.
    from whetstone.core.identity import TypedRef

    module = _load_module()
    monkeypatch.setattr(module, "_CONTROLLERS", {})
    module.register_run_controller(
        _Controller(
            TypedRef(schema_name="whetstone.other", content_hash="d" * 64)
        )
    )

    with pytest.raises(
        module.RunWorkflowError, match="exact Optimization Result ref"
    ):
        module.DbosRunner().run(_request(module))


def test_the_terminal_reference_is_serializable(monkeypatch) -> None:
    # The whole reason the boundary carries a reference: the record itself is
    # built on ImmutableJsonObject and cannot cross a DBOS checkpoint.
    import pickle

    reference = _identity_result()

    assert pickle.loads(pickle.dumps(reference)) == reference


def _identity_result():
    """Drive one real identity run and return its terminal result ref."""
    import tempfile

    from tests.optimization.support import (
        make_harness,
        make_store,
        pure_request,
        registry,
    )
    from whetstone.optimization.contracts import step_result_reference

    directory = tempfile.mkdtemp()
    store = make_store(Path(directory))
    request = pure_request()
    harness = make_harness(
        store=store, adapter_registry=registry(), run=request.run
    )
    step, _ref = harness.run_step(request)
    _terminal, terminal_ref = harness.terminalize(
        run=request.run, step_results=(step_result_reference(step),)
    )
    return terminal_ref


# --------------------------------------------------------------------------
# Real DBOS
# --------------------------------------------------------------------------


@pytest.fixture
def _clean_dbos():
    """Clear the DBOS singleton around a real-DBOS test, keeping the registry.

    DBOS keeps a process-global singleton plus a decorator registry. The
    parent workflow is registered by ``@DBOS.workflow()`` at module import,
    which happens the first time anything imports the runner -- long before
    any test constructs a DBOS instance. Destroying the *registry* would
    therefore erase the parent workflow permanently for the rest of the
    session, so this clears only the singleton and leaves the registry intact.

    An earlier test in the session may nonetheless have cleared the registry
    -- ``destroy(destroy_registry=True)`` is the correct teardown for a test
    that declares its own inline decorators -- which would leave the parent
    workflow unregistered here. Reloading the module re-runs its decorator, so
    the workflow is registered again regardless of what ran before.
    """
    import importlib

    from dbos import DBOS

    DBOS.destroy()
    importlib.reload(
        importlib.import_module("whetstone.orchestration.run_workflow")
    )
    yield
    DBOS.destroy()


@pytest.mark.skipif(
    "WHETSTONE_TEST_POSTGRES_DSN" not in os.environ,
    reason="WHETSTONE_TEST_POSTGRES_DSN is required for real DBOS replay",
)
@pytest.mark.usefixtures("_clean_dbos")
def test_real_dbos_parent_replays_a_completed_run_without_redriving() -> None:
    from dbos import DBOS, DBOSConfig

    from whetstone.coordination.run_workflow import (
        DbosRunner,
        RunRequest,
        register_run_controller,
    )

    suffix = uuid4().hex[:10]
    database_url = os.environ["WHETSTONE_TEST_POSTGRES_DSN"]
    config: DBOSConfig = {
        "name": f"run-parent-{suffix}",
        "system_database_url": database_url,
        "application_database_url": database_url,
        "application_version": f"run-parent-{suffix}",
        "run_admin_server": False,
        "use_listen_notify": False,
    }
    DBOS(config=config)
    result = _identity_result()
    identity = f"{suffix:>064}".replace(" ", "0")
    controller = _Controller(result, identity=identity)
    register_run_controller(controller)
    request = RunRequest(
        controller_identity_hash=identity,
        run_id=f"identity:parent:{suffix}",
        control_identity_hash=_OTHER_HASH,
    )
    try:
        DBOS.launch()
        runner = DbosRunner()
        first = runner.run(request)
        replay = runner.run(request)

        assert replay == first
        # The completed parent checkpoint replays; the run is not re-driven.
        assert controller.drive_calls == 1
    finally:
        DBOS.destroy()


@pytest.mark.skipif(
    "WHETSTONE_TEST_POSTGRES_DSN" not in os.environ,
    reason="WHETSTONE_TEST_POSTGRES_DSN is required for real DBOS replay",
)
@pytest.mark.usefixtures("_clean_dbos")
def test_real_dbos_parent_recovers_and_redrives_from_run_start() -> None:
    from dbos import DBOS, DBOSClient, DBOSConfig
    from dbos._error import (
        DBOSAwaitedWorkflowCancelledError,
        DBOSWorkflowCancelledError,
    )

    from whetstone.coordination.run_workflow import (
        DbosRunner,
        RunRequest,
        register_run_controller,
    )

    suffix = uuid4().hex[:10]
    database_url = os.environ["WHETSTONE_TEST_POSTGRES_DSN"]
    config: DBOSConfig = {
        "name": f"run-recovery-{suffix}",
        "system_database_url": database_url,
        "application_database_url": database_url,
        "application_version": f"run-recovery-{suffix}",
        "run_admin_server": False,
        "use_listen_notify": False,
    }
    DBOS(config=config)
    completed_step_calls = 0

    @DBOS.step(retries_allowed=False)
    def completed_step(run_id: str) -> str:
        nonlocal completed_step_calls
        completed_step_calls += 1
        return run_id

    result = _identity_result()
    identity = f"{suffix:>064}".replace(" ", "0")

    class CrashingController:
        runtime_identity_hash = identity

        def __init__(self) -> None:
            self.drive_calls = 0
            self.crash_once = True

        def drive(self, request):
            self.drive_calls += 1
            completed_step(request.run_id)
            if self.crash_once:
                self.crash_once = False
                raise DBOSWorkflowCancelledError(
                    "injected interruption after a completed step"
                )
            return result

    controller = CrashingController()
    register_run_controller(controller)
    request = RunRequest(
        controller_identity_hash=identity,
        run_id=f"identity:recovery:{suffix}",
        control_identity_hash=_OTHER_HASH,
    )
    client = None
    try:
        DBOS.launch()
        runner = DbosRunner()
        with pytest.raises(DBOSAwaitedWorkflowCancelledError):
            runner.run(request)
        assert controller.drive_calls == completed_step_calls == 1

        client = DBOSClient(system_database_url=database_url)
        recovered = client.resume_workflow(request.workflow_id()).get_result()

        assert recovered is not None
        # The recovered parent re-enters the body and re-drives the run, but
        # the step that already checkpointed is not re-executed.
        assert controller.drive_calls == 2
        assert completed_step_calls == 1
    finally:
        if client is not None:
            client.destroy()
        # The registry is deliberately preserved: the parent workflow was
        # registered at import time and must survive for later tests.
        DBOS.destroy()
