"""CLI and startup-wiring tests.

These pin the settled decisions the startup site carries: one registration
site for the proposal transport (COPRO, MIPROv2 and GEPA alike), GEPA's factory
registered by the runner rather than by ``CanonicalGepaAdapterFactory.create``,
and a completed cell that never constructs a DBOS runtime at all.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.provider import support as provider_support
from tests.runner.support import cell_config
from whetstone.coordination import proposal_provider, run_workflow
from whetstone.coordination.proposal_provider import ProposalProviderError
from whetstone.core.effects.authority import ReplayPolicy
from whetstone.optimization.gepa import factory as gepa_factory_module
from whetstone.optimization.gepa import runner as gepa_runner
from whetstone.optimization.proposal.proposer import (
    DurableProposalExecutor,
    ProviderProposerTransport,
)
from whetstone.runner import cli as cli_module
from whetstone.runner.cli import (
    DBOS_SYSTEM_DATABASE_URL_ENV,
    RunnerLaunch,
    app,
    load_task_split_manifest,
    run_cell_command,
)
from whetstone.runner.startup import register_runtime


@pytest.fixture(autouse=True)
def _isolated_registries() -> Iterator[None]:
    """Each test starts with empty capability registries.

    The registries are process-global by design -- a recovered workflow must
    find what startup bound -- so a test that registers must not leak that
    binding into the next one. Each is cleared and restored in place, because
    the register functions close over the module global rather than looking it
    up through the module, so rebinding the attribute would not affect them.
    """
    transport_registry = proposal_provider._TRANSPORT_REGISTRY
    with transport_registry._lock:
        transports = dict(transport_registry._transports)
    controllers = dict(run_workflow._CONTROLLERS)
    factories = dict(gepa_runner._GEPA_FACTORIES)
    with transport_registry._lock:
        transport_registry._transports.clear()
    run_workflow._CONTROLLERS.clear()
    gepa_runner._GEPA_FACTORIES.clear()
    yield
    with transport_registry._lock:
        transport_registry._transports.clear()
        transport_registry._transports.update(transports)
    run_workflow._CONTROLLERS.clear()
    run_workflow._CONTROLLERS.update(controllers)
    gepa_runner._GEPA_FACTORIES.clear()
    gepa_runner._GEPA_FACTORIES.update(factories)


def _transport() -> ProviderProposerTransport:
    provider_config = provider_support.openrouter_chat_config(
        model="proposal-model"
    )
    transport_policy = provider_support.build_transport_policy()
    return ProviderProposerTransport(
        resolve_provider_call_config=lambda _ref: provider_config,
        transport=provider_support.RecordingTransport(
            request=provider_support.build_request(),
            transport_policy=transport_policy,
            outcomes=[],
        ),
        execution_policy=provider_support.build_execution_policy(
            max_attempts=1, transport_policy=transport_policy
        ),
        clock=provider_support.FakeClock(),
        sleep=provider_support.SleepRecorder(),
    )


# --------------------------------------------------------------------------
# The single registration site
# --------------------------------------------------------------------------


def test_register_runtime_binds_the_transport_and_mints_one_executor(
    tmp_path: Path,
) -> None:
    """The site registers the transport and builds the one shared executor."""
    transport = _transport()
    registered = register_runtime(transport=transport)

    assert registered.transport_registry_key == (
        transport.durability_identity_hash
    )
    assert type(registered.proposal_executor) is DurableProposalExecutor
    assert (
        registered.proposal_executor.recovery_policy
        is ReplayPolicy.DURABLE_WORKFLOW
    )
    # The executor resolves the exact object the site registered.
    assert (
        proposal_provider._registered_transport(
            registered.transport_registry_key
        )
        is transport
    )


def _resolve_controller(request):
    """Resolve through the live module, never an import-time reference.

    A real-DBOS test elsewhere reloads ``run_workflow`` to re-register its
    workflow decorator, which replaces the module's classes and its registry
    dict. Binding either at import time would leave this module asserting
    against objects the runner no longer uses.
    """
    return run_workflow._registered_controller(request)


def test_register_runtime_binds_every_run_controller(tmp_path: Path) -> None:
    """A registered controller is resolvable by the parent run workflow."""
    config = cell_config(tmp_path)
    controller = config.controller
    registered = register_runtime(
        transport=_transport(), controllers=(controller,)
    )

    assert registered.controller_identity_hashes == (controller.runtime_hash,)
    request = controller.control.run_request(
        controller_identity_hash=controller.runtime_hash
    )
    assert _resolve_controller(request) is controller


def test_an_unregistered_controller_is_refused_at_resolution(
    tmp_path: Path,
) -> None:
    """Registration must happen before launch; nothing resolves without it."""
    controller = cell_config(tmp_path).controller
    request = controller.control.run_request(
        controller_identity_hash=controller.runtime_hash
    )

    with pytest.raises(
        run_workflow.RunWorkflowError,
        match="not registered before DBOS launch",
    ):
        _resolve_controller(request)


def test_rebinding_a_different_transport_to_one_key_is_refused() -> None:
    """Identity-keyed registration is only safe while it stays injective."""
    first = _transport()
    register_runtime(transport=first)
    second = _transport()
    assert second.durability_identity_hash == first.durability_identity_hash

    with pytest.raises(ProposalProviderError, match="already bound"):
        register_runtime(transport=second)


def test_registering_the_identical_transport_again_is_a_no_op() -> None:
    """Re-entering startup with the same objects must not fail."""
    transport = _transport()
    first = register_runtime(transport=transport)
    second = register_runtime(transport=transport)

    assert first.transport_registry_key == second.transport_registry_key


def test_gepa_factory_create_registers_no_proposal_transport() -> None:
    """GEPA's transport registration lives at the runner's startup site.

    Pins the relocation: ``create`` binds GEPA's own authorities and nothing
    else, so the runner remains the single place a transport is registered.
    """
    source = inspect.getsource(gepa_factory_module)

    assert "register_proposal_transport" not in source
    assert "register_miprov2_proposal_transport" not in source


def test_the_runner_has_exactly_one_registration_site() -> None:
    """Only ``startup`` may call a ``register_*`` capability function."""
    runner_root = Path(cli_module.__file__).parent
    callers = {
        path.name
        for path in runner_root.glob("*.py")
        if "register_proposal_transport(" in path.read_text()
        or "register_gepa_adapter_factory(" in path.read_text()
        or "register_run_controller(" in path.read_text()
    }

    assert callers == {"startup.py"}


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


def test_default_system_database_is_per_cell_and_stable(
    tmp_path: Path,
) -> None:
    """Two invocations of one cell resolve the same system database."""
    config = cell_config(tmp_path)
    first = cli_module._default_dbos_database_url(config)
    second = cli_module._default_dbos_database_url(cell_config(tmp_path))
    other = cli_module._default_dbos_database_url(
        cell_config(tmp_path, attempt=1)
    )

    assert first == second
    assert first != other
    assert first.startswith("sqlite:///")


def test_executor_id_is_derived_from_the_cell_identity(
    tmp_path: Path,
) -> None:
    """Distinct cells never share a DBOS executor id."""
    first = cli_module._dbos_executor_id(cell_config(tmp_path))
    other = cli_module._dbos_executor_id(cell_config(tmp_path, attempt=1))

    assert first != other
    assert len(first) == 64


def test_the_flag_and_env_var_resolve_the_system_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flag beats env var beats the per-cell default, in that order."""
    captured: list[str] = []

    class _RecordingDbos:
        """Records the config the lifecycle built, then stops the run."""

        def __init__(self, *, config: dict[str, str]) -> None:
            captured.append(config["system_database_url"])

        @staticmethod
        def launch() -> None:
            raise _LaunchReached

        @staticmethod
        def destroy() -> None:
            pass

    import dbos

    monkeypatch.setattr(dbos, "DBOS", _RecordingDbos)
    # One process registers one transport; re-registering the identical object
    # is the no-op startup relies on across invocations.
    transport = _transport()

    env_only = {DBOS_SYSTEM_DATABASE_URL_ENV: "sqlite:///env.db"}
    for flag, environ in (
        ("sqlite:///flag.db", env_only),
        (None, env_only),
        (None, {}),
    ):
        root = tmp_path / str(len(captured))
        root.mkdir()
        launch = RunnerLaunch(cell=cell_config(root), transport=transport)
        with pytest.raises(_LaunchReached):
            run_cell_command(launch, system_database_url=flag, environ=environ)

    assert captured[0] == "sqlite:///flag.db"
    assert captured[1] == "sqlite:///env.db"
    assert captured[2].startswith("sqlite:///")
    assert "/dbos/" in captured[2]


class _LaunchReached(Exception):
    """Raised once the lifecycle reached ``DBOS.launch()``."""


def test_factory_path_must_name_a_callable(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["cell", "--factory", "not-a-path"])

    assert result.exit_code != 0


def test_task_split_manifest_reports_an_unreadable_path(
    tmp_path: Path,
) -> None:
    """The folded-in loader names the file it could not read."""
    from whetstone.experiment.task_selection import TaskSplitManifestError

    with pytest.raises(TaskSplitManifestError, match="cannot read"):
        load_task_split_manifest(tmp_path / "absent.json")
