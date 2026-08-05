"""Durability contract for the best-effort proposal-provider boundary."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing
import os
import sys
import types
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock, get_ident
from typing import Any, ClassVar
from uuid import uuid4

import pytest
from dr_providers import FailureClass

from tests.optimization.processes import terminate_processes
from tests.optimization.support import candidate
from tests.provider import support as provider_support
from whetstone.core.effects.authority import ReplayPolicy
from whetstone.core.identity import IdentityRef, typed_ref_for_record
from whetstone.experiment.candidate import candidate_reference
from whetstone.optimization.proposal.proposer import (
    ProposalRequest,
    ProposerConfig,
    ProviderProposerTransport,
)
from whetstone.provider.language_model import PlainPromptAdapter

FULL_A = "a" * 64
FULL_B = "b" * 64
FULL_C = "c" * 64


class _ReplayDbos:
    """Checkpoint emulator for the concrete boundary unit seam."""

    workflow_id: ClassVar[str | None] = "proposal-workflow"
    step_id: ClassVar[int | None] = None
    retries_allowed: ClassVar[list[bool]] = []
    checkpoints: ClassVar[dict[str, Any]] = {}
    sleeps: ClassVar[list[float]] = []
    events: ClassVar[list[str]] = []
    child_results: ClassVar[dict[str, Any]] = {}
    next_workflow_id: ClassVar[str | None] = None
    before_checkpoint_publication: ClassVar[Callable[[], None] | None] = None

    @classmethod
    def step(cls, *, retries_allowed: bool):
        cls.retries_allowed.append(retries_allowed)

        def decorate(function):
            def checkpointed(*args):
                payload = [
                    item.model_dump(mode="json")
                    if hasattr(item, "model_dump")
                    else item
                    for item in args
                ]
                encoded = json.dumps(payload, sort_keys=True)
                key = f"{function.__name__}:{encoded}"
                if key not in cls.checkpoints:
                    cls.events.append(f"step:{function.__name__}")
                    prior_step_id = cls.step_id
                    cls.step_id = 0
                    try:
                        result = function(*args)
                    finally:
                        cls.step_id = prior_step_id
                    gate = cls.before_checkpoint_publication
                    cls.before_checkpoint_publication = None
                    if gate is not None:
                        gate()
                    cls.checkpoints[key] = result
                return cls.checkpoints[key]

            return checkpointed

        return decorate

    @classmethod
    def sleep(cls, seconds: float) -> None:
        cls.events.append(f"sleep:{seconds}")
        cls.sleeps.append(seconds)

    @classmethod
    def workflow(cls):
        def decorate(function):
            return function

        return decorate

    @classmethod
    def start_workflow(cls, function, *args):
        if cls.workflow_id is None or cls.step_id is not None:
            raise RuntimeError(
                "DBOS.start_workflow requires a workflow body context"
            )
        child_id = cls.next_workflow_id
        assert child_id is not None
        cls.next_workflow_id = None
        cls.events.append(f"start_workflow:{child_id}")
        if child_id not in cls.child_results:
            parent_id = cls.workflow_id
            parent_step_id = cls.step_id
            cls.workflow_id = child_id
            cls.step_id = None
            try:
                cls.child_results[child_id] = function(*args)
            finally:
                cls.workflow_id = parent_id
                cls.step_id = parent_step_id
        return _ReplayHandle(cls, child_id)

    @classmethod
    def get_child_result(cls, child_id: str):
        return cls.child_results[child_id]


def _reset_replay_dbos() -> None:
    _ReplayDbos.workflow_id = "proposal-workflow"
    _ReplayDbos.retries_allowed.clear()
    _ReplayDbos.checkpoints.clear()
    _ReplayDbos.sleeps.clear()
    _ReplayDbos.events.clear()
    _ReplayDbos.child_results.clear()
    _ReplayDbos.next_workflow_id = None
    _ReplayDbos.step_id = None
    _ReplayDbos.before_checkpoint_publication = None


class _ReplayHandle:
    def __init__(self, dbos_type, workflow_id: str) -> None:
        self._dbos_type = dbos_type
        self._workflow_id = workflow_id

    def get_result(self):
        return self._dbos_type.get_child_result(self._workflow_id)


def _load_boundary(dbos_type=_ReplayDbos):
    fake_dbos = types.ModuleType("dbos")
    fake_dbos.__dict__["DBOS"] = dbos_type

    class _SetWorkflowID:
        def __init__(self, workflow_id: str) -> None:
            self._workflow_id = workflow_id

        def __enter__(self):
            dbos_type.next_workflow_id = self._workflow_id
            return self

        def __exit__(self, *_args):
            return False

    fake_dbos.__dict__["SetWorkflowID"] = _SetWorkflowID
    prior = sys.modules.get("dbos")
    sys.modules["dbos"] = fake_dbos
    try:
        path = Path("src/whetstone/coordination/proposal_provider.py")
        spec = importlib.util.spec_from_file_location(
            "_proposal_provider_durability_test",
            path,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if prior is None:
            del sys.modules["dbos"]
        else:
            sys.modules["dbos"] = prior


def _executor(module, transport):
    registry_key = module.register_proposal_transport(transport)
    return module.DbosProposalExecutor(transport_registry_key=registry_key)


def _provider_transport(
    *outcomes,
    max_attempts: int = 1,
    prompt_adapter: PlainPromptAdapter | None = None,
    sleep: provider_support.SleepRecorder | None = None,
):
    provider_config = provider_support.openrouter_chat_config(
        model="proposal-model"
    )
    transport_policy = provider_support.build_transport_policy()
    recording = provider_support.RecordingTransport(
        request=provider_support.build_request(),
        transport_policy=transport_policy,
        outcomes=list(outcomes),
    )
    proposer = ProviderProposerTransport(
        resolve_provider_call_config=lambda _ref: provider_config,
        transport=recording,
        execution_policy=provider_support.build_execution_policy(
            max_attempts=max_attempts,
            transport_policy=transport_policy,
        ),
        prompt_adapter=prompt_adapter,
        clock=provider_support.FakeClock(),
        sleep=sleep if sleep is not None else provider_support.SleepRecorder(),
    )
    return proposer, _config(provider_config), recording


def _config(provider_config=None) -> ProposerConfig:
    exact = provider_config or provider_support.openrouter_chat_config(
        model="proposal-model"
    )
    return ProposerConfig(
        provider_call_config=IdentityRef(
            record_ref=typed_ref_for_record(
                "dr_providers.provider_call_config",
                exact.model_dump(mode="json"),
            ),
            identity_hash=exact.identity_hash,
        )
    )


def _request(*, request_ordinal: int = 0) -> ProposalRequest:
    return ProposalRequest(
        proposal_mode="instruction_proposal",
        request_ordinal=request_ordinal,
        base_candidate=candidate_reference(candidate("provider-base")),
        context={"proposal_prompt": "Improve this prompt."},
    )


def test_persisted_durability_contract_literals_are_pinned() -> None:
    module = _load_boundary()

    assert module.PROPOSAL_DBOS_POLICY_SCHEMA == (
        "whetstone.proposal_dbos_policy"
    )
    assert module.PROPOSAL_DBOS_POLICY_VERSION == 1
    assert module.PROPOSAL_DBOS_WORKFLOW_SCHEMA == (
        "whetstone.proposal_dbos_workflow"
    )
    assert module.PROPOSAL_DBOS_WORKFLOW_VERSION == 1
    assert module.PROPOSAL_DURABILITY_MODE == "at_least_once"

    assert module._proposal_policy_identity_payload(FULL_C) == {
        "automatic_dbos_retries": False,
        "durability_mode": "at_least_once",
        "logical_call_boundary": "one_retry_disabled_dbos_step",
        "provider_retry_owner": "provider_execution_policy",
        "transport_durability_identity_hash": FULL_C,
    }
    policy_identity = module._proposal_policy_identity(FULL_C)
    assert policy_identity == (
        "10bf4652e413c4dc834601b673d220a305a7a9ade07794558b5eef8a6e9b0489"
    )

    stable_config = ProposerConfig(
        provider_call_config=IdentityRef(
            record_ref=typed_ref_for_record(
                "dr_providers.provider_call_config",
                {"fixture": "stable"},
            ),
            identity_hash=FULL_A,
        )
    )
    workflow_payload = module._proposal_workflow_identity_payload(
        registry_key=FULL_C,
        policy_identity_hash=policy_identity,
        config=stable_config,
        request=_request(),
        count=1,
    )
    assert workflow_payload == {
        "count": 1,
        "policy_identity_hash": policy_identity,
        "proposal_request_identity_hash": _request().identity_hash(),
        "proposer_config_identity_hash": stable_config.identity_hash(),
        "transport_durability_identity_hash": FULL_C,
    }
    assert (
        module._proposal_workflow_identity(
            registry_key=FULL_C,
            policy_identity_hash=policy_identity,
            config=stable_config,
            request=_request(),
            count=1,
        )
        == "d28a3ad93d790e4bcee510ad3620ca8aa5ec5b5681aca2bdd4d3e68afff3ce72"
    )


def test_durability_contract_is_durable_workflow_recovery() -> None:
    module = _load_boundary()
    transport, _config_value, _recording = _provider_transport(
        provider_support.response_outcome(text="contract"),
    )
    executor = _executor(module, transport)

    contract = executor.durability_contract
    assert contract.recovery_policy is ReplayPolicy.DURABLE_WORKFLOW
    assert executor.recovery_policy is ReplayPolicy.DURABLE_WORKFLOW
    assert contract.policy_identity_hash == module._proposal_policy_identity(
        transport.durability_identity_hash
    )
    assert executor.policy_identity_hash == contract.policy_identity_hash


def test_completed_step_replays_without_a_second_transport_call() -> None:
    _reset_replay_dbos()
    module = _load_boundary()
    request = _request()
    transport, config, recording = _provider_transport(
        provider_support.response_outcome(text="improved"),
    )
    executor = _executor(module, transport)

    first = executor.execute(
        config=config,
        request=request,
        transport=transport,
        count=1,
    )
    replay = executor.execute(
        config=config,
        request=request,
        transport=transport,
        count=1,
    )

    assert first[0].template == "improved"
    assert replay == first
    assert len(recording.served) == 1
    assert _ReplayDbos.retries_allowed == [False]


def test_one_step_wraps_the_whole_logical_call_including_retries() -> None:
    """Provider retries run inside the step; DBOS never sleeps between them."""

    _reset_replay_dbos()
    module = _load_boundary()
    provider_sleep = provider_support.SleepRecorder()
    transport, config, recording = _provider_transport(
        provider_support.failure_outcome(failure_class=FailureClass.TRANSIENT),
        provider_support.response_outcome(text="after retry"),
        max_attempts=2,
        sleep=provider_sleep,
    )
    executor = _executor(module, transport)

    (draft,) = executor.execute(
        config=config,
        request=_request(),
        transport=transport,
        count=1,
    )

    assert draft.template == "after retry"
    assert len(recording.served) == 2
    assert provider_sleep.delays == [1.0]
    assert _ReplayDbos.retries_allowed == [False]
    assert _ReplayDbos.sleeps == []
    assert _ReplayDbos.events == [
        f"start_workflow:{next(iter(_ReplayDbos.child_results))}",
        "step:_logical_proposal_step",
    ]

    replay = executor.execute(
        config=config,
        request=_request(),
        transport=transport,
        count=1,
    )
    assert replay == (draft,)
    assert len(recording.served) == 2
    assert provider_sleep.delays == [1.0]
    assert _ReplayDbos.retries_allowed == [False]
    assert _ReplayDbos.sleeps == []


def test_accept_before_checkpoint_replays_at_least_once_effect() -> None:
    """A lost step checkpoint re-executes the accepted provider effect."""

    class _ProcessInterrupted(RuntimeError):
        pass

    def interrupt_before_checkpoint_publication() -> None:
        raise _ProcessInterrupted

    _reset_replay_dbos()
    module = _load_boundary()
    transport, config, recording = _provider_transport(
        provider_support.response_outcome(text="stable result"),
    )
    executor = _executor(module, transport)
    request = _request()
    _ReplayDbos.before_checkpoint_publication = (
        interrupt_before_checkpoint_publication
    )

    with pytest.raises(_ProcessInterrupted):
        executor.execute(
            config=config,
            request=request,
            transport=transport,
            count=1,
        )

    assert len(recording.served) == 1
    assert _ReplayDbos.checkpoints == {}
    assert _ReplayDbos.child_results == {}

    recovered = executor.execute(
        config=config,
        request=request,
        transport=transport,
        count=1,
    )

    assert recovered[0].template == "stable result"
    assert len(recording.served) == 2
    assert len(_ReplayDbos.checkpoints) == 1
    assert len(_ReplayDbos.child_results) == 1

    replay = executor.execute(
        config=config,
        request=request,
        transport=transport,
        count=1,
    )

    assert replay == recovered
    assert len(recording.served) == 2
    assert len(_ReplayDbos.checkpoints) == 1
    started_children = [
        event
        for event in _ReplayDbos.events
        if event.startswith("start_workflow:")
    ]
    assert len(started_children) == 3
    assert len(set(started_children)) == 1


def test_same_effect_has_same_child_identity_across_ambient_workflows() -> (
    None
):
    _reset_replay_dbos()
    module = _load_boundary()
    transport, config, recording = _provider_transport(
        provider_support.response_outcome(text="stable result"),
    )
    executor = _executor(module, transport)
    request = _request()

    first = executor.execute(
        config=config,
        request=request,
        transport=transport,
        count=1,
    )
    _ReplayDbos.workflow_id = "different-ambient-workflow"
    replay = executor.execute(
        config=config,
        request=request,
        transport=transport,
        count=1,
    )

    assert replay == first
    assert len(_ReplayDbos.child_results) == 1
    assert len(recording.served) == 1


def test_distinct_requests_get_distinct_child_workflows_and_effects() -> None:
    _reset_replay_dbos()
    module = _load_boundary()
    transport, config, recording = _provider_transport(
        provider_support.response_outcome(text="round-0"),
        provider_support.response_outcome(text="round-1"),
    )
    executor = _executor(module, transport)

    first = executor.execute(
        config=config,
        request=_request(request_ordinal=0),
        transport=transport,
        count=1,
    )
    second = executor.execute(
        config=config,
        request=_request(request_ordinal=1),
        transport=transport,
        count=1,
    )
    replay = executor.execute(
        config=config,
        request=_request(request_ordinal=0),
        transport=transport,
        count=1,
    )

    assert first[0].template == "round-0"
    assert second[0].template == "round-1"
    assert replay == first
    assert len(_ReplayDbos.child_results) == 2
    assert len(recording.served) == 2


def test_executor_rejects_dbos_step_context_before_child_or_provider() -> None:
    _reset_replay_dbos()
    module = _load_boundary()
    transport, config, recording = _provider_transport(
        provider_support.response_outcome(text="child result"),
    )
    executor = _executor(module, transport)
    parent_step = _ReplayDbos.step(retries_allowed=False)(
        lambda: executor.execute(
            config=config,
            request=_request(),
            transport=transport,
            count=1,
        )
    )

    with pytest.raises(
        module.ProposalProviderError,
        match="cannot start a child workflow from a DBOS step",
    ):
        parent_step()

    assert _ReplayDbos.child_results == {}
    assert recording.served == []
    assert not any(
        event.startswith("start_workflow:") for event in _ReplayDbos.events
    )


def test_executor_requires_workflow_body_before_child_or_provider() -> None:
    _reset_replay_dbos()
    module = _load_boundary()
    transport, config, recording = _provider_transport(
        provider_support.response_outcome(text="must not run"),
    )
    executor = _executor(module, transport)
    _ReplayDbos.workflow_id = None

    with pytest.raises(
        module.ProposalProviderError,
        match="requires a DBOS workflow body context",
    ):
        executor.execute(
            config=config,
            request=_request(),
            transport=transport,
            count=1,
        )

    assert _ReplayDbos.child_results == {}
    assert _ReplayDbos.events == []
    assert recording.served == []


def test_fresh_registry_requires_startup_transport_configuration() -> None:
    _reset_replay_dbos()
    configured_module = _load_boundary()
    transport, config, recording = _provider_transport(
        provider_support.response_outcome(text="must not run"),
    )
    registry_key = configured_module.register_proposal_transport(transport)
    module = _load_boundary()
    executor = module.DbosProposalExecutor(transport_registry_key=registry_key)

    with pytest.raises(module.ProposalProviderError, match="not registered"):
        executor.execute(
            config=config,
            request=_request(),
            transport=transport,
            count=1,
        )

    assert recording.served == []
    assert _ReplayDbos.events == []


def test_arbitrary_registry_key_cannot_bind_a_transport() -> None:
    _reset_replay_dbos()
    module = _load_boundary()
    transport, config, recording = _provider_transport(
        provider_support.response_outcome(text="must not run"),
    )
    executor = module.DbosProposalExecutor(transport_registry_key=FULL_A)

    with pytest.raises(module.ProposalProviderError, match="not registered"):
        executor.execute(
            config=config,
            request=_request(),
            transport=transport,
            count=1,
        )

    assert recording.served == []


def test_registration_rejects_a_structural_transport_duck_type() -> None:
    _reset_replay_dbos()
    module = _load_boundary()

    class _StructuralProposerTransport:
        execution_policy_hash = FULL_A
        prompt_adapter_identity_hash = FULL_B
        durability_identity_hash = FULL_C

        def draft(self, _config, _request, _count):
            raise AssertionError("structural transport must never be invoked")

    with pytest.raises(
        module.ProposalProviderError,
        match="requires ProviderProposerTransport",
    ):
        module.register_proposal_transport(_StructuralProposerTransport())


def test_unregistered_transport_instance_cannot_ride_a_bound_key() -> None:
    _reset_replay_dbos()
    module = _load_boundary()
    registered, config, registered_recording = _provider_transport(
        provider_support.response_outcome(text="registered result"),
    )
    impostor, _config_value, impostor_recording = _provider_transport(
        provider_support.response_outcome(text="must not run"),
    )
    executor = _executor(module, registered)

    with pytest.raises(
        module.ProposalProviderError,
        match="differs from its registered identity",
    ):
        executor.execute(
            config=config,
            request=_request(),
            transport=impostor,
            count=1,
        )

    assert registered_recording.served == []
    assert impostor_recording.served == []


def test_identity_keyed_registry_keeps_sequential_executions_separate() -> (
    None
):
    _reset_replay_dbos()
    module = _load_boundary()
    first, config, first_recording = _provider_transport(
        provider_support.response_outcome(text="first transport"),
    )
    second, _config_value, second_recording = _provider_transport(
        provider_support.response_outcome(text="second transport"),
        prompt_adapter=PlainPromptAdapter(output_field="changed"),
    )
    first_executor = _executor(module, first)
    second_executor = _executor(module, second)
    request = _request()

    first_result = first_executor.execute(
        config=config,
        request=request,
        transport=first,
        count=1,
    )
    second_result = second_executor.execute(
        config=config,
        request=request,
        transport=second,
        count=1,
    )
    first_replay = first_executor.execute(
        config=config,
        request=request,
        transport=first,
        count=1,
    )

    assert first_result[0].template == "first transport"
    assert second_result[0].template == "second transport"
    assert first_replay == first_result
    assert len(first_recording.served) == 1
    assert len(second_recording.served) == 1


def _assert_concurrent_conflicting_transport_binding_is_atomic() -> None:
    module = _load_boundary()
    registry = module._ProposalTransportRegistry()
    first, _config_value, _first_recording = _provider_transport(
        provider_support.response_outcome(text="first"),
    )
    twin, _config_twin, _twin_recording = _provider_transport(
        provider_support.response_outcome(text="twin"),
    )
    start = Barrier(3)

    class _ObservedLock:
        def __init__(self) -> None:
            self._lock = Lock()
            self._attempt_guard = Lock()
            self._attempts = 0
            self._owner: int | None = None
            self.contended = Event()

        def __enter__(self):
            with self._attempt_guard:
                self._attempts += 1
                if self._attempts == 2:
                    self.contended.set()
            self._lock.acquire()
            self._owner = get_ident()
            return self

        def __exit__(self, *_args) -> None:
            assert self._owner == get_ident()
            self._owner = None
            self._lock.release()

        def owned_by_current_thread(self) -> bool:
            return self._owner == get_ident()

    observed_lock = _ObservedLock()

    class _LockCheckedTransports:
        def __init__(self) -> None:
            self._values: dict[str, ProviderProposerTransport] = {}

        def get(self, registry_key: str):
            assert observed_lock.owned_by_current_thread()
            assert observed_lock.contended.wait(timeout=5)
            return self._values.get(registry_key)

        def __getitem__(self, registry_key: str):
            assert observed_lock.owned_by_current_thread()
            return self._values[registry_key]

        def __setitem__(
            self,
            registry_key: str,
            transport: ProviderProposerTransport,
        ) -> None:
            assert observed_lock.owned_by_current_thread()
            self._values[registry_key] = transport

    registry._lock = observed_lock
    registry._transports = _LockCheckedTransports()

    def bind(transport: ProviderProposerTransport):
        start.wait(timeout=5)
        try:
            return registry.bind(transport)
        except module.ProposalProviderError as error:
            return error

    assert first is not twin
    assert first.durability_identity_hash == twin.durability_identity_hash
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(bind, transport) for transport in (first, twin)]
        start.wait(timeout=5)
        outcomes = [future.result(timeout=5) for future in futures]

    successes = [outcome for outcome in outcomes if isinstance(outcome, str)]
    conflicts = [
        outcome
        for outcome in outcomes
        if isinstance(outcome, module.ProposalProviderError)
    ]
    assert successes == [first.durability_identity_hash]
    assert len(conflicts) == 1
    assert str(conflicts[0]) == "proposal transport key is already bound"
    winner = first if isinstance(outcomes[0], str) else twin
    assert registry.resolve(first.durability_identity_hash) is winner


def test_concurrent_conflicting_transport_binding_is_atomic() -> None:
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_assert_concurrent_conflicting_transport_binding_is_atomic,
    )
    process.start()
    try:
        process.join(timeout=10)
        assert not process.is_alive(), "registry contention did not terminate"
        assert process.exitcode == 0
    finally:
        terminate_processes((process,), timeout=5)


def test_conflicting_transport_under_one_key_is_rejected() -> None:
    _reset_replay_dbos()
    module = _load_boundary()
    first, _config_value, _first_recording = _provider_transport(
        provider_support.response_outcome(text="first"),
    )
    twin, _config_twin, _twin_recording = _provider_transport(
        provider_support.response_outcome(text="twin"),
    )
    module.register_proposal_transport(first)

    assert first.durability_identity_hash == twin.durability_identity_hash
    with pytest.raises(module.ProposalProviderError, match="already bound"):
        module.register_proposal_transport(twin)


@pytest.mark.skipif(
    "WHETSTONE_TEST_POSTGRES_DSN" not in os.environ,
    reason="WHETSTONE_TEST_POSTGRES_DSN is required for real DBOS replay",
)
def test_real_dbos_workflow_replays_completed_proposal() -> None:
    """A repeated real workflow ID returns its checkpointed first result."""

    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from whetstone.coordination.proposal_provider import (
        DbosProposalExecutor,
        register_proposal_transport,
    )

    suffix = uuid4().hex[:10]
    database_url = os.environ["WHETSTONE_TEST_POSTGRES_DSN"]
    dbos_config: DBOSConfig = {
        "name": f"whetstone-proposal-{suffix}",
        "system_database_url": database_url,
        "application_database_url": database_url,
        "application_version": f"proposal-{suffix}",
        "run_admin_server": False,
        "use_listen_notify": False,
    }
    DBOS(config=dbos_config)
    transport, config, recording = _provider_transport(
        provider_support.response_outcome(text="durable improvement"),
    )
    registry_key = register_proposal_transport(transport)
    executor = DbosProposalExecutor(transport_registry_key=registry_key)
    request = _request()

    @DBOS.workflow()
    def proposal_workflow() -> str:
        return executor.execute(
            config=config,
            request=request,
            transport=transport,
            count=1,
        )[0].template

    try:
        DBOS.launch()
        workflow_id = f"proposal-provider-{suffix}"
        with SetWorkflowID(workflow_id):
            first = proposal_workflow()
        with SetWorkflowID(workflow_id):
            replay = proposal_workflow()
        assert first == replay == "durable improvement"
        assert len(recording.served) == 1
    finally:
        DBOS.destroy()
