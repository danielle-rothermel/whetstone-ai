"""Durability contract for the MIPROv2 proposal provider boundary."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid4

import pytest
from dr_providers import FailureClass
from dr_store import ObjectStore, SqliteBackend

from tests.provider import support as provider_support
from whetstone.optimization.miprov2 import Miprov2ProposalEffectStore
from whetstone.optimization.proposer import (
    FakeProposerTransport,
    ProposalRequest,
    ProposerConfig,
    ProviderProposerTransport,
)

FULL_A = "a" * 64
FULL_B = "b" * 64
FULL_C = "c" * 64

if TYPE_CHECKING:
    from sqlalchemy import Engine


class _ReplayDbos:
    """Checkpoint emulator for the concrete boundary unit seam."""

    workflow_id = "miprov2-workflow"
    retries_allowed: ClassVar[list[bool]] = []
    checkpoints: ClassVar[dict[str, Any]] = {}
    sleeps: ClassVar[list[float]] = []
    events: ClassVar[list[str]] = []
    crash_after_step_once: ClassVar[str | None] = None
    child_results: ClassVar[dict[str, Any]] = {}
    next_workflow_id: ClassVar[str | None] = None

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
                    result = function(*args)
                    if cls.crash_after_step_once == function.__name__:
                        cls.crash_after_step_once = None
                        raise RuntimeError("injected crash before checkpoint")
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
        child_id = cls.next_workflow_id
        assert child_id is not None
        cls.next_workflow_id = None
        if child_id not in cls.child_results:
            parent_id = cls.workflow_id
            cls.workflow_id = child_id
            try:
                cls.child_results[child_id] = function(*args)
            finally:
                cls.workflow_id = parent_id
        return _ReplayHandle(cls, child_id)

    @classmethod
    def get_child_result(cls, child_id: str):
        return cls.child_results[child_id]


def _reset_replay_dbos() -> None:
    _ReplayDbos.retries_allowed.clear()
    _ReplayDbos.checkpoints.clear()
    _ReplayDbos.sleeps.clear()
    _ReplayDbos.events.clear()
    _ReplayDbos.crash_after_step_once = None
    _ReplayDbos.child_results.clear()
    _ReplayDbos.next_workflow_id = None


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
        path = Path("src/whetstone/orchestration/miprov2_provider.py")
        spec = importlib.util.spec_from_file_location(
            "_miprov2_provider_durability_test",
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


def _executor(
    module,
    transport,
    *,
    durability_mode="at_least_once",
    registry_key: str = FULL_C,
):
    module.register_miprov2_proposal_transport(registry_key, transport)
    return module.DbosMiprov2ProposalEffectExecutor(
        transport_registry_key=registry_key,
        durability_mode=durability_mode,
    )


class _OrderedReplayDbos:
    """Index-addressed DBOS emulator that detects operation-sequence drift."""

    workflow_id = "miprov2-ordered-workflow"
    cursor = 0
    checkpoints: ClassVar[list[tuple[str, Any]]] = []
    retries_allowed: ClassVar[list[bool]] = []
    child_results: ClassVar[dict[str, Any]] = {}
    child_checkpoints: ClassVar[dict[tuple[str, int], tuple[str, Any]]] = {}
    child_cursors: ClassVar[dict[str, int]] = {}
    current_child_id: ClassVar[str | None] = None
    next_workflow_id: ClassVar[str | None] = None

    @classmethod
    def reset(cls) -> None:
        cls.cursor = 0
        cls.checkpoints.clear()
        cls.retries_allowed.clear()
        cls.child_results.clear()
        cls.child_checkpoints.clear()
        cls.child_cursors.clear()
        cls.current_child_id = None
        cls.next_workflow_id = None

    @classmethod
    def begin_replay(cls) -> None:
        cls.cursor = 0

    @classmethod
    def workflow(cls):
        def decorate(function):
            return function

        return decorate

    @classmethod
    def _operation(cls, fingerprint: str, execute):
        index = cls.cursor
        cls.cursor += 1
        if index < len(cls.checkpoints):
            saved_fingerprint, saved_result = cls.checkpoints[index]
            if saved_fingerprint != fingerprint:
                raise AssertionError(
                    "DBOS operation sequence shifted: "
                    f"expected {saved_fingerprint}, got {fingerprint}"
                )
            return saved_result
        result = execute()
        cls.checkpoints.append((fingerprint, result))
        return result

    @classmethod
    def start_workflow(cls, function, *args):
        child_id = cls.next_workflow_id
        assert child_id is not None
        cls.next_workflow_id = None

        def start_child():
            if child_id not in cls.child_results:
                parent_id = cls.workflow_id
                cls.workflow_id = child_id
                cls.current_child_id = child_id
                cls.child_cursors[child_id] = 0
                try:
                    cls.child_results[child_id] = function(*args)
                finally:
                    cls.workflow_id = parent_id
                    cls.current_child_id = None
            return child_id

        recorded_child_id = cls._operation(
            f"start_workflow:{function.__name__}:{child_id}",
            start_child,
        )
        return _ReplayHandle(cls, recorded_child_id)

    @classmethod
    def get_child_result(cls, child_id: str):
        return cls._operation(
            f"get_result:{child_id}",
            lambda: cls.child_results[child_id],
        )

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
                fingerprint = (
                    f"{function.__name__}:"
                    f"{json.dumps(payload, sort_keys=True)}"
                )
                child_id = cls.current_child_id
                if child_id is not None:
                    index = cls.child_cursors[child_id]
                    cls.child_cursors[child_id] = index + 1
                    key = (child_id, index)
                    saved = cls.child_checkpoints.get(key)
                    if saved is not None:
                        saved_fingerprint, saved_result = saved
                        if saved_fingerprint != fingerprint:
                            raise AssertionError(
                                "child DBOS operation sequence shifted: "
                                f"expected {saved_fingerprint}, "
                                f"got {fingerprint}"
                            )
                        return saved_result
                    result = function(*args)
                    cls.child_checkpoints[key] = (fingerprint, result)
                    return result
                return cls._operation(fingerprint, lambda: function(*args))

            return checkpointed

        return decorate

    @staticmethod
    def sleep(_seconds: float) -> None:
        return None


def test_completed_dbos_proposal_step_replays_without_second_transport_call():
    _reset_replay_dbos()
    module = _load_boundary()
    config = ProposerConfig(
        provider_call_config_ref="provider://proposal",
        provider_call_config_hash=FULL_A,
    )
    request = ProposalRequest(
        proposal_mode="instruction_proposal",
        request_ordinal=0,
        base_ref="base",
        context={"proposal_prompt": "Improve this prompt."},
    )
    transport = FakeProposerTransport(
        {("instruction_proposal", 0): ("improved",)},
        execution_policy_hash=FULL_A,
        prompt_adapter_identity_hash=FULL_B,
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

    assert replay == first
    assert len(transport.calls) == 1
    assert _ReplayDbos.retries_allowed == [False, False]


def test_cached_application_result_still_consumes_dbos_operation_on_recovery(
    tmp_path,
) -> None:
    """Crash after app-result persistence cannot shift the next DBOS index."""

    _OrderedReplayDbos.reset()
    module = _load_boundary(_OrderedReplayDbos)
    effect_store = Miprov2ProposalEffectStore(
        ObjectStore(SqliteBackend(tmp_path / "ordered-replay.sqlite"))
    )
    config = ProposerConfig(
        provider_call_config_ref="provider://proposal",
        provider_call_config_hash=FULL_A,
    )
    request = _request()
    transport = FakeProposerTransport(
        {("instruction_proposal", 0): ("checkpointed",)},
        execution_policy_hash=FULL_A,
        prompt_adapter_identity_hash=FULL_B,
    )
    executor = _executor(module, transport)

    first = effect_store.execute(
        native_request_identity_hash=FULL_B,
        config=config,
        request=request,
        transport=transport,
        executor=executor,
    )
    # The application result is now persisted. Simulate an outer-workflow
    # crash before its next durable operation, then restart from DBOS index 0.
    _OrderedReplayDbos.begin_replay()
    replay = effect_store.execute(
        native_request_identity_hash=FULL_B,
        config=config,
        request=request,
        transport=transport,
        executor=executor,
    )

    @_OrderedReplayDbos.step(retries_allowed=False)
    def later_operation(value: str) -> str:
        return f"later:{value}"

    assert later_operation("safe") == "later:safe"
    assert replay == first
    assert len(transport.calls) == 1
    assert _OrderedReplayDbos.cursor == 2
    assert len(_OrderedReplayDbos.checkpoints) == 2


def _provider_transport(*outcomes, max_attempts: int = 1):
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
    )
    config = ProposerConfig(
        provider_call_config_ref="provider://proposal",
        provider_call_config_hash=provider_config.identity_hash,
    )
    return proposer, config, recording


def _request() -> ProposalRequest:
    return ProposalRequest(
        proposal_mode="instruction_proposal",
        request_ordinal=0,
        base_ref="base",
        run_id="miprov2-provider-durability",
        step_index=0,
        context={"proposal_prompt": "Improve this prompt."},
    )


def test_provider_retries_checkpoint_each_attempt_and_sleep_durably():
    _reset_replay_dbos()
    module = _load_boundary()
    transient = provider_support.failure_outcome(
        failure_class=FailureClass.TRANSIENT
    )
    proposer, config, recording = _provider_transport(
        transient,
        provider_support.response_outcome(text="after retry"),
        max_attempts=2,
    )

    executor = _executor(module, proposer)
    (draft,) = executor.execute(
        config=config,
        request=_request(),
        transport=proposer,
        count=1,
    )

    assert draft.template == "after retry"
    assert len(recording.served) == 2
    assert _ReplayDbos.retries_allowed == [False, False]
    assert _ReplayDbos.sleeps == [1.0]
    assert _ReplayDbos.events == [
        "step:_proposal_provider_attempt_step",
        "sleep:1.0",
        "step:_proposal_provider_attempt_step",
    ]

    replay = executor.execute(
        config=config,
        request=_request(),
        transport=proposer,
        count=1,
    )
    assert replay == (draft,)
    assert len(recording.served) == 2


def test_provider_idempotent_mode_rejects_unsupported_transport_preflight():
    _reset_replay_dbos()
    module = _load_boundary()
    proposer, config, recording = _provider_transport(
        provider_support.response_outcome(text="must not run"),
    )

    with pytest.raises(
        module.Miprov2ProposalDurabilityError,
        match="accepts stable idempotency evidence",
    ):
        _executor(
            module,
            proposer,
            durability_mode="provider_idempotent",
        ).execute(
            config=config,
            request=_request(),
            transport=proposer,
            count=1,
        )

    assert recording.served == []
    assert _ReplayDbos.retries_allowed == [False, False]


def test_fresh_registry_requires_startup_transport_configuration():
    _reset_replay_dbos()
    module = _load_boundary()
    proposer, config, recording = _provider_transport(
        provider_support.response_outcome(text="must not run"),
    )
    executor = module.DbosMiprov2ProposalEffectExecutor(
        transport_registry_key=FULL_C
    )

    with pytest.raises(
        module.Miprov2ProposalDurabilityError,
        match="not registered",
    ):
        executor.execute(
            config=config,
            request=_request(),
            transport=proposer,
            count=1,
        )

    assert recording.served == []
    assert _ReplayDbos.events == []


def test_identity_keyed_registry_keeps_concurrent_executors_separate():
    _reset_replay_dbos()
    module = _load_boundary()
    first = FakeProposerTransport(
        {("instruction_proposal", 0): ("first transport",)},
        execution_policy_hash=FULL_A,
        prompt_adapter_identity_hash=FULL_B,
    )
    second = FakeProposerTransport(
        {("instruction_proposal", 0): ("second transport",)},
        execution_policy_hash=FULL_A,
        prompt_adapter_identity_hash=FULL_B,
    )
    first_executor = _executor(module, first, registry_key=FULL_C)
    second_executor = _executor(module, second, registry_key="d" * 64)
    config = ProposerConfig(
        provider_call_config_ref="provider://proposal",
        provider_call_config_hash=FULL_A,
    )
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
    assert len(first.calls) == 1
    assert len(second.calls) == 1


class _IdempotentRecordingTransport:
    """Provider-idempotent physical seam with separately counted wire calls."""

    idempotency_policy_identity_hash = "d" * 64

    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self._by_key: dict[str, Any] = {}
        self.invocations: list[str] = []

    def __call__(self, request):
        return self._delegate(request)

    def invoke_idempotent(self, request, *, idempotency_key: str):
        self.invocations.append(idempotency_key)
        if idempotency_key not in self._by_key:
            self._by_key[idempotency_key] = self._delegate(request)
        return self._by_key[idempotency_key]


def test_provider_idempotency_closes_crash_after_wire_before_checkpoint():
    _reset_replay_dbos()
    module = _load_boundary()
    provider_config = provider_support.openrouter_chat_config(
        model="proposal-model"
    )
    transport_policy = provider_support.build_transport_policy()
    recording = provider_support.RecordingTransport(
        request=provider_support.build_request(),
        transport_policy=transport_policy,
        outcomes=[
            provider_support.response_outcome(text="crash-safe response")
        ],
    )
    physical = _IdempotentRecordingTransport(recording)
    proposer = ProviderProposerTransport(
        resolve_provider_call_config=lambda _ref: provider_config,
        transport=physical,
        execution_policy=provider_support.build_execution_policy(
            max_attempts=1,
            transport_policy=transport_policy,
        ),
    )
    config = ProposerConfig(
        provider_call_config_ref="provider://proposal",
        provider_call_config_hash=provider_config.identity_hash,
    )
    executor = _executor(
        module,
        proposer,
        durability_mode="provider_idempotent",
    )
    _ReplayDbos.crash_after_step_once = "_proposal_provider_attempt_step"

    with pytest.raises(RuntimeError, match="injected crash"):
        executor.execute(
            config=config,
            request=_request(),
            transport=proposer,
            count=1,
        )

    (draft,) = executor.execute(
        config=config,
        request=_request(),
        transport=proposer,
        count=1,
    )

    assert draft.template == "crash-safe response"
    assert len(physical.invocations) == 2
    assert physical.invocations[0] == physical.invocations[1]
    assert len(recording.served) == 1


def test_at_least_once_mode_exposes_irreducible_precheckpoint_window():
    """A crash after provider acceptance can repeat a non-idempotent wire."""

    _reset_replay_dbos()
    module = _load_boundary()
    proposer, config, recording = _provider_transport(
        provider_support.response_outcome(text="uncheckpointed response"),
        provider_support.response_outcome(text="repeated wire response"),
    )
    executor = _executor(
        module,
        proposer,
        durability_mode="at_least_once",
    )
    _ReplayDbos.crash_after_step_once = "_proposal_provider_attempt_step"

    with pytest.raises(RuntimeError, match="injected crash"):
        executor.execute(
            config=config,
            request=_request(),
            transport=proposer,
            count=1,
        )
    (recovered,) = executor.execute(
        config=config,
        request=_request(),
        transport=proposer,
        count=1,
    )

    assert recovered.template == "repeated wire response"
    assert len(recording.served) == 2


def test_real_dbos_workflow_replays_completed_proposal(
    clean_pg: str,
    pg_engine: Engine,
) -> None:
    """A repeated real workflow ID returns its checkpointed first result."""

    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from tests.orchestration.platform_support import engine_dsn
    from whetstone.orchestration.miprov2_provider import (
        DbosMiprov2ProposalEffectExecutor,
        register_miprov2_proposal_transport,
    )

    suffix = uuid4().hex[:10]
    database_url = engine_dsn(pg_engine)
    dbos_config: DBOSConfig = {
        "name": f"whetstone-miprov2-{suffix}",
        "system_database_url": database_url,
        "application_database_url": database_url,
        "application_version": f"miprov2-{suffix}",
        "run_admin_server": False,
        "use_listen_notify": False,
    }
    DBOS(config=dbos_config)
    transport = FakeProposerTransport(
        {("instruction_proposal", 0): ("durable improvement",)},
        execution_policy_hash=FULL_A,
        prompt_adapter_identity_hash=FULL_B,
    )
    register_miprov2_proposal_transport(FULL_C, transport)
    executor = DbosMiprov2ProposalEffectExecutor(transport_registry_key=FULL_C)
    config = ProposerConfig(
        provider_call_config_ref="provider://proposal",
        provider_call_config_hash=FULL_A,
    )
    request = ProposalRequest(
        proposal_mode="instruction_proposal",
        request_ordinal=0,
        base_ref="base",
        context={"proposal_prompt": "Improve this prompt."},
    )

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
        workflow_id = f"miprov2-proposal-{suffix}"
        with SetWorkflowID(workflow_id):
            first = proposal_workflow()
        with SetWorkflowID(workflow_id):
            replay = proposal_workflow()
        assert first == replay == "durable improvement"
        assert len(transport.calls) == 1
    finally:
        DBOS.destroy(destroy_registry=True)
