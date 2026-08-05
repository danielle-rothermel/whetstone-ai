"""Crash-window tests for GEPA's stable child effect broker."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, ClassVar, Literal

import pytest
from dr_store import ObjectStore, SqliteBackend

from tests.coordination.test_proposal_provider import (
    _executor,
    _provider_transport,
)
from tests.coordination.test_proposal_provider import (
    _load_boundary as _load_provider_boundary,
)
from tests.optimization.gepa.test_effects import (
    _context,
    _evaluation_request,
    _evaluation_result,
    _prompt_services,
    _proposal_authority,
)
from tests.provider import support as provider_support
from whetstone.core.identity import typed_ref_for_record
from whetstone.optimization.gepa.authorities import (
    CanonicalGepaProposalAuthority,
)
from whetstone.optimization.gepa.contracts import (
    GepaCandidateComponent,
    GepaEffectRecorder,
    GepaEffectSlot,
    GepaEvaluationEffectRequest,
    GepaProposalEffectRequest,
    GepaProposalEffectResult,
)
from whetstone.optimization.gepa.prompts import GepaRenderedPrompt
from whetstone.optimization.proposal.proposer import ProposalRequest


class _ReplayDbos:
    events: ClassVar[list[str]] = []
    child_results: ClassVar[dict[str, object]] = {}
    workflow_id: str | None = None
    fail_if_invoked = False
    crash_after_child_once = False

    @classmethod
    def workflow(cls):
        def decorate(function):
            def wrapped(*args, **kwargs):
                cls.events.append(function.__name__)
                if cls.fail_if_invoked:
                    raise AssertionError(
                        "completed ObjectStore effect reached DBOS"
                    )
                workflow_id = cls.workflow_id
                assert workflow_id is not None
                if workflow_id in cls.child_results:
                    return cls.child_results[workflow_id]
                result = function(*args, **kwargs)
                cls.child_results[workflow_id] = result
                if cls.crash_after_child_once:
                    cls.crash_after_child_once = False
                    raise RuntimeError("injected crash after child completion")
                return result

            return wrapped

        return decorate


def _load_boundary(dbos_type=_ReplayDbos):
    fake_dbos = types.ModuleType("dbos")
    fake_dbos.__dict__["DBOS"] = dbos_type

    class _SetWorkflowID:
        def __init__(self, workflow_id: str) -> None:
            self.workflow_id = workflow_id

        def __enter__(self):
            self._prior = dbos_type.workflow_id
            dbos_type.workflow_id = self.workflow_id
            return self

        def __exit__(self, *_args):
            dbos_type.workflow_id = self._prior
            return False

    fake_dbos.__dict__["SetWorkflowID"] = _SetWorkflowID
    prior = sys.modules.get("dbos")
    sys.modules["dbos"] = fake_dbos
    try:
        path = Path("src/whetstone/optimization/gepa/effect_runtime.py")
        spec = importlib.util.spec_from_file_location(
            "_gepa_effect_durability_test",
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


def _proposal_request() -> GepaProposalEffectRequest:
    services = _prompt_services()
    return GepaProposalEffectRequest(
        slot=GepaEffectSlot(context=_context(), invocation_ordinal=0),
        candidate=(
            GepaCandidateComponent(name="alpha", text="alpha-0"),
            GepaCandidateComponent(name="beta", text="beta-0"),
        ),
        components_to_update=("alpha", "beta"),
        component_name="alpha",
        rendered_prompt=GepaRenderedPrompt(text="Improve alpha."),
        authority=_proposal_authority(services),
    )


def _proposal_result(
    request: GepaProposalEffectRequest,
) -> GepaProposalEffectResult:
    attempt_ref = typed_ref_for_record(
        "test.gepa.proposal_attempt",
        {"request": request.identity_hash()},
    )
    return GepaProposalEffectResult(
        request_identity_hash=request.identity_hash(),
        raw_response="```\nalpha-improved\n```",
        parsed_components=(
            GepaCandidateComponent(name="alpha", text="alpha-improved"),
        ),
        request_evidence={"prompt": request.rendered_prompt.text},
        response_evidence={"raw": "alpha-improved"},
        provider_attempt_refs=(attempt_ref,),
    )


@pytest.mark.parametrize("effect_kind", ["evaluate", "propose"])
def test_completed_object_store_result_does_not_skip_stable_child(
    tmp_path,
    effect_kind: str,
) -> None:
    _ReplayDbos.events.clear()
    _ReplayDbos.child_results.clear()
    _ReplayDbos.fail_if_invoked = False
    _ReplayDbos.crash_after_child_once = False
    module = _load_boundary()
    store = ObjectStore(SqliteBackend(tmp_path / f"{effect_kind}.sqlite"))
    recorder = GepaEffectRecorder(store)
    if effect_kind == "evaluate":
        request = _evaluation_request()
        result = _evaluation_result(request)
        recorder.record_request(request)
        recorder.record_evaluation_result(request, result)
        invoke = module.DbosGepaEffectBroker(store).evaluate
    else:
        request = _proposal_request()
        result = _proposal_result(request)
        recorder.record_request(request)
        recorder.record_proposal_result(request, result)
        invoke = module.DbosGepaEffectBroker(store).propose
    authority = _DurableAuthority(
        request.authority.authority_identity_hash,
        result,
    )
    if effect_kind == "evaluate":
        module.register_gepa_evaluation_authority(
            authority.runtime_identity_hash,
            authority,
        )
    else:
        module.register_gepa_proposal_authority(
            authority.runtime_identity_hash,
            authority,
        )

    assert invoke(request) == result
    assert authority.calls == 1
    assert _ReplayDbos.events == [
        f"_gepa_{'evaluation' if effect_kind == 'evaluate' else 'proposal'}"
        "_effect_workflow"
    ]


class _DurableAuthority:
    def __init__(self, identity_hash: str, result) -> None:
        self.runtime_identity_hash = identity_hash
        self.result = result
        self.calls = 0

    def evaluate(self, request):
        del request
        self.calls += 1
        return self.result

    def propose(self, request):
        del request
        self.calls += 1
        return self.result


class _ProviderBackedProposalAuthority:
    """Exercise the real application store and decorated provider executor."""

    def __init__(self, *, store, request, transport, executor, config) -> None:
        self.runtime_identity_hash = request.authority.authority_identity_hash
        self._store = store
        self._transport = transport
        self._executor = executor
        self._config = config

    def propose(self, request):
        generic = ProposalRequest(
            proposal_mode="gepa_reflection",
            request_ordinal=request.slot.invocation_ordinal,
            base_candidate=(
                CanonicalGepaProposalAuthority._reflection_base_candidate(
                    request,
                    "alpha-0",
                )
            ),
            context={"proposal_prompt": request.rendered_prompt.text},
        )
        drafts = self._executor.execute(
            config=self._config,
            request=generic,
            transport=self._transport,
            count=1,
        )
        assert len(drafts) == 1
        draft = drafts[0]
        attempt_ref = typed_ref_for_record(
            "test.gepa.proposal_attempt",
            {"request": request.identity_hash()},
        )
        return GepaProposalEffectResult(
            request_identity_hash=request.identity_hash(),
            raw_response=draft.template,
            parsed_components=(
                GepaCandidateComponent(
                    name=request.component_name,
                    text=draft.template,
                ),
            ),
            request_evidence=draft.request_evidence.to_json(),
            response_evidence=draft.response_evidence.to_json(),
            provider_attempt_refs=(attempt_ref,),
        )


class _NestedReplayDbos:
    """Emulate GEPA's semantic child workflow wrapping the executor's own.

    Both DBOS layers load against this one emulator: the outer GEPA effect
    workflow (entered through ``SetWorkflowID``) and, inside it, the
    executor's ``start_workflow`` child whose body is one retry-disabled
    step.  Every operation is appended to ``events`` so a replay can be
    checked for its exact operation sequence.
    """

    workflow_id: ClassVar[str | None] = None
    step_id: ClassVar[int | None] = None
    next_workflow_id: ClassVar[str | None] = None
    retries_allowed: ClassVar[list[bool]] = []
    checkpoints: ClassVar[dict[str, Any]] = {}
    child_results: ClassVar[dict[str, Any]] = {}
    events: ClassVar[list[str]] = []

    @classmethod
    def reset(cls) -> None:
        cls.workflow_id = None
        cls.step_id = None
        cls.next_workflow_id = None
        cls.retries_allowed.clear()
        cls.checkpoints.clear()
        cls.child_results.clear()
        cls.events.clear()

    @classmethod
    def workflow(cls):
        def decorate(function):
            def wrapped(*args, **kwargs):
                workflow_id = cls.workflow_id
                assert workflow_id is not None
                cls.events.append(f"workflow:{function.__name__}")
                if workflow_id in cls.child_results:
                    return cls.child_results[workflow_id]
                prior_step_id = cls.step_id
                cls.step_id = None
                try:
                    result = function(*args, **kwargs)
                finally:
                    cls.step_id = prior_step_id
                cls.child_results[workflow_id] = result
                return result

            return wrapped

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
        parent_id = cls.workflow_id
        cls.workflow_id = child_id
        try:
            result = function(*args)
        finally:
            cls.workflow_id = parent_id
        return _NestedReplayHandle(result)

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
                key = f"{function.__name__}:{json.dumps(payload)}"
                cls.events.append(f"step:{function.__name__}")
                if key in cls.checkpoints:
                    return cls.checkpoints[key]
                prior_step_id = cls.step_id
                cls.step_id = 0
                try:
                    result = function(*args)
                finally:
                    cls.step_id = prior_step_id
                cls.checkpoints[key] = result
                return result

            return checkpointed

        return decorate


class _NestedReplayHandle:
    def __init__(self, result) -> None:
        self._result = result

    def get_result(self):
        return self._result


@pytest.mark.parametrize("effect_kind", ["evaluate", "propose"])
def test_child_completion_then_outer_result_bind_replays_without_authority(
    tmp_path,
    effect_kind: str,
) -> None:
    _ReplayDbos.events.clear()
    _ReplayDbos.child_results.clear()
    _ReplayDbos.fail_if_invoked = False
    module = _load_boundary()
    database = tmp_path / f"{effect_kind}-crash.sqlite"
    store = ObjectStore(SqliteBackend(database))
    if effect_kind == "evaluate":
        request = _evaluation_request()
        result = _evaluation_result(request)
        authority = _DurableAuthority(
            request.authority.authority_identity_hash,
            result,
        )
        module.register_gepa_evaluation_authority(
            authority.runtime_identity_hash,
            authority,
        )
        invoke = module.DbosGepaEffectBroker(store).evaluate
    else:
        request = _proposal_request()
        result = _proposal_result(request)
        authority = _DurableAuthority(
            request.authority.authority_identity_hash,
            result,
        )
        module.register_gepa_proposal_authority(
            authority.runtime_identity_hash,
            authority,
        )
        invoke = module.DbosGepaEffectBroker(store).propose
    _ReplayDbos.crash_after_child_once = True

    with pytest.raises(RuntimeError, match="after child completion"):
        invoke(request)
    assert authority.calls == 1
    assert (
        GepaEffectRecorder(store)._store.resolve(
            GepaEffectRecorder._result_key(request)
        )
        is None
    )

    assert invoke(request) == result
    assert authority.calls == 1


def _operation_shape(events: list[str]) -> list[str]:
    """Erase the content-addressed child ID from a recorded event sequence."""

    return [
        "start_workflow" if event.startswith("start_workflow:") else event
        for event in events
    ]


#: The exact operation sequence one uncached GEPA reflection effect performs:
#: the semantic child workflow starts the executor's own child workflow, whose
#: body is the single retry-disabled whole-call step.
_FULL_PROPOSAL_SHAPE = [
    "workflow:_gepa_proposal_effect_workflow",
    "start_workflow",
    "workflow:_proposal_provider_workflow",
    "step:_logical_proposal_step",
]

#: Replay of a completed effect stops at the GEPA semantic child.
_REPLAYED_PROPOSAL_SHAPE = ["workflow:_gepa_proposal_effect_workflow"]


def _nested_proposal_broker(tmp_path, name: str):
    """Wire one GEPA broker over the executor on the nested DBOS emulator."""

    _NestedReplayDbos.reset()
    provider_module = _load_provider_boundary(_NestedReplayDbos)
    effect_module = _load_boundary(_NestedReplayDbos)
    request = _proposal_request()
    transport, proposer_config, recording = _provider_transport(
        provider_support.response_outcome(text="alpha-improved"),
    )
    executor = _executor(provider_module, transport)
    store = ObjectStore(SqliteBackend(tmp_path / f"{name}.sqlite"))
    authority = _ProviderBackedProposalAuthority(
        store=store,
        request=request,
        transport=transport,
        executor=executor,
        config=proposer_config,
    )
    effect_module.register_gepa_proposal_authority(
        authority.runtime_identity_hash,
        authority,
    )
    return effect_module.DbosGepaEffectBroker(store), request, recording


def test_proposal_replay_preserves_inner_operation_sequence_without_recall(
    tmp_path,
) -> None:
    """A crash before the outer bind never re-runs the paid proposal call."""

    broker, request, recording = _nested_proposal_broker(
        tmp_path,
        "proposal-inner-replay",
    )
    original_record = broker._recorder.record_proposal_result

    def crash_before_outer_bind(*_args, **_kwargs):
        raise RuntimeError("injected crash before GEPA child commit")

    broker._recorder.record_proposal_result = crash_before_outer_bind
    with pytest.raises(RuntimeError, match="before GEPA child commit"):
        broker.propose(request)

    assert len(recording.served) == 1
    assert _operation_shape(_NestedReplayDbos.events) == _FULL_PROPOSAL_SHAPE
    assert _NestedReplayDbos.retries_allowed == [False]

    broker._recorder.record_proposal_result = original_record
    _NestedReplayDbos.events.clear()
    replay = broker.propose(request)

    assert replay.parsed_components[0].text == "alpha-improved"
    assert len(recording.served) == 1
    assert (
        _operation_shape(_NestedReplayDbos.events) == _REPLAYED_PROPOSAL_SHAPE
    )


def test_parent_replay_always_consumes_stable_child_operation(
    tmp_path,
) -> None:
    """Replaying one effect reuses the child rather than re-proposing."""

    broker, request, recording = _nested_proposal_broker(
        tmp_path,
        "parent-operation-replay",
    )

    first = broker.propose(request)

    assert _operation_shape(_NestedReplayDbos.events) == _FULL_PROPOSAL_SHAPE
    _NestedReplayDbos.events.clear()

    replay = broker.propose(request)

    assert replay == first
    assert len(recording.served) == 1
    assert (
        _operation_shape(_NestedReplayDbos.events) == _REPLAYED_PROPOSAL_SHAPE
    )
    assert _NestedReplayDbos.retries_allowed == [False]


@pytest.mark.skipif(
    "WHETSTONE_TEST_POSTGRES_DSN" not in os.environ,
    reason="WHETSTONE_TEST_POSTGRES_DSN is required for real DBOS replay",
)
@pytest.mark.parametrize("effect_kind", ["evaluate", "propose"])
def test_real_dbos_child_checkpoint_survives_outer_bind_crash(
    tmp_path,
    effect_kind: Literal["evaluate", "propose"],
) -> None:
    from uuid import uuid4

    from dbos import DBOS, DBOSConfig

    from whetstone.optimization.gepa.effect_runtime import (
        DbosGepaEffectBroker,
        register_gepa_evaluation_authority,
        register_gepa_proposal_authority,
    )

    suffix = uuid4().hex[:10]
    database_url = os.environ["WHETSTONE_TEST_POSTGRES_DSN"]
    config: DBOSConfig = {
        "name": f"gepa-effect-{suffix}",
        "system_database_url": database_url,
        "application_database_url": database_url,
        "application_version": f"gepa-effect-{suffix}",
        "run_admin_server": False,
        "use_listen_notify": False,
    }
    DBOS(config=config)
    if effect_kind == "evaluate":
        request = _evaluation_request()
    else:
        request = _proposal_request()
    context = request.slot.context.model_copy(
        update={"run_id": f"gepa:real-dbos:{suffix}"}
    )
    authority_hash = typed_ref_for_record(
        "test.gepa.real_authority",
        {"suffix": suffix},
    ).content_hash
    authority_binding = request.authority.model_copy(
        update={"authority_identity_hash": authority_hash}
    )
    request = request.model_copy(
        update={
            "slot": request.slot.model_copy(update={"context": context}),
            "authority": authority_binding,
        }
    )
    if effect_kind == "evaluate":
        result = _evaluation_result(
            GepaEvaluationEffectRequest.model_validate(request)
        )
    else:
        result = _proposal_result(
            GepaProposalEffectRequest.model_validate(request)
        )
    authority = _DurableAuthority(authority_hash, result)
    if effect_kind == "evaluate":
        register_gepa_evaluation_authority(authority_hash, authority)
    else:
        register_gepa_proposal_authority(authority_hash, authority)
    database = tmp_path / f"real-dbos-{effect_kind}-effect.sqlite"
    store = ObjectStore(SqliteBackend(database))
    first = DbosGepaEffectBroker(store)
    recorder_method = (
        "record_evaluation_result"
        if effect_kind == "evaluate"
        else "record_proposal_result"
    )
    original_record = getattr(first._recorder, recorder_method)

    def crash_before_bind(*_args, **_kwargs):
        raise RuntimeError("injected crash before outer result bind")

    setattr(first._recorder, recorder_method, crash_before_bind)
    try:
        DBOS.launch()
        with pytest.raises(RuntimeError, match="before outer result bind"):
            getattr(first, effect_kind)(request)
        assert authority.calls == 1
        setattr(first._recorder, recorder_method, original_record)
        replay = getattr(
            DbosGepaEffectBroker(ObjectStore(SqliteBackend(database))),
            effect_kind,
        )(request)
        assert replay == result
        assert authority.calls == 1
    finally:
        DBOS.destroy()
