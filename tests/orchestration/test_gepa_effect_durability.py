"""Crash-window tests for GEPA's stable child effect broker."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import ClassVar, Literal

import pytest
from dr_store import ObjectStore, SqliteBackend

from tests.optimization.test_gepa_effects import (
    _context,
    _evaluation_request,
    _evaluation_result,
    _prompt_services,
    _proposal_authority,
)
from tests.orchestration.test_proposal_provider_durability import (
    _executor,
    _OrderedReplayDbos,
    _provider_transport,
)
from tests.orchestration.test_proposal_provider_durability import (
    _load_boundary as _load_provider_boundary,
)
from tests.provider import support as provider_support
from whetstone.optimization.gepa_effects import (
    GepaCandidateComponent,
    GepaEffectRecorder,
    GepaEffectSlot,
    GepaEvaluationEffectRequest,
    GepaProposalEffectRequest,
    GepaProposalEffectResult,
)
from whetstone.optimization.gepa_prompts import GepaRenderedPrompt
from whetstone.optimization.identity import typed_ref_for_record
from whetstone.optimization.proposer import ProposalRequest
from whetstone.orchestration.gepa_authorities import (
    CanonicalGepaProposalAuthority,
)


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
        path = Path("src/whetstone/orchestration/gepa_effects.py")
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


class _SynchronousChildOrderedDbos(_OrderedReplayDbos):
    """Model the parent operation occupied by a synchronous child workflow."""

    @classmethod
    def workflow(cls):
        def decorate(function):
            def checkpointed(*args):
                child_id = cls.workflow_id

                def execute_child():
                    prior_child = cls.current_child_id
                    cls.current_child_id = child_id
                    cls.child_cursors[child_id] = 0
                    try:
                        return function(*args)
                    finally:
                        cls.current_child_id = prior_child

                return cls._operation(
                    f"workflow:{function.__name__}:{child_id}",
                    execute_child,
                )

            return checkpointed

        return decorate


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


def test_proposal_replay_preserves_inner_operation_sequence_without_recall(
    tmp_path,
) -> None:
    """A draft bind cannot shorten the child replay transcript."""

    _OrderedReplayDbos.reset()
    provider_module = _load_provider_boundary(_OrderedReplayDbos)
    effect_module = _load_boundary(_OrderedReplayDbos)
    request = _proposal_request()
    transport, proposer_config, recording = _provider_transport(
        provider_support.response_outcome(text="alpha-improved"),
    )
    executor = _executor(
        provider_module,
        transport,
        registry_key="9" * 64,
    )
    database = tmp_path / "proposal-inner-replay.sqlite"
    store = ObjectStore(SqliteBackend(database))
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
    broker = effect_module.DbosGepaEffectBroker(store)
    original_record = broker._recorder.record_proposal_result

    def crash_before_outer_bind(*_args, **_kwargs):
        raise RuntimeError("injected crash before GEPA child commit")

    broker._recorder.record_proposal_result = crash_before_outer_bind
    with pytest.raises(RuntimeError, match="before GEPA child commit"):
        broker.propose(request)
    assert len(recording.served) == 1

    broker._recorder.record_proposal_result = original_record
    _OrderedReplayDbos.begin_replay()
    replay = broker.propose(request)

    assert replay.parsed_components[0].text == "alpha-improved"
    assert len(recording.served) == 1
    assert _OrderedReplayDbos.cursor == 1
    assert len(_OrderedReplayDbos.checkpoints) == 1


def test_parent_replay_always_consumes_stable_child_operation(
    tmp_path,
) -> None:
    _SynchronousChildOrderedDbos.reset()
    provider_module = _load_provider_boundary(_SynchronousChildOrderedDbos)
    effect_module = _load_boundary(_SynchronousChildOrderedDbos)
    request = _proposal_request()
    transport, proposer_config, recording = _provider_transport(
        provider_support.response_outcome(text="alpha-improved"),
    )
    executor = _executor(
        provider_module,
        transport,
        registry_key="8" * 64,
    )
    store = ObjectStore(
        SqliteBackend(tmp_path / "parent-operation-replay.sqlite")
    )
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
    broker = effect_module.DbosGepaEffectBroker(store)

    first = broker.propose(request)

    @_SynchronousChildOrderedDbos.step(retries_allowed=False)
    def later_operation(value: str) -> str:
        return f"later:{value}"

    assert later_operation("safe") == "later:safe"
    _SynchronousChildOrderedDbos.begin_replay()

    replay = broker.propose(request)
    assert later_operation("safe") == "later:safe"

    assert replay == first
    assert len(recording.served) == 1
    assert _SynchronousChildOrderedDbos.cursor == 2
    assert len(_SynchronousChildOrderedDbos.checkpoints) == 2


@pytest.mark.parametrize("effect_kind", ["evaluate", "propose"])
def test_real_dbos_child_checkpoint_survives_outer_bind_crash(
    pg_engine,
    tmp_path,
    effect_kind: Literal["evaluate", "propose"],
) -> None:
    from uuid import uuid4

    from dbos import DBOS, DBOSConfig

    from whetstone.orchestration.gepa_effects import (
        DbosGepaEffectBroker,
        register_gepa_evaluation_authority,
        register_gepa_proposal_authority,
    )

    suffix = uuid4().hex[:10]
    database_url = pg_engine.url.render_as_string(hide_password=False)
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
