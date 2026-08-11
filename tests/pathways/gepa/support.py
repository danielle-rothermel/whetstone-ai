"""Shared DBOS fakes and loaders for GEPA pathway tests."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

from tests.optimization.gepa.support import (
    effect_context,
    evaluation_request,
    make_gepa_detailed_result,
    prompt_services,
    proposal_authority_binding,
)
from whetstone.core.identity import typed_ref_for_record
from whetstone.optimization.gepa.contracts import (
    GepaCandidateComponent,
    GepaEffectSlot,
    GepaProposalEffectRequest,
    GepaProposalEffectResult,
)
from whetstone.optimization.gepa.prompts import GepaRenderedPrompt


class _RunnerReplayDbos:
    workflow_ids: ClassVar[list[str]] = []

    @classmethod
    def workflow(cls):
        def decorate(function):
            return function

        return decorate


class _RunnerSetWorkflowID:
    def __init__(self, workflow_id: str) -> None:
        self._workflow_id = workflow_id

    def __enter__(self):
        _RunnerReplayDbos.workflow_ids.append(self._workflow_id)
        return self

    def __exit__(self, *_args):
        return False


def load_runner():
    # Import the real DBOS API before replacing only the runner's decorator
    # seam.
    import whetstone.optimization.gepa.factory  # noqa: F401

    fake_dbos = types.ModuleType("dbos")
    fake_dbos.__dict__["DBOS"] = _RunnerReplayDbos
    fake_dbos.__dict__["SetWorkflowID"] = _RunnerSetWorkflowID
    prior = sys.modules.get("dbos")
    sys.modules["dbos"] = fake_dbos
    module_name = "_gepa_runner_replay_test"
    try:
        path = Path("src/whetstone/optimization/gepa/runner.py")
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


class GepaAdapterFactory:
    def __init__(self, control, *, identity_salt: str = "unit") -> None:
        self.control = control
        self.runtime_hash = typed_ref_for_record(
            "test.gepa.factory",
            {
                "control": control.identity_hash(),
                "identity_salt": identity_salt,
            },
        ).content_hash
        self.create_calls = 0
        self.persist_calls = 0

    def create(self, *, control):
        assert control == self.control
        self.create_calls += 1
        return SimpleNamespace(effect_ordinal=0)

    def persist_result(self, *, control, adapter, detailed_result):
        assert control == self.control
        assert adapter.effect_ordinal == 7
        assert detailed_result == make_gepa_detailed_result(control)
        self.persist_calls += 1
        return typed_ref_for_record(
            "whetstone.gepa.run_result_artifact",
            {"control": control.identity_hash()},
        )


class DurableAuthority:
    def __init__(self, identity_hash: str, result) -> None:
        self.runtime_hash = identity_hash
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


class ReplayDbos:
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


def load_boundary(dbos_type=ReplayDbos):
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


def proposal_request() -> GepaProposalEffectRequest:
    services = prompt_services()
    return GepaProposalEffectRequest(
        slot=GepaEffectSlot(context=effect_context(), invocation_ordinal=0),
        candidate=(
            GepaCandidateComponent(name="alpha", text="alpha-0"),
            GepaCandidateComponent(name="beta", text="beta-0"),
        ),
        components_to_update=("alpha", "beta"),
        component_name="alpha",
        rendered_prompt=GepaRenderedPrompt(text="Improve alpha."),
        authority=proposal_authority_binding(services),
    )


def proposal_result(
    request: GepaProposalEffectRequest,
) -> GepaProposalEffectResult:
    attempt_ref = typed_ref_for_record(
        "test.gepa.proposal_attempt",
        {"request": request.identity_hash()},
    )
    return GepaProposalEffectResult(
        request_hash=request.identity_hash(),
        raw_response="```\nalpha-improved\n```",
        parsed_components=(
            GepaCandidateComponent(name="alpha", text="alpha-improved"),
        ),
        request_evidence={"prompt": request.rendered_prompt.text},
        response_evidence={"raw": "alpha-improved"},
        provider_attempt_refs=(attempt_ref,),
    )


__all__ = [
    "DurableAuthority",
    "GepaAdapterFactory",
    "ReplayDbos",
    "evaluation_request",
    "load_boundary",
    "load_runner",
    "proposal_request",
    "proposal_result",
]
