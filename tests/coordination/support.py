"""Shared proposal-provider durability test support."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from tests.provider import support as provider_support
from whetstone.core.identity import IdentityRef, typed_ref_for_record
from whetstone.optimization.proposal.proposer import (
    ProposerConfig,
    ProviderProposerTransport,
)
from whetstone.provider.language_model import PlainPromptAdapter


class ReplayDbos:
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


def reset_replay_dbos() -> None:
    ReplayDbos.workflow_id = "proposal-workflow"
    ReplayDbos.retries_allowed.clear()
    ReplayDbos.checkpoints.clear()
    ReplayDbos.sleeps.clear()
    ReplayDbos.events.clear()
    ReplayDbos.child_results.clear()
    ReplayDbos.next_workflow_id = None
    ReplayDbos.step_id = None
    ReplayDbos.before_checkpoint_publication = None


class _ReplayHandle:
    def __init__(self, dbos_type, workflow_id: str) -> None:
        self._dbos_type = dbos_type
        self._workflow_id = workflow_id

    def get_result(self):
        return self._dbos_type.get_child_result(self._workflow_id)


def load_proposal_provider_boundary(dbos_type: type[Any]):
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


def proposal_executor(module, transport):
    registry_key = module.register_proposal_transport(transport)
    return module.DbosProposalExecutor(transport_registry_key=registry_key)


def provider_transport(
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
    return proposer, _proposer_config(provider_config), recording


def _proposer_config(provider_config=None) -> ProposerConfig:
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
