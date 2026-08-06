from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from dr_serialize import Jsonable
from dr_store import ObjectStore, SqliteBackend

from tests.optimization.gepa.support import prompt_services
from tests.optimization.support import eval_config
from whetstone.core.effects.authority import ReplayPolicy
from whetstone.core.identity import IdentityRef, typed_ref_for_record
from whetstone.experiment.binding import eval_config_reference
from whetstone.optimization.gepa.authorities import (
    GEPA_PROPOSAL_ATTEMPT_EVIDENCE_SCHEMA,
    GEPA_WHOLE_CALL_EVIDENCE_BOUNDARY,
    CanonicalGepaProposalAuthority,
    GepaDataRegistry,
)
from whetstone.optimization.gepa.contracts import GepaDataInstance
from whetstone.optimization.gepa.control import configure_gepa
from whetstone.optimization.proposal.proposer import (
    DurableProposalExecutor,
    FakeProposerTransport,
    ProposerConfig,
)

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_E = "e" * 64
_F = "f" * 64


def _data(store: ObjectStore, index: int, data_id: str) -> GepaDataInstance:
    record = {"data_id": data_id, "input": str(index)}
    json_record = cast(Jsonable, record)
    _ref, _ = store.put("test.gepa.data", json_record)
    return GepaDataInstance(
        upstream_position=index,
        data_id=data_id,
        data_ref=typed_ref_for_record("test.gepa.data", json_record),
        loader_identity_hash=_F,
    )


def test_data_registry_binds_position_order_and_exact_refs(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "registry.sqlite"))
    first = _data(store, 0, _A)
    second = _data(store, 1, _B)
    registry = GepaDataRegistry(
        loader_identity_hash=_F,
        entries=(first, second),
    )

    registry.require_exact(first)
    with pytest.raises(ValueError, match="immutable data registry"):
        registry.require_exact(
            first.model_copy(update={"data_ref": second.data_ref})
        )
    with pytest.raises(ValueError, match="positions"):
        GepaDataRegistry(
            loader_identity_hash=_F,
            entries=(
                first.model_copy(update={"upstream_position": 1}),
                second.model_copy(update={"upstream_position": 0}),
            ),
        )


def test_authority_refuses_a_structurally_similar_executor(tmp_path) -> None:

    store = ObjectStore(SqliteBackend(tmp_path / "structural.sqlite"))
    services = prompt_services()
    transport = FakeProposerTransport(
        {},
        execution_policy_hash=_D,
        prompt_adapter_identity_hash=_E,
    )
    control = configure_gepa(
        reflection_model=ProposerConfig(
            provider_call_config=IdentityRef(
                record_ref=typed_ref_for_record(
                    "dr_providers.provider_call_config",
                    {"provider_call_config_ref": "provider://gepa"},
                ),
                identity_hash=_A,
            ),
        ),
        metric=eval_config_reference(eval_config()),
        reward_policy_hash=_B,
        evaluation_execution_policy_hash=_C,
        proposal_execution_policy_hash=_D,
        proposal_prompt_adapter_identity_hash=_E,
        proposal_durability_policy_identity_hash=_F,
        task_model_identity_hash=_A,
        prompt_format_identity_hash=services.descriptor.identity_hash(),
        prompt_binding_identity_hash=services.binding.identity_hash(),
        trainset_task_identities=(_A, _B),
        valset_task_identities=(_C,),
        component_names=("alpha", "beta"),
        num_predictors=2,
        max_metric_calls=0,
    )

    class _StructuralExecutor:
        policy_identity_hash = _F
        recovery_policy = ReplayPolicy.DURABLE_WORKFLOW

        def execute(self, *, config, request, transport, count):
            raise AssertionError("structural executor must never be invoked")

    with pytest.raises(TypeError, match="canonical DurableProposalExecutor"):
        CanonicalGepaProposalAuthority(
            store=store,
            control=control,
            prompt_services=services,
            transport=transport,
            proposal_executor=cast(
                DurableProposalExecutor,
                _StructuralExecutor(),
            ),
        )


def test_whole_call_evidence_boundary_literals_are_pinned(tmp_path) -> None:

    assert GEPA_WHOLE_CALL_EVIDENCE_BOUNDARY == "whole_call"
    assert GEPA_PROPOSAL_ATTEMPT_EVIDENCE_SCHEMA == (
        "whetstone.gepa.proposal_provider_attempt"
    )
    store = ObjectStore(SqliteBackend(tmp_path / "whole-call.sqlite"))
    authority = cast(Any, SimpleNamespace(_store=store))
    (ref,) = CanonicalGepaProposalAuthority._persist_attempt_evidence(
        authority,
        {"finish": "failed"},
    )

    assert ref.schema_name == GEPA_PROPOSAL_ATTEMPT_EVIDENCE_SCHEMA
    assert store.get(ref.reference) == {
        "boundary": "whole_call",
        "response_evidence": {"finish": "failed"},
    }
