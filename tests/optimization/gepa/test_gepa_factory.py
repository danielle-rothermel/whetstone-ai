"""Concrete GEPA authority/factory identity and artifact tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from dr_serialize import Jsonable
from dr_store import ObjectStore, SqliteBackend

from tests.optimization.gepa.test_effects import _prompt_services
from tests.optimization.support import eval_config
from whetstone.core.effects.authority import ReplayPolicy
from whetstone.core.identity import (
    IdentityRef,
    typed_ref_for_record,
)
from whetstone.evaluation.engine import EvaluationEngine
from whetstone.experiment.binding import eval_config_reference
from whetstone.experiment.candidate import (
    Candidate,
    candidate_reference,
)
from whetstone.optimization.gepa.authorities import (
    GEPA_DATA_LOADER_IDENTITY_HASH,
    GEPA_PROPOSAL_ATTEMPT_EVIDENCE_SCHEMA,
    GEPA_WHOLE_CALL_EVIDENCE_BOUNDARY,
    CanonicalGepaCandidateAssembler,
    CanonicalGepaEvaluationAuthority,
    CanonicalGepaProposalAuthority,
    GepaCandidateFieldBinding,
    GepaDataRegistry,
)
from whetstone.optimization.gepa.contracts import GepaDataInstance
from whetstone.optimization.gepa.control import configure_gepa
from whetstone.optimization.gepa.engine import GepaDetailedResult
from whetstone.optimization.gepa.factory import CanonicalGepaAdapterFactory
from whetstone.optimization.proposal.proposer import (
    DurableProposalExecutor,
    FakeProposerTransport,
    ProposalExecutorDurabilityContract,
    ProposerConfig,
    _durable_proposal_executor,
)

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_E = "e" * 64
_F = "f" * 64
_9 = "9" * 64


def _direct_executor(
    *, policy_identity_hash: str = _F
) -> DurableProposalExecutor:
    """Mint the canonical capability over an in-process pass-through."""

    def execute(*, config, request, transport, count):
        return transport.draft(config, request, count)

    return _durable_proposal_executor(
        durability_contract=ProposalExecutorDurabilityContract(
            recovery_policy=ReplayPolicy.DURABLE_WORKFLOW,
            policy_identity_hash=policy_identity_hash,
        ),
        execute=execute,
    )


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


def test_concrete_factory_creates_fresh_bound_adapters_and_persists(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "factory.sqlite"))
    services = _prompt_services()
    base = candidate_reference(
        Candidate(
            candidate_id="gepa-base",
            base_ref=typed_ref_for_record(
                "test.gepa.base",
                {"candidate_id": "gepa-base"},
            ),
            payload={"alpha": "alpha-0", "beta": "beta-0"},
        )
    )
    assembler = CanonicalGepaCandidateAssembler(
        base_candidate=base,
        fields=(
            GepaCandidateFieldBinding(
                component_name="alpha",
                candidate_field="alpha",
            ),
            GepaCandidateFieldBinding(
                component_name="beta",
                candidate_field="beta",
            ),
        ),
    )
    transport = FakeProposerTransport(
        {},
        execution_policy_hash=_D,
        prompt_adapter_identity_hash=_E,
    )
    executor = _direct_executor()
    metric = eval_config_reference(eval_config())
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
        metric=metric,
        reward_policy_hash=_B,
        evaluation_execution_policy_hash=_C,
        proposal_execution_policy_hash=_D,
        proposal_prompt_adapter_identity_hash=_E,
        proposal_durability_policy_identity_hash=(
            executor.policy_identity_hash
        ),
        task_model_identity_hash=_A,
        prompt_format_identity_hash=services.descriptor.identity_hash(),
        prompt_binding_identity_hash=services.binding.identity_hash(),
        trainset_task_identities=(_A, _B),
        valset_task_identities=(_C,),
        component_names=("alpha", "beta"),
        num_predictors=2,
        max_metric_calls=0,
    )
    engine = cast(
        EvaluationEngine,
        SimpleNamespace(
            eval_config_ref=metric,
            task_model_identity_hash=_A,
            execution_policy_identity_hash=_C,
            reward_policy_identity_hash=_B,
            sampling=SimpleNamespace(
                task_set=SimpleNamespace(task_identities=(_A, _B, _C)),
                repeat_plan=SimpleNamespace(repeat_count=1),
                instances=(
                    SimpleNamespace(id="a", prompt_inputs={"input": "a"}),
                    SimpleNamespace(id="b", prompt_inputs={"input": "b"}),
                    SimpleNamespace(id="c", prompt_inputs={"input": "c"}),
                ),
            ),
        ),
    )
    registry = GepaDataRegistry.from_engine(store=store, engine=engine)
    assert registry.loader_identity_hash == GEPA_DATA_LOADER_IDENTITY_HASH
    evaluator = CanonicalGepaEvaluationAuthority(
        store=store,
        engine=engine,
        control=control,
        candidate_assembler=assembler,
        data_registry=registry,
    )
    # A multi-repeat engine is refused at construction, before any paid
    # evaluation: the single-repeat contract is pinned into the GEPA
    # response-parser identity.
    multi_repeat_engine = cast(
        EvaluationEngine,
        SimpleNamespace(
            eval_config_ref=engine.eval_config_ref,
            task_model_identity_hash=engine.task_model_identity_hash,
            execution_policy_identity_hash=(
                engine.execution_policy_identity_hash
            ),
            reward_policy_identity_hash=engine.reward_policy_identity_hash,
            sampling=SimpleNamespace(
                task_set=engine.sampling.task_set,
                repeat_plan=SimpleNamespace(repeat_count=2),
                instances=engine.sampling.instances,
            ),
        ),
    )
    with pytest.raises(ValueError, match="single-repeat plan"):
        CanonicalGepaEvaluationAuthority(
            store=store,
            engine=multi_repeat_engine,
            control=control,
            candidate_assembler=assembler,
            data_registry=registry,
        )
    proposer = CanonicalGepaProposalAuthority(
        store=store,
        control=control,
        prompt_services=services,
        transport=transport,
        proposal_executor=executor,
    )
    factory = CanonicalGepaAdapterFactory(
        store=store,
        run_id="gepa:factory",
        control=control,
        evaluation_authority=evaluator,
        proposal_authority=proposer,
        prompt_services=services,
    )

    first = factory.create(control=control)
    second = factory.create(control=control)

    assert first is not second
    assert first.effect_count == second.effect_count == 0
    drifted = control.model_copy(update={"seed": 9})
    with pytest.raises(ValueError, match="control identity drift"):
        factory.create(control=drifted)
    component_drift = control.model_copy(
        update={"component_names": ("beta", "alpha")}
    )
    drifted_evaluator = CanonicalGepaEvaluationAuthority(
        store=store,
        engine=engine,
        control=component_drift,
        candidate_assembler=assembler,
        data_registry=registry,
    )
    drifted_proposer = CanonicalGepaProposalAuthority(
        store=store,
        control=component_drift,
        prompt_services=services,
        transport=transport,
        proposal_executor=executor,
    )
    with pytest.raises(ValueError, match="component names"):
        CanonicalGepaAdapterFactory(
            store=store,
            run_id="gepa:component-drift",
            control=component_drift,
            evaluation_authority=drifted_evaluator,
            proposal_authority=drifted_proposer,
            prompt_services=services,
        )
    detail = GepaDetailedResult(
        candidates=({"alpha": "alpha-0", "beta": "beta-0"},),
        parents=((),),
        val_aggregate_scores=(0.0,),
        val_subscores=({_C: 0.0},),
        per_val_instance_best_candidates={_C: (0,)},
        discovery_eval_counts=(0,),
        seed=control.seed,
        best_idx=0,
        control_identity_hash=control.identity_hash(),
    )
    artifact_ref = factory.persist_result(
        control=control,
        adapter=first,
        detailed_result=detail,
    )
    assert store.get(artifact_ref.reference)

    other_executor = _direct_executor(policy_identity_hash=_9)
    other_control = control.model_copy(
        update={
            "proposal_durability_policy_identity_hash": (
                other_executor.policy_identity_hash
            )
        }
    )
    # An executor advertising a different durability policy cannot serve an
    # authority whose control names the original one.
    with pytest.raises(ValueError, match="prompt/durability"):
        CanonicalGepaProposalAuthority(
            store=store,
            control=control,
            prompt_services=services,
            transport=transport,
            proposal_executor=other_executor,
        )
    # Binding control and executor to the same policy identity is accepted.
    CanonicalGepaProposalAuthority(
        store=store,
        control=other_control,
        prompt_services=services,
        transport=transport,
        proposal_executor=other_executor,
    )


def test_authority_refuses_a_structurally_similar_executor(tmp_path) -> None:
    """Only the exact canonical capability may carry the paid GEPA call."""

    store = ObjectStore(SqliteBackend(tmp_path / "structural.sqlite"))
    services = _prompt_services()
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
    """The persisted coarsest-boundary record keeps its exact literals."""

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
