from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from dr_store import ObjectStore, SqliteBackend

from tests.optimization.gepa.support import prompt_services
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
    CanonicalGepaCandidateAssembler,
    CanonicalGepaEvaluationAuthority,
    CanonicalGepaProposalAuthority,
    GepaCandidateFieldBinding,
    GepaDataRegistry,
)
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

    def execute(*, config, request, transport, count):
        return transport.draft(config, request, count)

    return _durable_proposal_executor(
        durability_contract=ProposalExecutorDurabilityContract(
            recovery_policy=ReplayPolicy.DURABLE_WORKFLOW,
            policy_identity_hash=policy_identity_hash,
        ),
        execute=execute,
    )


def test_concrete_factory_creates_fresh_bound_adapters_and_persists(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "factory.sqlite"))
    services = prompt_services()
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
    with pytest.raises(ValueError, match="prompt/durability"):
        CanonicalGepaProposalAuthority(
            store=store,
            control=control,
            prompt_services=services,
            transport=transport,
            proposal_executor=other_executor,
        )
    CanonicalGepaProposalAuthority(
        store=store,
        control=other_control,
        prompt_services=services,
        transport=transport,
        proposal_executor=other_executor,
    )
