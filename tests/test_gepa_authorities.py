"""Bind the canonical GEPA authorities against a real RuntimeEvalEngine."""

from __future__ import annotations

import pytest

from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.experiment.candidate import candidate_reference
from whetstone.optim.gepa.authorities import (
    CanonicalGepaCandidateAssembler,
    CanonicalGepaEvalAuthority,
    GepaCandidateFieldBinding,
    GepaDataRegistry,
)
from whetstone.optim.gepa.contracts import (
    GepaCandidateComponent,
    GepaEffectContext,
    GepaEffectSlot,
    GepaEvaluationEffectRequest,
)
from whetstone.optim.gepa.control import configure_gepa
from whetstone.optim.gepa.upstream_adapter import (
    GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH,
)
from whetstone.optim.proposal.proposer import (
    ProposerConfig,
    prompt_adapter_identity_hash,
)
from whetstone.provider.language_model import PlainPromptAdapter
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    build_toy_experiment,
)

GEPA_COMPONENT = "generate"


def _toy_gepa_control(*, engine, experiment, max_metric_calls: int):
    prompt_adapter = PlainPromptAdapter()
    return configure_gepa(
        reflection_model=ProposerConfig(
            provider_call_config=engine.provider_execution_policy_ref,
        ),
        metric=engine.eval_config_ref,
        reward_policy_hash=experiment.reward_policy.identity_hash(),
        evaluation_execution_policy_hash=(
            engine.execution_policy_identity_hash()
        ),
        proposal_execution_policy_hash=(
            engine.execution_policy_identity_hash()
        ),
        proposal_prompt_adapter_identity_hash=prompt_adapter_identity_hash(
            prompt_adapter
        ),
        proposal_durability_policy_identity_hash="c" * 64,
        task_model_identity_hash=engine.task_model_identity_hash(),
        prompt_format_identity_hash="d" * 64,
        prompt_binding_identity_hash="e" * 64,
        trainset_task_hashes=engine.sampling.task_hashes,
        valset_task_hashes=None,
        component_names=(GEPA_COMPONENT,),
        num_predictors=1,
        max_metric_calls=max_metric_calls,
    )


def _build(store):
    experiment = build_toy_experiment(num_seeds=1)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        store, experiment=experiment
    )
    control = _toy_gepa_control(
        engine=engine, experiment=experiment, max_metric_calls=2
    )
    registry = GepaDataRegistry.from_engine(store=store, engine=engine)
    assembler = CanonicalGepaCandidateAssembler(
        base_candidate=candidate_reference(experiment.initial_candidate),
        fields=(
            GepaCandidateFieldBinding(
                component_name=GEPA_COMPONENT,
                candidate_field=TOY_MUTATION_FIELD,
            ),
        ),
    )
    return experiment, engine, control, registry, assembler


def test_eval_authority_binds_a_real_runtime_eval_engine(sqlite_store) -> None:
    """A RuntimeEvalEngine binds without any consumer-side hash shim."""
    _experiment, engine, control, registry, assembler = _build(sqlite_store)

    authority = CanonicalGepaEvalAuthority(
        store=sqlite_store,
        engine=engine,
        control=control,
        candidate_assembler=assembler,
        data_registry=registry,
    )

    binding = authority.binding
    assert binding.provider_route_identity_hash == (
        engine.task_model_identity_hash()
    )
    assert binding.execution_policy_identity_hash == (
        engine.execution_policy_identity_hash()
    )
    assert binding.reward_policy_identity_hash == (
        engine.reward_policy_identity_hash()
    )
    assert authority.component_names == (GEPA_COMPONENT,)


def test_eval_authority_rejects_a_conflicting_task_model_route(
    sqlite_store,
) -> None:
    _experiment, engine, control, registry, assembler = _build(sqlite_store)
    drifted = control.model_copy(
        update={"task_model_identity_hash": "f" * 64}
    )

    with pytest.raises(ValueError, match="task-model route"):
        CanonicalGepaEvalAuthority(
            store=sqlite_store,
            engine=engine,
            control=drifted,
            candidate_assembler=assembler,
            data_registry=registry,
        )


def test_data_registry_keys_entries_by_engine_task_id(sqlite_store) -> None:
    """The registry data_id is the engine task_id the engine resolves."""
    _experiment, engine, _control, registry, _assembler = _build(sqlite_store)

    task_ids = tuple(task.task_id for task in engine.sampling.tasks)
    assert registry.data_ids == task_ids
    assert registry.task_hashes == engine.sampling.task_hashes
    for entry, task in zip(registry.entries, engine.sampling.tasks, strict=True):
        assert entry.data_id == task.task_id
        assert entry.task_hash == task.task_hash

    # The engine resolves the registry data ids directly.
    subset = engine.for_task_ids(task_ids[:1])
    assert subset.sampling.task_hashes == engine.sampling.task_hashes[:1]


def test_eval_authority_evaluates_against_a_real_engine(sqlite_store) -> None:
    """evaluate() resolves task ids and verifies the persisted candidate."""
    experiment, engine, control, registry, assembler = _build(sqlite_store)
    authority = CanonicalGepaEvalAuthority(
        store=sqlite_store,
        engine=engine,
        control=control,
        candidate_assembler=assembler,
        data_registry=registry,
    )
    seed_text = experiment.initial_candidate.payload[TOY_MUTATION_FIELD]
    request = GepaEvaluationEffectRequest(
        slot=GepaEffectSlot(
            context=GepaEffectContext(
                run_id="gepa-authority-eval",
                control_identity_hash=control.identity_hash(),
                source_manifest_identity_hash=(
                    control.gepa_source_manifest_hash
                ),
                adapter_identity_hash=GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH,
            ),
            invocation_ordinal=0,
        ),
        candidate=(
            GepaCandidateComponent(name=GEPA_COMPONENT, text=seed_text),
        ),
        data=registry.entries,
        capture_traces=True,
        authority=authority.binding,
    )

    result = authority.evaluate(request)

    assert result.request_hash == request.identity_hash()
    assert result.logical_metric_calls == len(registry.entries)
    assert len(result.rows) == len(registry.entries)
    for row, entry in zip(result.rows, registry.entries, strict=True):
        assert row.data == entry
        assert row.failure_ref is None
        assert row.evidence_refs
        assert row.trajectory is not None
        assert row.trajectory.data_id == entry.data_id
