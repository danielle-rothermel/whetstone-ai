"""Materialization Record, proposal/diff-check, selection, planning
(deliverable 6)."""

from __future__ import annotations

import pytest
from dr_serialize import IdentityDocumentError, validate_identity_document
from dr_store import MemoryBackend, ObjectStore

from tests.graph.support import (
    EVALUATION_PROCEDURE_CONFIG_SCHEMA,
    PROVIDER_CALL_CONFIG_SCHEMA,
    eval_config,
    fake_hash,
    llm_eval_graph_definition,
)
from whetstone.graph.materialization import (
    MATERIALIZATION_RECORD_SCHEMA,
    OptimizationProposal,
    ProposalValidationError,
    VarianceAxis,
    diff_check,
    distinct_graph_hashes,
    expand_axes,
    materialization_record_reference,
    materialize_proposal,
    plan_execution,
    select_ordered_subset,
    store_materialization_record,
    validate_proposal,
)
from whetstone.graph.nodes import (
    eval_variable_assignment,
    llm_call_variable_assignment,
)
from whetstone.graph.rollout import EvaluationContext, EvaluationRole

MUTATION_SURFACE = frozenset({"generate.provider", "generate.temperature"})


# --- Proposal validation + Diff Check --------------------------------------


def test_diff_check_accepts_changes_inside_mutation_surface() -> None:
    proposal = OptimizationProposal(
        base_assignment={"generate.provider": "openai"},
        assignment={"generate.provider": "openrouter"},
    )
    changed = diff_check(proposal, mutation_surface=MUTATION_SURFACE)
    assert changed == frozenset({"generate.provider"})


def test_diff_check_rejects_changes_outside_mutation_surface() -> None:
    proposal = OptimizationProposal(
        base_assignment={"evaluate.procedure": "p1"},
        assignment={"evaluate.procedure": "p2"},
    )
    with pytest.raises(ProposalValidationError, match="Mutation Surface"):
        diff_check(proposal, mutation_surface=MUTATION_SURFACE)


def test_validate_proposal_rejects_assignment_outside_surface() -> None:
    proposal = OptimizationProposal(
        assignment={"evaluate.procedure": "p2"},
    )
    with pytest.raises(ProposalValidationError, match="not in the Mutation"):
        validate_proposal(proposal, mutation_surface=MUTATION_SURFACE)


# --- Fixed-axis expansion --------------------------------------------------


def test_expand_axes_produces_one_slot_per_combination() -> None:
    axes = [
        VarianceAxis(name="temp", values=(0.0, 0.7)),
        VarianceAxis(name="seed", values=(1, 2, 3)),
    ]
    combos = expand_axes(axes)
    assert len(combos) == 6
    assert {"temp": 0.0, "seed": 1} in combos


def test_expand_axes_with_no_axes_is_one_empty_slot() -> None:
    assert expand_axes([]) == [{}]


# --- Materialization Record schema -----------------------------------------


def _proc_hash() -> str:
    return eval_config().evaluation_procedure_config_hash


def _build_assignments(procedure_hash: str, provider_hash: str):
    def build(_proposal_assignment, _axis_selection):
        return {
            "generate": llm_call_variable_assignment(
                provider_call_config_schema=PROVIDER_CALL_CONFIG_SCHEMA,
                provider_call_config_hash=provider_hash,
            ),
            "evaluate": eval_variable_assignment(
                evaluation_procedure_config_schema=(
                    EVALUATION_PROCEDURE_CONFIG_SCHEMA
                ),
                evaluation_procedure_config_hash=procedure_hash,
            ),
        }

    return build


def _materialize(provider_hash: str, procedure_hash: str):
    proposal = OptimizationProposal(
        base_assignment={"generate.provider": "openai"},
        assignment={"generate.provider": "openai"},
    )
    return materialize_proposal(
        definition=llm_eval_graph_definition(),
        proposal=proposal,
        mutation_surface=MUTATION_SURFACE,
        axes=[],
        build_assignments=_build_assignments(procedure_hash, provider_hash),
        materializer_name="whetstone.proposal_materializer",
        materializer_version="1",
        definition_ref="whetstone.rollout_definition/v1",
    )


def test_materialization_emits_one_record_per_slot() -> None:
    slots = _materialize(fake_hash("a"), _proc_hash())
    assert len(slots) == 1
    record = slots[0].record
    assert record.graph_hash == slots[0].record.graph_config_ref
    assert record.materializer_name == "whetstone.proposal_materializer"
    assert record.diff_check_passed
    assert record.validation_passed


def test_record_field_groups_present() -> None:
    record = _materialize(fake_hash("a"), _proc_hash())[0].record
    dumped = record.model_dump()
    # definition, source lineage, assignments/axes, materializer, result,
    # validation/diff, provenance groups.
    for field in (
        "definition_ref",
        "source_proposal_assignment",
        "assignments",
        "axis_selections",
        "materializer_name",
        "graph_config_ref",
        "graph_hash",
        "changed_paths",
        "diff_check_passed",
    ):
        assert field in dumped


def test_record_has_no_identity_hash_field() -> None:
    record = _materialize(fake_hash("a"), _proc_hash())[0].record
    dumped = record.model_dump()
    assert "identity_hash" not in dumped
    assert "config_identity_hash" not in dumped


def test_record_is_not_a_valid_identity_document() -> None:
    # A Materialization Record is stored by Content Hash only; it is NOT an
    # Identity Document and must not be accepted as one.
    record = _materialize(fake_hash("a"), _proc_hash())[0].record
    with pytest.raises(IdentityDocumentError):
        validate_identity_document(record.model_dump(mode="json"))


def test_record_stored_only_by_typed_object_reference() -> None:
    record = _materialize(fake_hash("a"), _proc_hash())[0].record
    store = ObjectStore(MemoryBackend())
    reference = store_materialization_record(store, record)
    assert reference.schema == MATERIALIZATION_RECORD_SCHEMA
    assert len(reference.content_hash) == 64
    # It resolves back verified through Content Hash.
    assert store.get(reference) == record.record_content()
    # The reference is exactly the content-addressed one.
    assert reference == materialization_record_reference(record)


# --- Selection Policy ------------------------------------------------------


def test_selection_policy_selects_ordered_subset() -> None:
    axes = [VarianceAxis(name="k", values=(1, 2, 3, 4))]
    slots = materialize_proposal(
        definition=llm_eval_graph_definition(),
        proposal=OptimizationProposal(),
        mutation_surface=MUTATION_SURFACE,
        axes=axes,
        build_assignments=_build_assignments(_proc_hash(), fake_hash("a")),
        materializer_name="m",
        materializer_version="1",
        definition_ref="d",
    )
    selected = select_ordered_subset(slots, selected_indices=[0, 2])
    assert len(selected) == 2
    assert selected[0] is slots[0]
    assert selected[1] is slots[2]


def test_selection_indices_must_be_ascending_subset() -> None:
    slots = _materialize(fake_hash("a"), _proc_hash())
    with pytest.raises(ValueError, match="ascending"):
        select_ordered_subset(slots, selected_indices=[0, 0])


# --- Execution planning: dedup by graph_hash -------------------------------


def _context() -> EvaluationContext:
    ec = eval_config()
    return EvaluationContext(
        eval_config_hash=ec.config_identity_hash,
        role=EvaluationRole.INTERNAL,
        campaign="camp-1",
    )


def test_two_records_sharing_one_graph_hash_dedup_to_one_matrix() -> None:
    """Deliverable-6 fixture: two selected records sharing one graph_hash ->
    one work matrix, one aggregate, no duplicate execution keys."""
    proc = _proc_hash()
    provider = fake_hash("a")
    # Two independent materializations that produce the SAME graph_hash.
    slot_a = _materialize(provider, proc)[0]
    slot_b = _materialize(provider, proc)[0]
    assert slot_a.record.graph_hash == slot_b.record.graph_hash

    selected = [slot_a, slot_b]
    ec = eval_config()
    context = EvaluationContext(
        eval_config_hash=ec.config_identity_hash,
        role=EvaluationRole.INTERNAL,
        campaign="camp-1",
    )
    plan = plan_execution(
        selected=selected,
        selected_refs=["ref-a", "ref-b"],
        eval_config_hash=ec.config_identity_hash,
        context=context,
        task_identities=["t0", "t1"],
        repeat_ids=["r0", "r1", "r2"],
    )
    # D = 1 distinct admitted graph_hash.
    assert plan.distinct_admitted_count == 1
    # Exactly D x task_count x repeat_count = 1 x 2 x 3 = 6 work items.
    assert len(plan.work_items) == 6
    # D = 1 aggregate.
    assert len(plan.aggregates) == 1
    # No duplicate Rollout Execution Keys.
    keys = [wi.execution_key for wi in plan.work_items]
    assert len(keys) == len(set(keys))
    # Both selected records remain separately attributable while sharing the
    # graph_hash.
    assert plan.record_ref_to_graph_hash == (
        ("ref-a", slot_a.record.graph_hash),
        ("ref-b", slot_b.record.graph_hash),
    )


def test_distinct_graph_hashes_produce_distinct_matrices() -> None:
    proc = _proc_hash()
    slot_a = _materialize(fake_hash("a"), proc)[0]
    slot_b = _materialize(fake_hash("b"), proc)[0]
    assert slot_a.record.graph_hash != slot_b.record.graph_hash
    ec = eval_config()
    plan = plan_execution(
        selected=[slot_a, slot_b],
        selected_refs=["ref-a", "ref-b"],
        eval_config_hash=ec.config_identity_hash,
        context=_context(),
        task_identities=["t0"],
        repeat_ids=["r0"],
    )
    assert plan.distinct_admitted_count == 2
    assert len(plan.work_items) == 2  # 2 x 1 x 1
    assert len(plan.aggregates) == 2


def test_distinct_graph_hashes_helper() -> None:
    proc = _proc_hash()
    a = _materialize(fake_hash("a"), proc)[0].record
    b = _materialize(fake_hash("a"), proc)[0].record
    c = _materialize(fake_hash("b"), proc)[0].record
    assert distinct_graph_hashes([a, b, c]) == [a.graph_hash, c.graph_hash]


def test_plan_requires_aligned_selected_refs() -> None:
    slot = _materialize(fake_hash("a"), _proc_hash())[0]
    with pytest.raises(ValueError, match="align"):
        plan_execution(
            selected=[slot],
            selected_refs=["a", "b"],
            eval_config_hash=eval_config().config_identity_hash,
            context=_context(),
            task_identities=["t0"],
            repeat_ids=["r0"],
        )


def test_record_reference_not_in_rollout_or_work_keys() -> None:
    # Rollout Execution Keys carry no Materialization Record reference.
    slot = _materialize(fake_hash("a"), _proc_hash())[0]
    plan = plan_execution(
        selected=[slot],
        selected_refs=["ref-a"],
        eval_config_hash=eval_config().config_identity_hash,
        context=_context(),
        task_identities=["t0"],
        repeat_ids=["r0"],
    )
    dumped = plan.work_items[0].model_dump()
    flat = repr(dumped)
    assert "ref-a" not in flat
    assert "materializ" not in flat.lower()
