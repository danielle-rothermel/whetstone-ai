"""Official selection over complete certified aggregate evidence.

Proves deterministic derivation of Objective Vectors from certified aggregates,
the deterministic Pareto Front + explicit tie behavior, the single official
selection, the persisted derivation/order/tie/selection evidence, refusal of
incomplete or missing evidence, and that Reward is never computed or accepted
as an Objective on the official-selection path.
"""

from __future__ import annotations

import pytest
from dr_code.eval import AggregationStatus

from whetstone.objectives import (
    Direction,
    IncompleteEvidenceError,
    ObjectiveSpec,
    SelectionCandidate,
    SelectionEvidence,
    TieBehavior,
    select_official,
)

from .support import (
    GRAPH_A,
    GRAPH_B,
    compression_aggregate,
    incomplete_pass_rate_aggregate,
    pass_rate_aggregate,
)

SPECS = (
    ObjectiveSpec(
        objective_name="pass_rate",
        aggregate_name="average_binary_test_pass_rate",
        direction=Direction.MAXIMIZE,
    ),
    ObjectiveSpec(
        objective_name="compression",
        aggregate_name="mean_compression_ratio",
        direction=Direction.MINIMIZE,
    ),
)


def _candidate(
    *,
    candidate_id: str,
    graph_hash: str,
    pass_value: float,
    compression_value: float,
) -> SelectionCandidate:
    return SelectionCandidate(
        candidate_id=candidate_id,
        graph_hash=graph_hash,
        aggregates={
            "average_binary_test_pass_rate": pass_rate_aggregate(
                graph_hash=graph_hash, value=pass_value
            ),
            "mean_compression_ratio": compression_aggregate(
                graph_hash=graph_hash, value=compression_value
            ),
        },
    )


def test_selection_derives_and_selects_non_dominated() -> None:
    candidates = [
        _candidate(
            candidate_id="c0",
            graph_hash=GRAPH_A,
            pass_value=1.0,
            compression_value=2.0,
        ),
        _candidate(
            candidate_id="c1",
            graph_hash=GRAPH_B,
            pass_value=0.5,
            compression_value=3.0,
        ),
    ]
    evidence = select_official(candidates, objective_specs=SPECS)
    # c0 dominates c1 (higher pass, lower compression).
    assert evidence.selected_candidate_id == "c0"
    assert [m.candidate_id for m in evidence.front.members] == ["c0"]
    assert not evidence.selected_by_tie_rule


def test_selection_persists_derivation_order_tie_selection() -> None:
    candidates = [
        _candidate(
            candidate_id="c0",
            graph_hash=GRAPH_A,
            pass_value=1.0,
            compression_value=2.0,
        ),
    ]
    evidence = select_official(candidates, objective_specs=SPECS)
    # Derivation: objective specs preserved; each objective carries lineage.
    assert evidence.objective_specs == SPECS
    vector = evidence.candidate_vectors[0]
    assert vector.names == ("pass_rate", "compression")
    assert vector.objectives[0].derivation.source_name == (
        "average_binary_test_pass_rate"
    )
    # Order: candidate order preserved.
    assert evidence.candidate_order == ("c0",)
    # Tie: explicit behavior recorded.
    assert evidence.tie_behavior is TieBehavior.STABLE_INDEX
    # Selection: content-addressable persisted record.
    content = evidence.record_content()
    assert content["selected_candidate_id"] == "c0"
    # Round-trips through its own content projection.
    assert SelectionEvidence.model_validate(content) == evidence


def test_selection_is_deterministic() -> None:
    candidates = [
        _candidate(
            candidate_id="c0",
            graph_hash=GRAPH_A,
            pass_value=1.0,
            compression_value=2.0,
        ),
        _candidate(
            candidate_id="c1",
            graph_hash=GRAPH_B,
            pass_value=0.7,
            compression_value=1.0,
        ),
    ]
    a = select_official(candidates, objective_specs=SPECS)
    b = select_official(candidates, objective_specs=SPECS)
    assert a == b


def test_selection_tie_keeps_stable_lowest_index() -> None:
    # Identical vectors: both on the front; selection is the lowest index and
    # the tie rule flag is set.
    candidates = [
        _candidate(
            candidate_id="c0",
            graph_hash=GRAPH_A,
            pass_value=0.8,
            compression_value=2.0,
        ),
        _candidate(
            candidate_id="c1",
            graph_hash=GRAPH_B,
            pass_value=0.8,
            compression_value=2.0,
        ),
    ]
    evidence = select_official(candidates, objective_specs=SPECS)
    assert [m.candidate_id for m in evidence.front.members] == ["c0", "c1"]
    assert evidence.selected_candidate_id == "c0"
    assert evidence.selected_index == 0
    assert evidence.selected_by_tie_rule


def test_selection_refuses_missing_aggregate() -> None:
    candidate = SelectionCandidate(
        candidate_id="c0",
        graph_hash=GRAPH_A,
        aggregates={
            "average_binary_test_pass_rate": pass_rate_aggregate(),
            # mean_compression_ratio deliberately absent.
        },
    )
    with pytest.raises(IncompleteEvidenceError, match="missing aggregate"):
        select_official([candidate], objective_specs=SPECS)


def test_selection_refuses_incomplete_evidence() -> None:
    incomplete = incomplete_pass_rate_aggregate()
    # Sanity: the aggregate is genuinely not OK.
    assert incomplete.aggregation_output.status is not AggregationStatus.OK
    candidate = SelectionCandidate(
        candidate_id="c0",
        graph_hash=GRAPH_A,
        aggregates={
            "average_binary_test_pass_rate": incomplete,
            "mean_compression_ratio": compression_aggregate(),
        },
    )
    with pytest.raises(IncompleteEvidenceError, match="not OK"):
        select_official([candidate], objective_specs=SPECS)


def test_selection_never_names_reward_objective() -> None:
    # An ObjectiveSpec that tries to publish under the reserved Reward name is
    # refused when the Objective is built during selection.
    reward_spec = ObjectiveSpec(
        objective_name="reward",
        aggregate_name="average_binary_test_pass_rate",
        direction=Direction.MAXIMIZE,
    )
    candidate = SelectionCandidate(
        candidate_id="c0",
        graph_hash=GRAPH_A,
        aggregates={"average_binary_test_pass_rate": pass_rate_aggregate()},
    )
    with pytest.raises(ValueError, match="reserved Reward name"):
        select_official([candidate], objective_specs=(reward_spec,))


def test_selection_rejects_duplicate_candidate_ids() -> None:
    c = _candidate(
        candidate_id="dup",
        graph_hash=GRAPH_A,
        pass_value=1.0,
        compression_value=2.0,
    )
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        select_official([c, c], objective_specs=SPECS)
