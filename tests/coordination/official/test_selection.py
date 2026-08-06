from __future__ import annotations

import pytest
from dr_code.eval import AggregationStatus

from tests.experiment.support import (
    GRAPH_A,
    GRAPH_B,
    SELECTION_QUALITY_AGGREGATE_NAME,
    compression_aggregate,
    incomplete_quality_aggregate,
    quality_aggregate,
)
from whetstone.coordination.official.selection import (
    IncompleteEvidenceError,
    ObjectiveSpec,
    SelectionCandidate,
    SelectionEvidence,
    select_official,
)
from whetstone.experiment.objectives import (
    Direction,
    TieBehavior,
)

SPECS = (
    ObjectiveSpec(
        objective_name="quality",
        aggregate_name=SELECTION_QUALITY_AGGREGATE_NAME,
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
    quality_value: float,
    compression_value: float,
) -> SelectionCandidate:
    return SelectionCandidate(
        candidate_id=candidate_id,
        graph_hash=graph_hash,
        aggregates={
            SELECTION_QUALITY_AGGREGATE_NAME: quality_aggregate(
                graph_hash=graph_hash, value=quality_value
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
            quality_value=1.0,
            compression_value=2.0,
        ),
        _candidate(
            candidate_id="c1",
            graph_hash=GRAPH_B,
            quality_value=0.5,
            compression_value=3.0,
        ),
    ]
    evidence = select_official(candidates, objective_specs=SPECS)
    assert evidence.selected_candidate_id == "c0"
    assert [m.candidate_id for m in evidence.front.members] == ["c0"]
    assert not evidence.selected_by_tie_rule


def test_selection_persists_derivation_order_tie_selection() -> None:
    candidates = [
        _candidate(
            candidate_id="c0",
            graph_hash=GRAPH_A,
            quality_value=1.0,
            compression_value=2.0,
        ),
    ]
    evidence = select_official(candidates, objective_specs=SPECS)
    assert evidence.objective_specs == SPECS
    vector = evidence.candidate_vectors[0]
    assert vector.names == ("quality", "compression")
    assert (
        vector.objectives[0].derivation.source_name
        == SELECTION_QUALITY_AGGREGATE_NAME
    )
    assert evidence.candidate_order == ("c0",)
    assert evidence.tie_behavior is TieBehavior.STABLE_INDEX
    content = evidence.record_content()
    assert content["selected_candidate_id"] == "c0"
    assert SelectionEvidence.model_validate(content) == evidence


def test_selection_is_deterministic() -> None:
    candidates = [
        _candidate(
            candidate_id="c0",
            graph_hash=GRAPH_A,
            quality_value=1.0,
            compression_value=2.0,
        ),
        _candidate(
            candidate_id="c1",
            graph_hash=GRAPH_B,
            quality_value=0.7,
            compression_value=1.0,
        ),
    ]
    a = select_official(candidates, objective_specs=SPECS)
    b = select_official(candidates, objective_specs=SPECS)
    assert a == b


def test_selection_tie_keeps_stable_lowest_index() -> None:
    candidates = [
        _candidate(
            candidate_id="c0",
            graph_hash=GRAPH_A,
            quality_value=0.8,
            compression_value=2.0,
        ),
        _candidate(
            candidate_id="c1",
            graph_hash=GRAPH_B,
            quality_value=0.8,
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
            SELECTION_QUALITY_AGGREGATE_NAME: quality_aggregate(),
        },
    )
    with pytest.raises(IncompleteEvidenceError, match="missing aggregate"):
        select_official([candidate], objective_specs=SPECS)


def test_selection_refuses_incomplete_evidence() -> None:
    incomplete = incomplete_quality_aggregate()
    assert incomplete.aggregation_output.status is not AggregationStatus.OK
    candidate = SelectionCandidate(
        candidate_id="c0",
        graph_hash=GRAPH_A,
        aggregates={
            SELECTION_QUALITY_AGGREGATE_NAME: incomplete,
            "mean_compression_ratio": compression_aggregate(),
        },
    )
    with pytest.raises(IncompleteEvidenceError, match="not OK"):
        select_official([candidate], objective_specs=SPECS)


def test_selection_never_names_reward_objective() -> None:
    reward_spec = ObjectiveSpec(
        objective_name="reward",
        aggregate_name=SELECTION_QUALITY_AGGREGATE_NAME,
        direction=Direction.MAXIMIZE,
    )
    candidate = SelectionCandidate(
        candidate_id="c0",
        graph_hash=GRAPH_A,
        aggregates={SELECTION_QUALITY_AGGREGATE_NAME: quality_aggregate()},
    )
    with pytest.raises(ValueError, match="reserved Reward name"):
        select_official([candidate], objective_specs=(reward_spec,))


def test_selection_requires_one_shared_eval_config_hash() -> None:
    baseline = _candidate(
        candidate_id="c0",
        graph_hash=GRAPH_A,
        quality_value=1.0,
        compression_value=2.0,
    )
    other_config = SelectionCandidate(
        candidate_id="c1",
        graph_hash=GRAPH_B,
        aggregates={
            SELECTION_QUALITY_AGGREGATE_NAME: quality_aggregate(
                graph_hash=GRAPH_B, value=2.0, tasks=3
            ),
            "mean_compression_ratio": compression_aggregate(
                graph_hash=GRAPH_B, value=1.0, tasks=3
            ),
        },
    )
    quality = other_config.aggregates[SELECTION_QUALITY_AGGREGATE_NAME]
    baseline_quality = baseline.aggregates[SELECTION_QUALITY_AGGREGATE_NAME]
    assert quality.eval_config_hash != baseline_quality.eval_config_hash
    with pytest.raises(ValueError, match="share one eval_config_hash"):
        select_official([baseline, other_config], objective_specs=SPECS)


def test_selection_accepts_one_shared_eval_config_hash() -> None:
    candidates = [
        _candidate(
            candidate_id="c0",
            graph_hash=GRAPH_A,
            quality_value=1.0,
            compression_value=2.0,
        ),
        _candidate(
            candidate_id="c1",
            graph_hash=GRAPH_B,
            quality_value=2.0,
            compression_value=1.0,
        ),
    ]
    hashes = {
        aggregate.eval_config_hash
        for candidate in candidates
        for aggregate in candidate.aggregates.values()
    }
    assert len(hashes) == 1
    evidence = select_official(candidates, objective_specs=SPECS)
    assert evidence.selected_candidate_id in {"c0", "c1"}


def test_selection_rejects_duplicate_candidate_ids() -> None:
    c = _candidate(
        candidate_id="dup",
        graph_hash=GRAPH_A,
        quality_value=1.0,
        compression_value=2.0,
    )
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        select_official([c, c], objective_specs=SPECS)
