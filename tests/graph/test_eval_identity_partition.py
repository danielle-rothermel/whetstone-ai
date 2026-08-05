"""Eval identity partition validation (deliverable 4).

Proves that a Procedure change alters BOTH graph_hash and eval_config_hash,
while a Sampling/Aggregation-only change alters ONLY eval_config_hash. Also
proves the exact Procedure-identity match between the composite Eval Config
and the Eval Node / Graph Config reference.
"""

from __future__ import annotations

import pytest
from dr_code.eval import EvalConfig
from dr_graph import GraphConfig, graph_hash

from tests.graph.support import (
    build_graph_config,
    eval_config,
    fake_hash,
    procedure_config,
)
from whetstone.graph.eval_config import (
    EvalIdentityMismatchError,
    eval_config_hash,
    sole_eval_node_procedure_hash,
    validate_eval_identity_partition,
)


def _graph_for(ec: EvalConfig) -> GraphConfig:
    return build_graph_config(
        provider_call_config_hash=fake_hash("a"),
        evaluation_procedure_config_hash=(ec.evaluation_procedure_config_hash),
    )


def test_matching_procedure_identity_validates() -> None:
    ec = eval_config()
    graph = _graph_for(ec)
    validate_eval_identity_partition(graph, ec)  # no raise
    assert sole_eval_node_procedure_hash(graph) == (
        ec.evaluation_procedure_config_hash
    )


def test_mismatched_procedure_identity_rejected() -> None:
    ec = eval_config()
    # Build a graph whose Eval Node references a different Procedure hash.
    graph = build_graph_config(
        provider_call_config_hash=fake_hash("a"),
        evaluation_procedure_config_hash=fake_hash("f"),
    )
    with pytest.raises(EvalIdentityMismatchError):
        validate_eval_identity_partition(graph, ec)


def test_procedure_change_alters_both_hashes() -> None:
    base_proc = procedure_config(zero_denominator="not_applicable")
    changed_proc = procedure_config(zero_denominator="error")
    base_ec = eval_config(procedure=base_proc)
    changed_ec = eval_config(procedure=changed_proc)

    base_graph = _graph_for(base_ec)
    changed_graph = _graph_for(changed_ec)

    # Procedure change alters graph_hash ...
    assert graph_hash(base_graph) != graph_hash(changed_graph)
    # ... and eval_config_hash.
    assert eval_config_hash(base_ec) != eval_config_hash(changed_ec)


def test_sampling_or_aggregation_change_alters_only_eval_config_hash() -> None:
    proc = procedure_config()
    base_ec = eval_config(procedure=proc, reduction="mean")
    changed_ec = eval_config(procedure=proc, reduction="sum")

    base_graph = _graph_for(base_ec)
    changed_graph = _graph_for(changed_ec)

    # Aggregation-only change leaves the Procedure identity (hence
    # graph_hash) untouched ...
    assert (
        base_ec.evaluation_procedure_config_hash
        == changed_ec.evaluation_procedure_config_hash
    )
    assert graph_hash(base_graph) == graph_hash(changed_graph)
    # ... but changes eval_config_hash.
    assert eval_config_hash(base_ec) != eval_config_hash(changed_ec)


def test_eval_config_hash_is_dr_code_composite_identity() -> None:
    ec = eval_config()
    assert eval_config_hash(ec) == ec.config_identity_hash
    assert len(eval_config_hash(ec)) == 64
