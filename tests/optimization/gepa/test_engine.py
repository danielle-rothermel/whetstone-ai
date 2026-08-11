from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pytest
from gepa import EvaluationBatch

from tests.optimization.support import (
    FULL_A,
    FULL_B,
    FULL_C,
    FULL_D,
    eval_config,
)
from whetstone.core.identity import (
    IdentityRef,
    typed_ref_for_record,
)
from whetstone.experiment.binding import eval_config_reference
from whetstone.optimization.gepa.control import configure_gepa
from whetstone.optimization.gepa.engine import run_gepa_engine
from whetstone.optimization.gepa.source import (
    GEPA_SOURCE_MANIFEST_HASH,
    verify_installed_gepa_source,
)
from whetstone.optimization.gepa.upstream_adapter import (
    GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH,
)
from whetstone.optimization.proposal.proposer import ProposerConfig


def _identity(index: int) -> str:
    return f"{index:064x}"


class _ScriptedAdapter:
    def __init__(self, control) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.proposal_calls = 0
        self.effect_context = _EffectContext(
            control_identity_hash=control.identity_hash(),
            source_manifest_identity_hash=GEPA_SOURCE_MANIFEST_HASH,
            adapter_identity_hash=GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH,
        )
        self.evaluation_authority = _EvaluationAuthority(
            evaluation_config_hash=control.metric.config_hash,
            reward_policy_identity_hash=control.reward_policy_hash,
            provider_route_identity_hash=control.task_model_identity_hash,
            execution_policy_identity_hash=(
                control.evaluation_execution_policy_hash
            ),
            failure_score=control.failure_score,
            add_format_failure_as_feedback=(
                control.add_format_failure_as_feedback
            ),
            warn_on_score_mismatch=control.warn_on_score_mismatch,
            selection_seed=control.seed,
        )
        self.proposal_authority = _ProposalAuthority(
            prompt_binding_identity_hash=(
                control.prompt_binding_identity_hash
            ),
            proposer_config=control.reflection_model,
            execution_policy_identity_hash=(
                control.proposal_execution_policy_hash
            ),
            prompt_adapter_identity_hash=(
                control.proposal_prompt_adapter_identity_hash
            ),
            durability_policy_identity_hash=(
                control.proposal_durability_policy_identity_hash
            ),
        )
        self.prompt_format_identity_hash = control.prompt_format_identity_hash

    def reset_effect_ordinal(self) -> None:
        self.events.clear()
        self.proposal_calls = 0

    def evaluate(self, batch, candidate, capture_traces=False):
        self.events.append(
            (
                "evaluate",
                tuple(item.id for item in batch),
                tuple(candidate.items()),
                capture_traces,
            )
        )
        levels = {
            component: int(text.rsplit("-", 1)[-1])
            for component, text in candidate.items()
        }
        if levels == {"component_a": 0, "component_b": 0}:
            scores = [0.5 for _ in batch]
        else:
            scores = [
                float(
                    levels[
                        ("component_a" if item.id % 2 == 0 else "component_b")
                    ]
                )
                for item in batch
            ]
        outputs = [
            {"id": item.id, "candidate": dict(candidate)} for item in batch
        ]
        trajectories = (
            [
                {
                    "id": item.id,
                    "candidate": dict(candidate),
                    "score": score,
                }
                for item, score in zip(batch, scores, strict=True)
            ]
            if capture_traces
            else None
        )
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self, candidate, eval_batch, components_to_update
    ):
        assert eval_batch.trajectories is not None
        return {
            component: [
                {
                    "Inputs": {"id": trajectory["id"]},
                    "Generated Outputs": str(candidate[component]),
                    "Feedback": str(trajectory["score"]),
                }
                for trajectory in eval_batch.trajectories
            ]
            for component in components_to_update
        }

    def propose_new_texts(
        self, candidate, reflective_dataset, components_to_update
    ):
        del reflective_dataset
        self.proposal_calls += 1
        proposal = {
            component: f"{component}-1" for component in components_to_update
        }
        self.events.append(
            (
                "propose",
                tuple(candidate.items()),
                tuple(components_to_update),
                tuple(proposal.items()),
            )
        )
        return proposal


@dataclass(frozen=True)
class _EffectContext:
    control_identity_hash: str
    source_manifest_identity_hash: str
    adapter_identity_hash: str


@dataclass(frozen=True)
class _EvaluationAuthority:
    evaluation_config_hash: str
    reward_policy_identity_hash: str
    provider_route_identity_hash: str
    execution_policy_identity_hash: str
    failure_score: float
    add_format_failure_as_feedback: bool
    warn_on_score_mismatch: bool
    selection_seed: int


@dataclass(frozen=True)
class _ProposalAuthority:
    prompt_binding_identity_hash: str
    proposer_config: ProposerConfig
    execution_policy_identity_hash: str
    prompt_adapter_identity_hash: str
    durability_policy_identity_hash: str


@dataclass(frozen=True)
class _Data:
    id: int
    data_id: str


def _control(**overrides):
    values: dict[str, Any] = {
        "reflection_model": ProposerConfig(
            provider_call_config=IdentityRef(
                record_ref=typed_ref_for_record(
                    "dr_providers.provider_call_config",
                    {"provider_call_config_ref": "provider://reflection"},
                ),
                record_hash=FULL_A,
            ),
        ),
        "metric": eval_config_reference(eval_config()),
        "reward_policy_hash": FULL_B,
        "evaluation_execution_policy_hash": FULL_C,
        "proposal_execution_policy_hash": FULL_A,
        "proposal_prompt_adapter_identity_hash": FULL_B,
        "proposal_durability_policy_identity_hash": FULL_D,
        "task_model_identity_hash": FULL_D,
        "prompt_format_identity_hash": FULL_A,
        "prompt_binding_identity_hash": FULL_B,
        "trainset_task_hashes": tuple(_identity(i) for i in range(1, 9)),
        "valset_task_hashes": tuple(_identity(i) for i in range(101, 109)),
        "component_names": ("component_a", "component_b"),
        "num_predictors": 2,
        "max_metric_calls": 160,
        "seed": 12,
    }
    values.update(overrides)
    return configure_gepa(**values)


def test_frozen_source_guard_matches_installed_gepa() -> None:
    assert verify_installed_gepa_source() == GEPA_SOURCE_MANIFEST_HASH


def test_upstream_merge_is_accepted_without_a_merge_proposal_call() -> None:
    control = _control()
    adapter = _ScriptedAdapter(control)

    result = run_gepa_engine(
        control=control,
        seed_candidate={
            "component_a": "component_a-0",
            "component_b": "component_b-0",
        },
        trainset=[
            _Data(id=index, data_id=identity)
            for index, identity in zip(
                range(8),
                control.trainset_task_hashes,
                strict=True,
            )
        ],
        valset=[
            _Data(id=index, data_id=identity)
            for index, identity in zip(
                range(100, 108),
                control.valset_task_hashes,
                strict=True,
            )
        ],
        adapter=adapter,
    )

    assert adapter.proposal_calls == 2
    assert result.parents == ((None,), (0,), (0,), (1, 2))
    assert result.candidates[-1] == {
        "component_a": "component_a-1",
        "component_b": "component_b-1",
    }
    assert result.best_idx == 3
    assert set(result.per_val_instance_best_candidates) == set(
        control.valset_task_hashes
    )
    assert result.control_identity_hash == control.identity_hash()
    assert result.source_manifest_hash == GEPA_SOURCE_MANIFEST_HASH


def test_engine_rejects_adapter_and_order_identity_drift() -> None:
    control = _control()
    adapter = _ScriptedAdapter(control)
    trainset = [
        _Data(id=index, data_id=identity)
        for index, identity in zip(
            range(8),
            control.trainset_task_hashes,
            strict=True,
        )
    ]
    valset = [
        _Data(id=index, data_id=identity)
        for index, identity in zip(
            range(100, 108),
            control.valset_task_hashes,
            strict=True,
        )
    ]

    with pytest.raises(ValueError, match="trainset order/identity"):
        run_gepa_engine(
            control=control,
            seed_candidate={
                "component_a": "component_a-0",
                "component_b": "component_b-0",
            },
            trainset=list(reversed(trainset)),
            valset=valset,
            adapter=adapter,
        )
    with pytest.raises(ValueError, match="seed_candidate component order"):
        run_gepa_engine(
            control=control,
            seed_candidate={
                "component_b": "component_b-0",
                "component_a": "component_a-0",
            },
            trainset=trainset,
            valset=valset,
            adapter=adapter,
        )
    adapter.effect_context = _EffectContext(
        control_identity_hash=FULL_A,
        source_manifest_identity_hash=GEPA_SOURCE_MANIFEST_HASH,
        adapter_identity_hash=GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH,
    )
    with pytest.raises(ValueError, match="conflicts with GepaControl"):
        run_gepa_engine(
            control=control,
            seed_candidate={
                "component_a": "component_a-0",
                "component_b": "component_b-0",
            },
            trainset=trainset,
            valset=valset,
            adapter=adapter,
        )


@pytest.mark.parametrize(
    ("target", "field", "drifted"),
    [
        ("context", "adapter_identity_hash", _identity(900)),
        (
            "evaluation",
            "evaluation_config_hash",
            _identity(901),
        ),
        ("evaluation", "reward_policy_identity_hash", _identity(902)),
        ("evaluation", "provider_route_identity_hash", _identity(903)),
        ("evaluation", "execution_policy_identity_hash", _identity(904)),
        ("evaluation", "failure_score", -1.0),
        ("evaluation", "add_format_failure_as_feedback", True),
        ("evaluation", "warn_on_score_mismatch", False),
        ("evaluation", "selection_seed", 13),
        (
            "proposal",
            "proposer_config",
            ProposerConfig(
                provider_call_config=IdentityRef(
                    record_ref=typed_ref_for_record(
                        "dr_providers.provider_call_config",
                        {"provider_call_config_ref": "provider://drifted"},
                    ),
                    record_hash=FULL_D,
                ),
            ),
        ),
        ("proposal", "prompt_binding_identity_hash", _identity(905)),
        ("proposal", "execution_policy_identity_hash", _identity(906)),
        ("proposal", "prompt_adapter_identity_hash", _identity(907)),
        ("proposal", "durability_policy_identity_hash", _identity(908)),
        ("adapter", "prompt_format_identity_hash", _identity(909)),
    ],
)
def test_engine_rejects_each_control_owned_adapter_authority_drift(
    target, field, drifted
) -> None:
    control = _control()
    adapter = _ScriptedAdapter(control)
    if target == "context":
        adapter.effect_context = replace(
            adapter.effect_context,
            **{field: drifted},
        )
    elif target == "evaluation":
        adapter.evaluation_authority = replace(
            adapter.evaluation_authority,
            **{field: drifted},
        )
    elif target == "proposal":
        adapter.proposal_authority = replace(
            adapter.proposal_authority,
            **{field: drifted},
        )
    else:
        setattr(adapter, field, drifted)
    trainset = [
        _Data(id=index, data_id=identity)
        for index, identity in zip(
            range(8),
            control.trainset_task_hashes,
            strict=True,
        )
    ]
    valset = [
        _Data(id=index, data_id=identity)
        for index, identity in zip(
            range(100, 108),
            control.valset_task_hashes,
            strict=True,
        )
    ]

    with pytest.raises(ValueError, match="conflicts with"):
        run_gepa_engine(
            control=control,
            seed_candidate={
                "component_a": "component_a-0",
                "component_b": "component_b-0",
            },
            trainset=trainset,
            valset=valset,
            adapter=adapter,
        )


def test_track_stats_does_not_change_engine_execution_or_best_choice() -> None:
    results = []
    traces = []
    for track_stats in (False, True):
        control = _control(track_stats=track_stats)
        adapter = _ScriptedAdapter(control)
        result = run_gepa_engine(
            control=control,
            seed_candidate={
                "component_a": "component_a-0",
                "component_b": "component_b-0",
            },
            trainset=[
                _Data(id=index, data_id=identity)
                for index, identity in zip(
                    range(8),
                    control.trainset_task_hashes,
                    strict=True,
                )
            ],
            valset=[
                _Data(id=index, data_id=identity)
                for index, identity in zip(
                    range(100, 108),
                    control.valset_task_hashes,
                    strict=True,
                )
            ],
            adapter=adapter,
        )
        results.append(result)
        traces.append(adapter.events)

    assert traces[0] == traces[1]
    assert results[0].candidates == results[1].candidates
    assert results[0].parents == results[1].parents
    assert results[0].val_aggregate_scores == results[1].val_aggregate_scores
    assert results[0].best_idx == results[1].best_idx
