from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from tests.optimization.support import (
    FULL_A,
    FULL_B,
    FULL_C,
    FULL_D,
    eval_config,
)
from whetstone.optimization import (
    ProposerConfig,
    eval_config_reference,
    typed_ref_for_record,
)
from whetstone.optimization.gepa import (
    GEPA_ADAPTER_KEY,
    GepaOptimizer,
    project_gepa_terminal,
)
from whetstone.optimization.gepa_control import configure_gepa
from whetstone.optimization.gepa_engine import GepaDetailedResult
from whetstone.optimization.gepa_source import GEPA_SOURCE_MANIFEST_HASH


@dataclass(frozen=True)
class _Data:
    data_id: str


@dataclass(frozen=True)
class _Context:
    control_identity_hash: str
    source_manifest_identity_hash: str


class _Adapter:
    def __init__(self, control_identity_hash: str) -> None:
        self.effect_context = _Context(
            control_identity_hash=control_identity_hash,
            source_manifest_identity_hash=GEPA_SOURCE_MANIFEST_HASH,
        )
        self.reset_count = 0

    def reset_effect_ordinal(self) -> None:
        self.reset_count += 1

    def evaluate(self, batch, candidate, capture_traces=False):
        raise AssertionError("engine call is replaced in this façade test")


class _Factory:
    def __init__(self, adapter: _Adapter) -> None:
        self.adapter = adapter
        self.controls = []
        self.persisted = []

    def create(self, *, control):
        self.controls.append(control)
        return self.adapter

    def persist_result(self, *, control, adapter, detailed_result):
        self.persisted.append((control, adapter, detailed_result))
        return typed_ref_for_record(
            "whetstone.gepa.canonical_run",
            detailed_result.model_dump(mode="json"),
        )


def _control(**overrides):
    values: dict[str, Any] = {
        "reflection_model": ProposerConfig(
            provider_call_config_ref="provider://reflection",
            provider_call_config_hash=FULL_A,
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
        "trainset_task_identities": (FULL_A,),
        "valset_task_identities": None,
        "component_names": ("prompt",),
        "num_predictors": 1,
        "max_metric_calls": 1,
    }
    values.update(overrides)
    return configure_gepa(**values)


def _detailed(control) -> GepaDetailedResult:
    return GepaDetailedResult(
        candidates=({"prompt": "seed"}, {"prompt": "best"}),
        parents=((None,), (0,)),
        val_aggregate_scores=(0.0, 1.0),
        val_subscores=({FULL_A: 0.0}, {FULL_A: 1.0}),
        per_val_instance_best_candidates={FULL_A: (1,)},
        discovery_eval_counts=(0, 1),
        total_metric_calls=1,
        num_full_val_evals=2,
        seed=control.seed,
        best_idx=1,
        control_identity_hash=control.identity_hash(),
    )


def test_public_gepa_name_is_canonical() -> None:
    assert GEPA_ADAPTER_KEY == "gepa"


def test_optimizer_uses_fresh_factory_adapter_and_direct_engine(
    monkeypatch,
) -> None:
    control = _control()
    adapter = _Adapter(control.identity_hash())
    factory = _Factory(adapter)
    optimizer = GepaOptimizer(control=control, adapter_factory=factory)
    detailed = _detailed(control)
    observed = {}

    def fake_run_gepa_engine(**kwargs):
        observed.update(kwargs)
        return detailed

    monkeypatch.setattr(
        "whetstone.optimization.gepa.run_gepa_engine",
        fake_run_gepa_engine,
    )
    trainset = [_Data(data_id=FULL_A)]

    result = optimizer.run_detailed(
        seed_candidate={"prompt": "answer carefully"},
        trainset=trainset,
    )

    assert result.detailed_result == detailed
    assert factory.controls == [control]
    assert factory.persisted == [(control, adapter, detailed)]
    assert result.artifact_ref.schema_name == "whetstone.gepa.canonical_run"
    assert observed == {
        "control": control,
        "seed_candidate": {"prompt": "answer carefully"},
        "trainset": trainset,
        "valset": None,
        "adapter": adapter,
    }


def test_terminal_projection_honors_track_stats_without_changing_best() -> (
    None
):
    hidden_control = _control(track_stats=False)
    tracked_control = _control(track_stats=True)
    hidden_detail = _detailed(hidden_control)
    tracked_detail = _detailed(tracked_control)

    hidden = project_gepa_terminal(
        control=hidden_control,
        detailed_result=hidden_detail,
        artifact_ref=typed_ref_for_record("gepa", {"mode": "hidden"}),
    )
    tracked = project_gepa_terminal(
        control=tracked_control,
        detailed_result=tracked_detail,
        artifact_ref=typed_ref_for_record("gepa", {"mode": "tracked"}),
    )

    assert (
        hidden.best_candidate == tracked.best_candidate == {"prompt": "best"}
    )
    assert hidden.detailed_results is None
    assert tracked.detailed_results == tracked_detail


def test_terminal_is_not_exposed_until_post_engine_persistence(
    monkeypatch,
) -> None:
    control = _control()
    adapter = _Adapter(control.identity_hash())

    class CrashOnceFactory(_Factory):
        def __init__(self, bound_adapter):
            super().__init__(bound_adapter)
            self.persist_attempts = 0

        def persist_result(self, *, control, adapter, detailed_result):
            self.persist_attempts += 1
            if self.persist_attempts == 1:
                raise RuntimeError("crash after engine")
            return super().persist_result(
                control=control,
                adapter=adapter,
                detailed_result=detailed_result,
            )

    factory = CrashOnceFactory(adapter)
    optimizer = GepaOptimizer(control=control, adapter_factory=factory)
    detailed = _detailed(control)
    engine_calls = 0

    def fake_run_gepa_engine(**kwargs):
        nonlocal engine_calls
        engine_calls += 1
        return detailed

    monkeypatch.setattr(
        "whetstone.optimization.gepa.run_gepa_engine",
        fake_run_gepa_engine,
    )
    trainset = [_Data(data_id=FULL_A)]

    with pytest.raises(RuntimeError, match="crash after engine"):
        optimizer.run(
            seed_candidate={"prompt": "seed"},
            trainset=trainset,
        )

    terminal = optimizer.run(
        seed_candidate={"prompt": "seed"},
        trainset=trainset,
    )

    assert engine_calls == 2
    assert factory.persist_attempts == 2
    assert terminal.best_candidate == {"prompt": "best"}
    assert terminal.detailed_results is None


def test_legacy_fake_gepa_symbols_are_gone() -> None:
    import whetstone.optimization.gepa as module

    assert not hasattr(module, "GEPA_VARIANT")
    assert not hasattr(module, "ACCEPTANCE_POLICY")
    assert not hasattr(module, "strict_pareto_accepts")
    assert not hasattr(module, "GepaAdapter")
