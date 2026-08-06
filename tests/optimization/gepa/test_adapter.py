from __future__ import annotations

from dataclasses import dataclass

import pytest

from tests.optimization.gepa.support import (
    gepa_control,
    make_gepa_detailed_result,
)
from tests.optimization.support import FULL_A
from whetstone.core.identity import typed_ref_for_record
from whetstone.optimization.gepa.adapter import (
    GEPA_ADAPTER_KEY,
    GepaOptimizer,
    project_gepa_terminal,
)
from whetstone.optimization.gepa.source import GEPA_SOURCE_MANIFEST_HASH


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


def test_public_gepa_name_is_canonical() -> None:
    assert GEPA_ADAPTER_KEY == "gepa"


def test_optimizer_uses_fresh_factory_adapter_and_direct_engine(
    monkeypatch,
) -> None:
    control = gepa_control()
    adapter = _Adapter(control.identity_hash())
    factory = _Factory(adapter)
    optimizer = GepaOptimizer(control=control, adapter_factory=factory)
    detailed = make_gepa_detailed_result(control)
    observed = {}

    def fake_run_gepa_engine(**kwargs):
        observed.update(kwargs)
        return detailed

    monkeypatch.setattr(
        "whetstone.optimization.gepa.adapter.run_gepa_engine",
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
    hidden_control = gepa_control(track_stats=False)
    tracked_control = gepa_control(track_stats=True)
    hidden_detail = make_gepa_detailed_result(hidden_control)
    tracked_detail = make_gepa_detailed_result(tracked_control)

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
    control = gepa_control()
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
    detailed = make_gepa_detailed_result(control)
    engine_calls = 0

    def fake_run_gepa_engine(**kwargs):
        nonlocal engine_calls
        engine_calls += 1
        return detailed

    monkeypatch.setattr(
        "whetstone.optimization.gepa.adapter.run_gepa_engine",
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
