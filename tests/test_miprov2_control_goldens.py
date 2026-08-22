"""Persisted MIPROv2 control identity after demo_mode replaced zeroshot_opt."""

from __future__ import annotations

from dr_store.sync import open_sqlite

from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.miprov2.control import (
    MIPROV2_CONTROL_SCHEMA,
    MIPROV2_CONTROL_SCHEMA_VERSION,
    Miprov2DemoMode,
)
from whetstone.testing.toy.miprov2 import build_toy_miprov2_control


def test_control_schema_version_is_seven() -> None:
    assert MIPROV2_CONTROL_SCHEMA == "whetstone.miprov2_optimizer_config"
    assert MIPROV2_CONTROL_SCHEMA_VERSION == 7


def test_identity_payload_stores_demo_mode_not_zeroshot_opt(tmp_path) -> None:
    with open_sqlite(str(tmp_path / "ctrl.sqlite")) as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(store)
        control = build_toy_miprov2_control(engine=engine)

    payload = control.identity_payload()
    assert payload["demo_mode"] == Miprov2DemoMode.FEWSHOT.value
    assert "zeroshot_opt" not in payload
    assert control.zeroshot_opt is False


def test_the_three_demo_modes_mint_distinct_control_hashes(tmp_path) -> None:
    hashes: dict[Miprov2DemoMode, str] = {}
    with open_sqlite(str(tmp_path / "modes.sqlite")) as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(store)
        for mode in Miprov2DemoMode:
            hashes[mode] = build_toy_miprov2_control(
                engine=engine, demo_mode=mode
            ).identity_hash()

    assert len(set(hashes.values())) == len(Miprov2DemoMode)
