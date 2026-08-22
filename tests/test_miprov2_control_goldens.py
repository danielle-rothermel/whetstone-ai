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


_TOY_CONTROL_HASHES = {
    Miprov2DemoMode.FEWSHOT: (
        "9ded7dce4290fa46e69cbff8ed492b4f"
        "585d047e09b18f8758681c9bfd1d37a1"
    ),
    Miprov2DemoMode.ZEROSHOT: (
        "3ee6639103d4f7e6f94447ecfd35b0b2"
        "b32b129b7fbf0b13c8fe1b2a2907a51c"
    ),
    Miprov2DemoMode.GROUND_ONLY: (
        "68e65ed997b547a22b351721307a1dda"
        "e11fbe56af9ea266146f9ea27255aca3"
    ),
}


def test_the_three_demo_modes_mint_distinct_control_hashes(tmp_path) -> None:
    with open_sqlite(str(tmp_path / "modes.sqlite")) as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(store)
        hashes = {
            mode: build_toy_miprov2_control(
                engine=engine, demo_mode=mode
            ).identity_hash()
            for mode in Miprov2DemoMode
        }

    assert hashes == _TOY_CONTROL_HASHES
    assert len(set(hashes.values())) == len(Miprov2DemoMode)
