"""Persisted MIPROv2 control identity.

Schema 8 added ``num_seeds``: the repeats every in-search evaluation of the
run pays for. A control that evaluates each task three times is a materially
different control from one that evaluates it once, so the count is part of
the control's identity and these hashes moved when it was added.
"""

from __future__ import annotations

from dr_store.sync import open_sqlite

from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.miprov2.control import (
    MIPROV2_CONTROL_SCHEMA,
    MIPROV2_CONTROL_SCHEMA_VERSION,
    Miprov2DemoMode,
)
from whetstone.testing.toy.miprov2 import build_toy_miprov2_control


def test_control_schema_version_is_eight() -> None:
    assert MIPROV2_CONTROL_SCHEMA == "whetstone.miprov2_optimizer_config"
    assert MIPROV2_CONTROL_SCHEMA_VERSION == 8


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
        "3105b9de6ef96b54856a0778fa6eb5cc"
        "cd558260dc400407a3caff9a9dd54b16"
    ),
    Miprov2DemoMode.ZEROSHOT: (
        "f1a2dff75b477e4f42b7c3dae248ebf8"
        "d1d023f74639958aa7e093149c27e528"
    ),
    Miprov2DemoMode.GROUND_ONLY: (
        "19c6140b7dacff73e775bc8897552d95"
        "0f207f84c39b8360970536ee24eeccdd"
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


def test_identity_payload_pins_the_search_repeat_count(tmp_path) -> None:
    """The repeats an in-search evaluation pays for are part of identity."""
    from whetstone.testing.toy.experiment import build_toy_experiment

    with open_sqlite(str(tmp_path / "repeats.sqlite")) as store:
        runtime_config = ReferenceEvalRuntimeConfig()
        once = build_toy_miprov2_control(
            engine=runtime_config.build_engine(
                store, experiment=build_toy_experiment(num_seeds=1)
            )
        )
        thrice = build_toy_miprov2_control(
            engine=runtime_config.build_engine(
                store, experiment=build_toy_experiment(num_seeds=3)
            )
        )

    assert once.identity_payload()["num_seeds"] == 1
    assert thrice.identity_payload()["num_seeds"] == 3
    # A different repeat count is a different control, not the same one.
    assert once.identity_hash() != thrice.identity_hash()
