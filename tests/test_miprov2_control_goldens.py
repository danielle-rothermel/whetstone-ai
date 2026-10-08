"""Persisted MIPROv2 control identity.

Schema 8 added ``num_seeds``: the repeats every in-search evaluation of the
run pays for. A control that evaluates each task three times is a materially
different control from one that evaluates it once, so the count is part of
the control's identity and these hashes moved when it was added.

The payload also carries ``task_model_identity_hash``: the identity hash of
the toy experiment's dr-providers call config. A dr-providers release that
changes definition or config identity therefore moves these hashes too.
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
        "952e342b9c69bf3a2ae474f9fdc2a689"
        "392431ce3632e8686a907172d93b95c4"
    ),
    Miprov2DemoMode.ZEROSHOT: (
        "b3dd61de2d8d66b4a116fd94b1af8e9d"
        "f11a904e3bdb75f7b4d3e99dada07f5f"
    ),
    Miprov2DemoMode.GROUND_ONLY: (
        "ea29769e3276876c642cd8d20268cc32"
        "f42816306a2bcdbd33542e86a96c51b8"
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
