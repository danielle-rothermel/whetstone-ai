from __future__ import annotations


def test_runtime_bootstrap_importable_without_dbos() -> None:
    import importlib

    module = importlib.import_module("whetstone.coordination.runtime_bootstrap")
    assert hasattr(module, "register_runtime")
    assert hasattr(module, "RegisteredRuntime")

    registry = importlib.import_module(
        "whetstone.coordination.run_controller_registry"
    )
    assert hasattr(registry, "register_run_controller")
