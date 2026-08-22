from __future__ import annotations


def test_runtime_bootstrap_importable_without_dbos() -> None:
    import importlib

    module = importlib.import_module("whetstone.coordination.runtime_bootstrap")
    assert hasattr(module, "build_runtime")
    assert hasattr(module, "RegisteredRuntime")
    assert not hasattr(module, "register_runtime")
