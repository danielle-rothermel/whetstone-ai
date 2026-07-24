"""PR4 contains only generic optimization core modules."""

from pathlib import Path


def test_optimization_package_has_only_core_modules() -> None:
    root = Path("src/whetstone/optimization")
    assert {path.name for path in root.glob("*.py")} == {
        "__init__.py",
        "adapters.py",
        "harness.py",
        "identity.py",
        "mutation.py",
        "proposer.py",
        "reward.py",
        "schema.py",
        "tool_eval.py",
        "tool_store.py",
        "tools.py",
    }


def test_harness_has_no_concrete_adapter_or_runner_dependencies() -> None:
    text = Path("src/whetstone/optimization/harness.py").read_text()
    forbidden = (
        "Copro",
        "COPRO",
        "Miprov2",
        "MIPRO",
        "Gepa",
        "GEPA",
        "Codex",
        "whetstone." + "runner",
        "whetstone." + "envs",
    )
    assert all(symbol not in text for symbol in forbidden)


def test_core_has_no_runner_or_environment_imports() -> None:
    text = "\n".join(
        path.read_text()
        for path in Path("src/whetstone/optimization").glob("*.py")
    )
    assert "whetstone." + "runner" not in text
    assert "whetstone." + "envs" not in text
