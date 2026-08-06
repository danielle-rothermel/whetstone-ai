from pathlib import Path

FUNCTIONAL_PACKAGES = {
    "coordination",
    "core",
    "envs",
    "evaluation",
    "execution",
    "experiment",
    "optimization",
    "provider",
}


def _package_directories(root: Path) -> set[str]:
    return {
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }


def test_source_and_test_trees_share_functional_packages() -> None:
    assert _package_directories(Path("src/whetstone")) == FUNCTIONAL_PACKAGES
    assert _package_directories(Path("tests")) == FUNCTIONAL_PACKAGES


def test_optimization_root_contains_only_shared_orchestration() -> None:
    root = Path("src/whetstone/optimization")
    assert {path.name for path in root.glob("*.py")} == {
        "__init__.py",
        "adapters.py",
        "contracts.py",
        "harness.py",
        "run_store.py",
    }
    assert _package_directories(root) == {
        "copro",
        "gepa",
        "miprov2",
        "proposal",
        "tools",
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


def test_shared_optimization_modules_avoid_execution_policy() -> None:
    text = "\n".join(
        path.read_text()
        for path in Path("src/whetstone/optimization").glob("*.py")
    )
    assert "whetstone." + "runner" not in text
    assert "whetstone." + "envs" not in text
