from __future__ import annotations

import ast
from pathlib import Path

_FORBIDDEN_PREFIXES = (
    "whetstone.runner",
    "whetstone.optimization.adapters",
    "whetstone.optimization.copro",
    "whetstone.optimization.proposal",
)


def _module_level_imports(tree: ast.Module) -> list[str]:
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and node.level == 0:
                imports.append(node.module)
    return imports


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def test_code_comp_registry_imports_exclude_runner_and_optimization() -> None:
    root = Path("src/whetstone/envs/code_comp/registry.py")
    tree = ast.parse(root.read_text(), filename=str(root))
    for module in _module_level_imports(tree):
        assert not module.startswith(_FORBIDDEN_PREFIXES), (
            f"{root} imports forbidden module {module!r}"
        )


def test_code_comp_surface_modules_exclude_runner_and_optimization() -> None:
    root = Path("src/whetstone/envs/code_comp")
    targets = [
        root / "behavior_matrix.py",
        root / "preview.py",
        *sorted((root / "modes").glob("*.py")),
    ]
    for path in targets:
        tree = ast.parse(path.read_text(), filename=str(path))
        for module in _module_level_imports(tree):
            if module.startswith("whetstone.optimization.validation"):
                continue
            assert not module.startswith(_FORBIDDEN_PREFIXES), (
                f"{path} imports forbidden module {module!r}"
            )


def test_code_comp_imports_exclude_runner_and_optimization() -> None:
    root = Path("src/whetstone/envs/code_comp")
    for path in _python_files(root):
        tree = ast.parse(path.read_text(), filename=str(path))
        for module in _module_level_imports(tree):
            if module.startswith("whetstone.optimization.validation"):
                continue
            assert not module.startswith(_FORBIDDEN_PREFIXES), (
                f"{path} imports forbidden module {module!r}"
            )
