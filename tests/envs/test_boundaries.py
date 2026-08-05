"""Static ownership boundaries for environment contracts."""

from __future__ import annotations

import ast
from pathlib import Path


def _python_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.rglob("*.py")))


def test_environment_imports_exclude_orchestration_and_concrete_adapters() -> (
    None
):
    forbidden = (
        "whetstone." + "runner",
        "whetstone.optimization.adapters",
        "mcp",
    )
    for path in _python_files(Path("src/whetstone/envs")):
        tree = ast.parse(path.read_text(), filename=str(path))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append(node.module)
        for module in imports:
            assert not module.startswith(forbidden), (
                f"{path} imports forbidden module {module!r}"
            )
