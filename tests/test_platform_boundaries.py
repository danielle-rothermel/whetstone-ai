from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src" / "dr_dspy"
PURE_PACKAGE_PATHS = (
    SRC_ROOT / "graph",
    SRC_ROOT / "humaneval",
    SRC_ROOT / "eval_failures",
)
PURE_MODULE_PATHS = (
    SRC_ROOT / "serialization.py",
    SRC_ROOT / "lm" / "boundary.py",
)
DSPY_FREE_PATHS = (
    *PURE_PACKAGE_PATHS,
    SRC_ROOT / "lm" / "boundary.py",
)
FORBIDDEN_PLATFORM_IMPORTS = (
    "dbos",
    "psycopg",
    "psycopg_pool",
    "dr_dspy.experiments",
    "dr_dspy.harness",
    "dr_dspy.runtime",
)
FORBIDDEN_DSPY_IMPORTS = ("dspy",)
ALLOWED_EXISTING_IMPORTS = {
    # recordable_jsonb is the documented JSONB adapter exception at the
    # persistence boundary. The rest of eval_failures should stay DB-free.
    SRC_ROOT / "eval_failures" / "recording.py": {
        "psycopg.types.json",
    },
}


def python_files() -> list[Path]:
    files = list(PURE_MODULE_PATHS)
    for package_path in PURE_PACKAGE_PATHS:
        files.extend(package_path.rglob("*.py"))
    return sorted(files)


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    return imports


def is_relative_to(path: Path, base_path: Path) -> bool:
    return path == base_path or path.is_relative_to(base_path)


def matches_module(module: str, forbidden_modules: tuple[str, ...]) -> bool:
    return any(
        module == forbidden or module.startswith(f"{forbidden}.")
        for forbidden in forbidden_modules
    )


def forbids_dspy(path: Path) -> bool:
    return any(
        is_relative_to(path, dspy_free_path)
        for dspy_free_path in DSPY_FREE_PATHS
    )


def is_forbidden_import(path: Path, module: str) -> bool:
    if module in ALLOWED_EXISTING_IMPORTS.get(path, set()):
        return False
    if matches_module(module, FORBIDDEN_PLATFORM_IMPORTS):
        return True
    return forbids_dspy(path) and matches_module(
        module,
        FORBIDDEN_DSPY_IMPORTS,
    )


def test_pure_modules_do_not_import_legacy_or_runtime_surfaces() -> None:
    violations = []
    for path in python_files():
        for module in sorted(imported_modules(path)):
            if is_forbidden_import(path, module):
                relative_path = path.relative_to(PROJECT_ROOT)
                violations.append(f"{relative_path}: {module}")

    assert violations == []
