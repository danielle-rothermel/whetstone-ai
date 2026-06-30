"""Archive phase contract tests for legacy labeling and importability."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_PATH = PROJECT_ROOT / "README.md"

LEGACY_MODULE_PATHS = (
    PROJECT_ROOT / "src" / "dr_dspy" / "experiments" / "__init__.py",
    PROJECT_ROOT / "src" / "dr_dspy" / "harness" / "__init__.py",
    PROJECT_ROOT / "src" / "dr_dspy" / "runtime.py",
    PROJECT_ROOT / "src" / "dr_dspy" / "lm" / "__init__.py",
    PROJECT_ROOT / "src" / "dr_dspy" / "lm" / "runner.py",
    PROJECT_ROOT / "src" / "dr_dspy" / "lm" / "openrouter.py",
    PROJECT_ROOT / "src" / "dr_dspy" / "lm" / "signatures.py",
    PROJECT_ROOT / "src" / "dr_dspy" / "lm" / "logging.py",
)

LEGACY_IMPORT_MODULES = (
    "dr_dspy.experiments",
    "dr_dspy.harness",
    "dr_dspy.runtime",
    "dr_dspy.lm.runner",
)


def module_docstring(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstring = ast.get_docstring(tree)
    assert docstring is not None, f"missing module docstring: {path}"
    return docstring


def test_legacy_orchestration_modules_remain_importable() -> None:
    for module_name in LEGACY_IMPORT_MODULES:
        importlib.import_module(module_name)


def test_legacy_module_docstrings_mark_v0_surfaces() -> None:
    for path in LEGACY_MODULE_PATHS:
        docstring = module_docstring(path)
        assert "legacy" in docstring.lower(), path


def readme_lines_containing(readme: str, needle: str) -> list[str]:
    return [line for line in readme.splitlines() if needle in line]


def test_readme_marks_legacy_package_layout_and_generation_path() -> None:
    readme = README_PATH.read_text(encoding="utf-8")

    assert "legacy v0" in readme.lower()

    experiments_lines = readme_lines_containing(readme, "`experiments/`")
    assert experiments_lines, "README missing experiments package layout entry"
    assert all("legacy" in line.lower() for line in experiments_lines), (
        experiments_lines
    )

    harness_lines = readme_lines_containing(readme, "`harness/`")
    assert harness_lines, "README missing harness package layout entry"
    assert all("legacy" in line.lower() for line in harness_lines), (
        harness_lines
    )

    assert "legacy `lm.runner.run_predictor`" in readme
