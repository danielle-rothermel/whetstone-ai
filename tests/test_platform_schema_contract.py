"""Whetstone must never let dr_platform default to the ``platform`` prefix.

The kernel stores live in run schemas migrated with ``prefix="whetstone"``;
a dr_platform call that omits ``schema=`` silently targets ``platform_*``
tables that do not exist (UndefinedTable at runtime, first seen when
``submit-canary`` registration failed against ``v6accept_0713e``).
"""

import ast
import importlib
import inspect
import pkgutil
from pathlib import Path

import dr_platform

SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "whetstone"


def _schema_defaulting_names() -> set[str]:
    """Names of dr_platform callables with an optional PlatformSchema param."""
    modules = [dr_platform]
    for info in pkgutil.iter_modules(dr_platform.__path__):
        modules.append(importlib.import_module(f"dr_platform.{info.name}"))
    names: set[str] = set()
    for module in modules:
        for name, obj in vars(module).items():
            if name.startswith("_") or not inspect.isfunction(obj):
                continue
            try:
                parameters = inspect.signature(obj).parameters
            except (TypeError, ValueError):
                continue
            schema_param = parameters.get("schema")
            if (
                schema_param is not None
                and schema_param.default is None
                and "PlatformSchema" in str(schema_param.annotation)
            ):
                names.add(name)
    return names


def test_every_dr_platform_call_passes_explicit_schema() -> None:
    defaulting = _schema_defaulting_names()
    assert "submit" in defaulting and "list_attempts" in defaulting

    # Covered call forms: direct and aliased ``from dr_platform import ...``
    # calls, which are the only forms whetstone uses.  Module-attribute calls
    # (``import dr_platform``) and the one schema-defaulting class
    # (``PostgresClaimTransitionStore``) appear nowhere in src/whetstone;
    # extend this walker before introducing either form.
    violations: list[str] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        local_to_dr_platform_name: dict[str, str] = {}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.split(".")[0] == "dr_platform"
            ):
                for alias in node.names:
                    local_to_dr_platform_name[alias.asname or alias.name] = (
                        alias.name
                    )
        watched = {
            local
            for local, original in local_to_dr_platform_name.items()
            if original in defaulting
        }
        if not watched:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in watched
                and not any(kw.arg == "schema" for kw in node.keywords)
            ):
                violations.append(
                    f"{path.relative_to(SRC_ROOT.parent.parent)}:{node.lineno}"
                    f" {node.func.id}() without schema="
                )
    assert not violations, (
        "dr_platform calls defaulting to the platform prefix:\n"
        + "\n".join(violations)
    )
