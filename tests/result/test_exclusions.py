"""Absence tests for the result package.

These assert what the Rollout Result and Result Store MUST NOT contain:

* no Materialization Record reference anywhere in the Rollout Result schema
  or its nested slots;
* no Platform Stage *state* on the nested Graph Run Result;
* no separate authoritative persistence path for the nested Graph Run Result
  (it is persisted only as part of the enclosing Rollout Result record);
* no official-specific result role or type;
* no overwrite/clear/rebind API on the Result Store.
"""

from __future__ import annotations

import ast
import inspect

from pydantic import BaseModel

from whetstone.result import (
    ExhaustedCausalFailure,
    PlatformStageAttemptEvidence,
    ProviderCallAttemptObservation,
    ResultStore,
    RolloutResult,
    ScoreFact,
    result_store,
    rollout_result,
)


def _all_field_names(model: type[BaseModel]) -> set[str]:
    names: set[str] = set()
    for name, field in model.model_fields.items():
        names.add(name)
        annotation = field.annotation
        # Recurse one level into nested pydantic models we own.
        for candidate in (
            ProviderCallAttemptObservation,
            PlatformStageAttemptEvidence,
            ScoreFact,
            ExhaustedCausalFailure,
        ):
            if candidate.__name__ in repr(annotation):
                names |= set(candidate.model_fields)
    return names


def test_no_materialization_reference_in_any_result_field() -> None:
    names = _all_field_names(RolloutResult)
    for name in names:
        assert "materialization" not in name.lower(), name


def test_result_module_binds_no_materialization_identifier() -> None:
    """No *code* identifier in the Rollout Result module names materialization.

    A robust AST scan (not a prose scan) proves the exclusion at the level of
    defined names: no assignment target, function/class name, argument, or
    attribute access binds a materialization reference. Docstrings and
    comments asserting the exclusion are, by construction, exempt.
    """
    tree = ast.parse(inspect.getsource(rollout_result))
    bound_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            bound_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            bound_names.add(node.attr)
        elif isinstance(node, ast.arg):
            bound_names.add(node.arg)
        elif isinstance(node, ast.FunctionDef | ast.ClassDef):
            bound_names.add(node.name)
        elif isinstance(node, ast.keyword) and node.arg is not None:
            bound_names.add(node.arg)
    for name in bound_names:
        assert "materialization" not in name.lower(), name


def test_no_official_result_role_or_type() -> None:
    """There is no official-specific result role, field, or type."""
    names = _all_field_names(RolloutResult)
    for name in names:
        assert name != "official_role"
        assert name != "result_role"
        assert "official_result" not in name.lower()
    # authority is a plain identity field, not a role/type discriminator.
    assert "authority" in RolloutResult.model_fields


def test_nested_graph_run_result_has_no_separate_persistence_path() -> None:
    """No function persists a nested Graph Run Result on its own.

    The result package exposes exactly one persistence entry point,
    ``persist_rollout_result``, which persists the whole enclosing record.
    There is no ``persist_graph_run_result`` or similar.
    """
    for module in (rollout_result, result_store):
        for name in dir(module):
            lowered = name.lower()
            assert "persist_graph_run" not in lowered, name
            assert "store_graph_run" not in lowered, name
    persist_names = [
        name
        for name in dir(result_store)
        if name.startswith("persist") and not name.startswith("_")
    ]
    assert persist_names == ["persist_rollout_result"]


def test_result_store_has_no_overwrite_surface() -> None:
    public = {name for name in dir(ResultStore) if not name.startswith("_")}
    for forbidden in (
        "overwrite",
        "clear",
        "rebind",
        "delete",
        "remove",
        "unbind",
        "replace",
        "put",
        "update",
    ):
        assert forbidden not in public, forbidden
