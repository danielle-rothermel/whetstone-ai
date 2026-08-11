from __future__ import annotations

import importlib
import pkgutil
from dataclasses import is_dataclass

import whetstone.evaluation as local_evaluation
import whetstone.experiment.task_selection as task_selection_package
from whetstone.experiment.task_selection import (
    ResolvedSplit,
    TaskRoleSelection,
    TaskSplitRole,
    TaskSplitRoles,
)


def test_task_selection_has_no_duplicate_eval_contract_types() -> None:
    evaluation_type_names = {
        name
        for name in local_evaluation.__all__
        if isinstance(getattr(local_evaluation, name), type)
    }
    modules = (
        task_selection_package,
        *(
            importlib.import_module(module_info.name)
            for module_info in pkgutil.walk_packages(
                task_selection_package.__path__,
                prefix=f"{task_selection_package.__name__}.",
            )
        ),
    )
    duplicate_definitions = {
        f"{module.__name__}.{name}"
        for module in modules
        for name, value in vars(module).items()
        if name in evaluation_type_names
        and isinstance(value, type)
        and value.__module__ == module.__name__
    }

    assert duplicate_definitions == set()


def test_internal_value_objects_are_frozen_slotted_dataclasses() -> None:
    roles = TaskSplitRoles(
        pool_key="ed1",
        train_ids=("Synthetic/0",),
        val_ids=("Synthetic/1",),
        test_ids=("Synthetic/2",),
        content_hash="a" * 64,
    )
    selection = TaskRoleSelection(
        manifest_content_hash="b" * 64,
        pool_key="ed1",
        role=TaskSplitRole.TRAIN,
        task_ids=("Synthetic/0",),
    )
    resolved: ResolvedSplit[str] = ResolvedSplit(
        internal=("a",),
        official=("b",),
        manifest_tag="tsm:abc.ed1",
        official_capped=None,
    )
    for value in (roles, resolved):
        assert is_dataclass(value)
        assert hasattr(type(value), "__slots__")
        assert type(value).__dataclass_params__.frozen

    assert selection.model_config.get("frozen") is True
