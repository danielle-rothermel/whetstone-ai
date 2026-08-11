from __future__ import annotations

import importlib
import pkgutil

import whetstone.evaluation as local_evaluation
import whetstone.evaluation.preview as preview_package


def test_preview_package_has_no_duplicate_contract_types() -> None:
    evaluation_type_names = {
        name
        for name in local_evaluation.__all__
        if isinstance(getattr(local_evaluation, name), type)
    }
    modules = (
        preview_package,
        *(
            importlib.import_module(module_info.name)
            for module_info in pkgutil.walk_packages(
                preview_package.__path__,
                prefix=f"{preview_package.__name__}.",
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
