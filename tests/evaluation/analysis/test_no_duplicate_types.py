from __future__ import annotations

import importlib
import pkgutil
from dataclasses import is_dataclass

import whetstone.evaluation as local_evaluation
import whetstone.evaluation.analysis as analysis_package
from whetstone.evaluation.analysis import (
    AnchorCalibrationResult,
    BootstrapCI,
    PowerConfig,
    PowerResult,
    PowerSurfacePoint,
    VarianceDecomposition,
)


def test_analysis_package_has_no_duplicate_contract_types() -> None:
    evaluation_type_names = {
        name
        for name in local_evaluation.__all__
        if isinstance(getattr(local_evaluation, name), type)
    }
    modules = (
        analysis_package,
        *(
            importlib.import_module(module_info.name)
            for module_info in pkgutil.walk_packages(
                analysis_package.__path__,
                prefix=f"{analysis_package.__name__}.",
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
    value_objects = (
        BootstrapCI(point=0.5, low=0.25, high=0.75, level=0.95, resamples=10),
        PowerConfig(),
        VarianceDecomposition(
            base_rate=0.5,
            within_sample_var=0.25,
            interaction_var=0.1,
            between_task_var=0.2,
            anchor_samples=3,
            n_tasks_observed=2,
        ),
        PowerSurfacePoint(
            n_tasks=2,
            num_samples=1,
            calls=2,
            mdd_at_target=0.1,
        ),
    )
    for value in value_objects:
        assert is_dataclass(value)
        assert hasattr(type(value), "__slots__")
        assert type(value).__dataclass_params__.frozen

    result = PowerResult(
        config=PowerConfig(),
        certified_headroom=0.0,
        target_gap=0.0,
        naive_mean=0.5,
        ceiling_mean=0.5,
        pool_ceiling=1,
        decomposition=value_objects[2],
    )
    assert is_dataclass(result)
    assert hasattr(type(result), "__slots__")
    assert type(result).__dataclass_params__.frozen

    assert is_dataclass(AnchorCalibrationResult)
    assert hasattr(AnchorCalibrationResult, "__slots__")
    assert AnchorCalibrationResult.__dataclass_params__.frozen
