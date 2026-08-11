"""Legacy import path.

Implementation lives in whetstone.envs.code_comp.
"""

from whetstone.envs.code_comp.behavior_matrix import (
    DEFAULT_CONCURRENCY,
    DEFAULT_TASK_MANIFEST,
    EXCLUDED_TASK_IDS,
    FULL_BUDGET_RATIOS,
    BehaviorMatrixPlan,
    Ed1BehaviorMatrixTreatmentPlan,
    build_matrix_plan,
    run_ed1_baseline_behavior_matrix,
)
from whetstone.optimization.validation.matrix import map_openai_credential

Ed1BehaviorMatrixPlan = BehaviorMatrixPlan

__all__ = [
    "DEFAULT_CONCURRENCY",
    "DEFAULT_TASK_MANIFEST",
    "EXCLUDED_TASK_IDS",
    "FULL_BUDGET_RATIOS",
    "BehaviorMatrixPlan",
    "Ed1BehaviorMatrixPlan",
    "Ed1BehaviorMatrixTreatmentPlan",
    "build_matrix_plan",
    "map_openai_credential",
    "run_ed1_baseline_behavior_matrix",
]
