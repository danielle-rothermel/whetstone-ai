from __future__ import annotations

from types import ModuleType

import pytest

import dr_dspy.humaneval_direct_dbos as direct_dbos
import dspy
from dr_dspy.lm_utils import ModelConfig
from dr_dspy.signatures import DspySignatureConfig, FieldSignature

TEST_DIRECT_CONFIG = direct_dbos.DirectHumanEvalExperimentConfig(
    script_kind="test_humaneval_eval_only_dbos_v0",
    dataset_name="evalplus/humanevalplus",
    dataset_split="test",
    solve_signature=DspySignatureConfig(
        name="Solve",
        fields=(
            FieldSignature(name="prompt", type=str, role=dspy.InputField()),
            FieldSignature(
                name="code", type=dspy.Code, role=dspy.OutputField()
            ),
        ),
        instructions="Write functional Python code for the prompt.",
    ),
    default_model_configs=(
        ModelConfig(model="test/model-a", reasoning={}),
    ),
    default_sample_count=10,
    default_seed=0,
    default_temperatures=(0.0,),
    default_repetitions=1,
    default_max_completion_tokens=1000,
    default_subprocess_timeout=15.0,
)


@pytest.fixture(scope="session")
def eval_dbos_harness() -> ModuleType:
    direct_dbos.create_app(TEST_DIRECT_CONFIG)
    return direct_dbos
