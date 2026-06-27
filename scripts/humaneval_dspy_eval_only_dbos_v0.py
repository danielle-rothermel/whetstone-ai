from __future__ import annotations

import dspy
from dr_dspy.humaneval_direct_dbos import (
    DirectHumanEvalExperimentConfig,
    create_app,
)
from dr_dspy.lm_utils import ModelConfig
from dr_dspy.runtime import run_typer_app
from dr_dspy.signatures import DspySignatureConfig, FieldSignature

SCRIPT_KIND = "humaneval_eval_only_dbos_v0"
DATASET_NAME = "evalplus/humanevalplus"
DATASET_SPLIT = "test"
DEFAULT_SAMPLE_COUNT = 10
DEFAULT_SEED = 0
DEFAULT_TEMPERATURES = (0.0,)
DEFAULT_REPETITIONS = 1
DEFAULT_MAX_COMPLETION_TOKENS = 1000
DEFAULT_SUBPROCESS_TIMEOUT = 15.0

SOLVE_SIGNATURE = DspySignatureConfig(
    name="Solve",
    fields=(
        FieldSignature(name="prompt", type=str, role=dspy.InputField()),
        FieldSignature(name="code", type=dspy.Code, role=dspy.OutputField()),
    ),
    instructions=(
        "Write functional code in Python according to the prompt. "
        "Output only the code solution."
    ),
)

DEFAULT_MODEL_CONFIGS = (
    ModelConfig(model="openai/gpt-5.1-codex-mini", reasoning={}),
    ModelConfig(model="moonshotai/kimi-k2-0905", reasoning={}),
    ModelConfig(model="qwen/qwen3-coder-next", reasoning={}),
    ModelConfig(
        model="deepseek/deepseek-v3.1-terminus",
        reasoning={"enabled": False},
    ),
    ModelConfig(model="moonshotai/kimi-k2", reasoning={}),
    ModelConfig(model="z-ai/glm-4.7", reasoning={"enabled": False}),
    ModelConfig(model="z-ai/glm-5", reasoning={"enabled": False}),
    ModelConfig(
        model="deepseek/deepseek-v4-pro",
        reasoning={"enabled": False},
    ),
    ModelConfig(
        model="deepseek/deepseek-v4-flash",
        reasoning={"enabled": False},
    ),
    ModelConfig(model="mistralai/mistral-large-2512", reasoning={}),
    ModelConfig(model="openai/gpt-oss-120b", reasoning={"effort": "low"}),
    ModelConfig(model="mistralai/codestral-2508", reasoning={}),
    ModelConfig(model="qwen/qwen3-coder-flash", reasoning={}),
    ModelConfig(model="openai/gpt-5-nano", reasoning={"effort": "low"}),
    ModelConfig(
        model="deepseek/deepseek-chat-v3.1",
        reasoning={"enabled": False},
    ),
    ModelConfig(model="openai/gpt-5.4-nano", reasoning={"effort": "none"}),
)

EXPERIMENT_CONFIG = DirectHumanEvalExperimentConfig(
    script_kind=SCRIPT_KIND,
    dataset_name=DATASET_NAME,
    dataset_split=DATASET_SPLIT,
    solve_signature=SOLVE_SIGNATURE,
    default_model_configs=DEFAULT_MODEL_CONFIGS,
    default_sample_count=DEFAULT_SAMPLE_COUNT,
    default_seed=DEFAULT_SEED,
    default_temperatures=DEFAULT_TEMPERATURES,
    default_repetitions=DEFAULT_REPETITIONS,
    default_max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    default_subprocess_timeout=DEFAULT_SUBPROCESS_TIMEOUT,
)

app = create_app(EXPERIMENT_CONFIG)

if __name__ == "__main__":
    run_typer_app(app)
