from __future__ import annotations

import dspy
from dr_dspy.experiments.humaneval_encdec import (
    EncDecHumanEvalExperimentConfig,
    EncDecPair,
    create_app,
)
from dr_dspy.lm.signatures import DspySignatureConfig, FieldSignature
from dr_dspy.lm.utils import ModelConfig
from dr_dspy.runtime import run_typer_app

SCRIPT_KIND = "humaneval_eval_only_encdec_dbos_v0"
DATASET_NAME = "evalplus/humanevalplus"
DATASET_SPLIT = "test"
DEFAULT_SAMPLE_COUNT = 164  # all HumanEval+ tasks
DEFAULT_SEED = 0
DEFAULT_ENCODER_TEMPERATURES = (0.0,)
DEFAULT_DECODER_TEMPERATURES = (0.0,)
DEFAULT_BUDGET_RATIOS: tuple[float | None, ...] = (
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    None,
)
DEFAULT_REPETITIONS = 3
DEFAULT_MAX_COMPLETION_TOKENS = 2000
DEFAULT_SUBPROCESS_TIMEOUT = 15.0

ENCODER_SIGNATURE = DspySignatureConfig(
    name="EncodeCode",
    fields=(
        FieldSignature(name="code", type=str, role=dspy.InputField()),
        FieldSignature(name="description", type=str, role=dspy.OutputField()),
    ),
    instructions=(
        "Encode this Python function implementation into a complete "
        "lossless description. Preserve all behavior needed to reconstruct "
        "the code, but do not output Python code."
    ),
)

ENCODER_BUDGETED_SIGNATURE = DspySignatureConfig(
    name="EncodeCodeBudgeted",
    fields=(
        FieldSignature(name="code", type=str, role=dspy.InputField()),
        FieldSignature(
            name="max_characters", type=int, role=dspy.InputField()
        ),
        FieldSignature(name="description", type=str, role=dspy.OutputField()),
    ),
    instructions=(
        "Encode this Python function implementation into a complete "
        "lossless description. Preserve all behavior needed to reconstruct "
        "the code, but do not output Python code. Keep the description "
        "within at most max_characters characters."
    ),
)

DECODER_SIGNATURE = DspySignatureConfig(
    name="DecodeCode",
    fields=(
        FieldSignature(name="description", type=str, role=dspy.InputField()),
        FieldSignature(name="code", type=dspy.Code, role=dspy.OutputField()),
    ),
    instructions=(
        "Decode the description into functional Python code. "
        "Output only Python code."
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

# Each model encodes and decodes its own output (self-pairs).
DEFAULT_MODEL_PAIRS = tuple(
    EncDecPair(encoder=config, decoder=config)
    for config in DEFAULT_MODEL_CONFIGS
)

EXPERIMENT_CONFIG = EncDecHumanEvalExperimentConfig(
    script_kind=SCRIPT_KIND,
    dataset_name=DATASET_NAME,
    dataset_split=DATASET_SPLIT,
    encoder_signature=ENCODER_SIGNATURE,
    budgeted_encoder_signature=ENCODER_BUDGETED_SIGNATURE,
    decoder_signature=DECODER_SIGNATURE,
    default_model_pairs=DEFAULT_MODEL_PAIRS,
    default_sample_count=DEFAULT_SAMPLE_COUNT,
    default_seed=DEFAULT_SEED,
    default_encoder_temperatures=DEFAULT_ENCODER_TEMPERATURES,
    default_decoder_temperatures=DEFAULT_DECODER_TEMPERATURES,
    default_budget_ratios=DEFAULT_BUDGET_RATIOS,
    default_repetitions=DEFAULT_REPETITIONS,
    default_max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    default_subprocess_timeout=DEFAULT_SUBPROCESS_TIMEOUT,
)

app = create_app(EXPERIMENT_CONFIG)

if __name__ == "__main__":
    run_typer_app(app)
