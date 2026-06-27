from __future__ import annotations

import dspy
from dr_dspy.humaneval_encdec_dbos import (
    EncDecHumanEvalExperimentConfig,
    EncDecPair,
    create_app,
)
from dr_dspy.lm_utils import ModelConfig
from dr_dspy.runtime import run_typer_app
from dr_dspy.signatures import DspySignatureConfig, FieldSignature

SCRIPT_KIND = "humaneval_eval_only_encdec_dbos_v0"
DATASET_NAME = "evalplus/humanevalplus"
DATASET_SPLIT = "test"
DEFAULT_SAMPLE_COUNT = 10
DEFAULT_SEED = 0
DEFAULT_ENCODER_TEMPERATURES = (0.0,)
DEFAULT_DECODER_TEMPERATURES = (0.0,)
DEFAULT_REPETITIONS = 1
DEFAULT_MAX_COMPLETION_TOKENS = 2000
DEFAULT_SUBPROCESS_TIMEOUT = 15.0

ENCODER_SIGNATURE = DspySignatureConfig(
    name="EncodeCode",
    fields=(
        FieldSignature(name="code", type=str, role=dspy.InputField()),
        FieldSignature(
            name="description", type=str, role=dspy.OutputField()
        ),
    ),
    instructions=(
        "Encode this Python function implementation into a complete "
        "lossless description. Preserve all behavior needed to reconstruct "
        "the code, but do not output Python code."
    ),
)

DECODER_SIGNATURE = DspySignatureConfig(
    name="DecodeCode",
    fields=(
        FieldSignature(
            name="description", type=str, role=dspy.InputField()
        ),
        FieldSignature(name="code", type=dspy.Code, role=dspy.OutputField()),
    ),
    instructions=(
        "Decode the description into functional Python code. "
        "Output only Python code."
    ),
)

DEFAULT_MODEL_PAIRS = (
    EncDecPair(
        encoder=ModelConfig(
            model="openai/gpt-5.1-codex-mini",
            reasoning={},
        ),
        decoder=ModelConfig(
            model="openai/gpt-5.1-codex-mini",
            reasoning={},
        ),
    ),
)

EXPERIMENT_CONFIG = EncDecHumanEvalExperimentConfig(
    script_kind=SCRIPT_KIND,
    dataset_name=DATASET_NAME,
    dataset_split=DATASET_SPLIT,
    encoder_signature=ENCODER_SIGNATURE,
    decoder_signature=DECODER_SIGNATURE,
    default_model_pairs=DEFAULT_MODEL_PAIRS,
    default_sample_count=DEFAULT_SAMPLE_COUNT,
    default_seed=DEFAULT_SEED,
    default_encoder_temperatures=DEFAULT_ENCODER_TEMPERATURES,
    default_decoder_temperatures=DEFAULT_DECODER_TEMPERATURES,
    default_repetitions=DEFAULT_REPETITIONS,
    default_max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    default_subprocess_timeout=DEFAULT_SUBPROCESS_TIMEOUT,
)

app = create_app(EXPERIMENT_CONFIG)

if __name__ == "__main__":
    run_typer_app(app)
