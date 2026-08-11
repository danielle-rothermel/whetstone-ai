from __future__ import annotations

from dr_code.humaneval import HUMANEVAL_OVERRIDE_SET
from dr_code.humaneval.plus_dataset import HF_REVISION

from whetstone.evaluation import identity_hash_for

CODE_COMP_ENV_NAME = "code_comp"


BLENDED_METRIC_ID = "primary_score_with_bounded_compression_penalty"

CODE_COMP_CANONICAL_MODEL = "deepseek/deepseek-v4-flash"

CODE_COMP_DEFAULT_BUDGET_RATIO = 0.5

CODE_COMP_DATASET_ID = "evalplus/humanevalplus"
CODE_COMP_DATASET_REVISION = identity_hash_for(
    schema="whetstone.humaneval.dataset_coordinate",
    payload={
        "dataset_id": CODE_COMP_DATASET_ID,
        "upstream_revision": HF_REVISION,
        "override_set": HUMANEVAL_OVERRIDE_SET.model_dump(mode="json"),
    },
)

CODE_COMP_SUBMISSION_SCORE_NAME = "humaneval_submission_score"

CODE_COMP_BLENDED_REWARD_NAME = "blended_reward"
CODE_COMP_COMPRESSED_DESCRIPTION_LENGTH_NAME = "compressed_description_length"
CODE_COMP_COMPRESSION_NAME = "compression_ratio"

CODE_COMP_STRATUM = "humaneval_plus"

DEFINITION_VERSION = "1"

ENCODER_FRAME = (
    "{body}\n"
    "Use at most {max_budget} characters.\n"
    "```python\n{input_code}\n```"
)

ENCODER_FRAME_NO_BUDGET = "{body}\n```python\n{input_code}\n```"

ENCODER_BODY_A = "Provide a concise description of the following code."

ENCODER_BODY_B = (
    "Please compress the following code into a description another agent can "
    "use to reconstruct a function that behaves the same as the following "
    "code."
)

CODE_COMP_INVALID_BODY = "code_comp_invalid_encoder_body"

MUTATION_FIELD = "user_prompt_template"

DECODER_TEMPLATE = (
    "Decode the description into functional Python code. Output only Python "
    "code.\n\n{encoder_output}"
)

__all__ = [
    "BLENDED_METRIC_ID",
    "CODE_COMP_BLENDED_REWARD_NAME",
    "CODE_COMP_CANONICAL_MODEL",
    "CODE_COMP_COMPRESSED_DESCRIPTION_LENGTH_NAME",
    "CODE_COMP_COMPRESSION_NAME",
    "CODE_COMP_DATASET_ID",
    "CODE_COMP_DATASET_REVISION",
    "CODE_COMP_DEFAULT_BUDGET_RATIO",
    "CODE_COMP_ENV_NAME",
    "CODE_COMP_INVALID_BODY",
    "CODE_COMP_STRATUM",
    "CODE_COMP_SUBMISSION_SCORE_NAME",
    "DECODER_TEMPLATE",
    "DEFINITION_VERSION",
    "ENCODER_BODY_A",
    "ENCODER_BODY_B",
    "ENCODER_FRAME",
    "ENCODER_FRAME_NO_BUDGET",
    "MUTATION_FIELD",
]
