from __future__ import annotations

from dr_code.humaneval import HUMANEVAL_OVERRIDE_SET
from dr_code.humaneval.plus_dataset import HF_REVISION

from whetstone.evaluation import identity_hash_for

ED1_ENV_NAME = "ed1"

BLENDED_METRIC_ID = "primary_score_with_bounded_compression_penalty"

ED1_CANONICAL_MODEL = "deepseek/deepseek-v4-flash"

ED1_DEFAULT_BUDGET_RATIO = 0.5

ED1_DATASET_ID = "evalplus/humanevalplus"
ED1_DATASET_REVISION = identity_hash_for(
    schema="whetstone.humaneval.dataset_coordinate",
    payload={
        "dataset_id": ED1_DATASET_ID,
        "upstream_revision": HF_REVISION,
        "override_set": HUMANEVAL_OVERRIDE_SET.model_dump(mode="json"),
    },
)

ED1_SUBMISSION_SCORE_NAME = "humaneval_submission_score"

ED1_BLENDED_REWARD_NAME = "blended_reward"
ED1_COMPRESSED_DESCRIPTION_LENGTH_NAME = "compressed_description_length"
ED1_COMPRESSION_NAME = "compression_ratio"

ED1_STRATUM = "humaneval_plus"

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

ED1_INVALID_BODY = "ed1_invalid_encoder_body"

MUTATION_FIELD = "user_prompt_template"

DECODER_TEMPLATE = (
    "Decode the description into functional Python code. Output only Python "
    "code.\n\n{encoder_output}"
)

__all__ = [
    "BLENDED_METRIC_ID",
    "DECODER_TEMPLATE",
    "DEFINITION_VERSION",
    "ED1_BLENDED_REWARD_NAME",
    "ED1_CANONICAL_MODEL",
    "ED1_COMPRESSED_DESCRIPTION_LENGTH_NAME",
    "ED1_COMPRESSION_NAME",
    "ED1_DATASET_ID",
    "ED1_DATASET_REVISION",
    "ED1_DEFAULT_BUDGET_RATIO",
    "ED1_ENV_NAME",
    "ED1_INVALID_BODY",
    "ED1_STRATUM",
    "ED1_SUBMISSION_SCORE_NAME",
    "ENCODER_BODY_A",
    "ENCODER_BODY_B",
    "ENCODER_FRAME",
    "ENCODER_FRAME_NO_BUDGET",
    "MUTATION_FIELD",
]
