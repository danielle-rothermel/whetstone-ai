"""Legacy import path.

Implementation lives in whetstone.envs.code_comp.
"""

from whetstone.envs.code_comp.constants import (
    BLENDED_METRIC_ID,
    DECODER_TEMPLATE,
    ED1_BLENDED_REWARD_NAME,
    ED1_CANONICAL_MODEL,
    ED1_COMPRESSED_DESCRIPTION_LENGTH_NAME,
    ED1_COMPRESSION_NAME,
    ED1_DATASET_ID,
    ED1_DATASET_REVISION,
    ED1_DEFAULT_BUDGET_RATIO,
    ED1_ENV_NAME,
    ED1_INVALID_BODY,
    ED1_SUBMISSION_SCORE_NAME,
    ENCODER_BODY_A,
    ENCODER_BODY_B,
    ENCODER_FRAME,
    ENCODER_FRAME_NO_BUDGET,
)
from whetstone.envs.code_comp.dataset import (
    Ed1Instance,
    ed1_instance_from_task,
    humaneval_task_from_instance,
    load_ed1_tasks,
)
from whetstone.envs.code_comp.modes.encdec import (
    Ed1Experiment,
    Ed1TaskModelConfig,
    Ed1TaskModelKind,
    build_ed1_experiment,
    ed1_blend_config_from_metadata,
    ed1_ceiling_candidate,
    ed1_initial_candidate,
    ed1_preview_metadata,
    ed1_runtime_from_metadata,
    ed1_task_model_from_metadata,
)
from whetstone.envs.code_comp.mutation_surface import (
    Ed1BodyError,
    ed1_body_rejection,
    render_encoder_frame,
    validate_ed1_body,
)
from whetstone.envs.code_comp.preview import (
    build_ed1_preview_engine,
    run_ed1_anchor_baseline_preview,
    run_ed1_anchor_baseline_sweep,
    run_ed1_copro_scoring_preview,
)
from whetstone.envs.code_comp.procedure import (
    build_code_eval_procedure_config,
    build_ed1_procedure_config,
)
from whetstone.envs.code_comp.reward.blended import (
    ED1_DEFAULT_BLEND_CONFIG,
    BoundedCompressionMetricConfig,
    build_ed1_blended_reward_policy,
    ed1_blended_aggregate_values,
    ed1_reward_from_blended,
    reward_from_primary_score,
)

__all__ = [
    "BLENDED_METRIC_ID",
    "DECODER_TEMPLATE",
    "ED1_BLENDED_REWARD_NAME",
    "ED1_CANONICAL_MODEL",
    "ED1_COMPRESSED_DESCRIPTION_LENGTH_NAME",
    "ED1_COMPRESSION_NAME",
    "ED1_DATASET_ID",
    "ED1_DATASET_REVISION",
    "ED1_DEFAULT_BLEND_CONFIG",
    "ED1_DEFAULT_BUDGET_RATIO",
    "ED1_ENV_NAME",
    "ED1_INVALID_BODY",
    "ED1_SUBMISSION_SCORE_NAME",
    "ENCODER_BODY_A",
    "ENCODER_BODY_B",
    "ENCODER_FRAME",
    "ENCODER_FRAME_NO_BUDGET",
    "BoundedCompressionMetricConfig",
    "Ed1BodyError",
    "Ed1Experiment",
    "Ed1Instance",
    "Ed1TaskModelConfig",
    "Ed1TaskModelKind",
    "build_code_eval_procedure_config",
    "build_ed1_blended_reward_policy",
    "build_ed1_experiment",
    "build_ed1_preview_engine",
    "build_ed1_procedure_config",
    "ed1_blend_config_from_metadata",
    "ed1_blended_aggregate_values",
    "ed1_body_rejection",
    "ed1_ceiling_candidate",
    "ed1_initial_candidate",
    "ed1_instance_from_task",
    "ed1_preview_metadata",
    "ed1_reward_from_blended",
    "ed1_runtime_from_metadata",
    "ed1_task_model_from_metadata",
    "humaneval_task_from_instance",
    "load_ed1_tasks",
    "render_encoder_frame",
    "reward_from_primary_score",
    "run_ed1_anchor_baseline_preview",
    "run_ed1_anchor_baseline_sweep",
    "run_ed1_copro_scoring_preview",
    "validate_ed1_body",
]
