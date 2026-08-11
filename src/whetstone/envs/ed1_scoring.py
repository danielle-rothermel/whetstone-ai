"""Legacy import path.

Implementation lives in whetstone.envs.code_comp.scoring.
"""

from dr_code.caching import CheckpointedExecutionCache
from dr_code.humaneval import (
    HumanEvalSubmissionRequest,
    score_humaneval_submissions_batch,
)
from dr_store import SqliteRecordCache

from whetstone.envs.code_comp.scoring import (
    ED1_SCORING_PREFLIGHT_TASK_ID,
    ED1_SCORING_PROFILE_ID,
    ED1_SCORING_PROFILE_VERSION,
    BatchScoringDeadlineExceeded,
    CheckpointedCodeBatchScorer,
    CodeBatchScorer,
    CodeScore,
    CodeScoringInput,
    _project_submission_score,
    run_ed1_scoring_preflight,
    score_ed1_submission,
)

__all__ = [
    "ED1_SCORING_PREFLIGHT_TASK_ID",
    "ED1_SCORING_PROFILE_ID",
    "ED1_SCORING_PROFILE_VERSION",
    "BatchScoringDeadlineExceeded",
    "CheckpointedCodeBatchScorer",
    "CheckpointedExecutionCache",
    "CodeBatchScorer",
    "CodeScore",
    "CodeScoringInput",
    "HumanEvalSubmissionRequest",
    "SqliteRecordCache",
    "_project_submission_score",
    "run_ed1_scoring_preflight",
    "score_ed1_submission",
    "score_humaneval_submissions_batch",
]
