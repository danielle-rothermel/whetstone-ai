from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import UNIQUE, StrEnum, verify
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from dr_code.humaneval import (
    HUMANEVAL_OVERRIDE_SET,
    HumanEvalTask,
    load_humaneval_plus,
)
from dr_code.humaneval.plus_dataset import HF_REVISION
from dr_providers import ProviderCallConfig
from dr_store import ObjectStore
from pydantic import BaseModel, ConfigDict
from whetstone_envs.core import Instance

from whetstone.core.identity import TypedRef
from whetstone.core.roles import EvaluationRole
from whetstone.envs.ed1_runtime import (
    Ed1ScoringRuntimeSummary,
    ed1_environment_fingerprint,
)
from whetstone.envs.ed1_scoring import (
    ED1_SCORING_PROFILE_ID,
    ED1_SCORING_PROFILE_VERSION,
    CodeBatchScorer,
    CodeScore,
    run_ed1_scoring_preflight,
)
from whetstone.evaluation.preview.preflight import PreviewMetadata
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.experiment.task_selection import TaskRoleSelection

if TYPE_CHECKING:
    from whetstone.evaluation.analysis.power import PowerConfig
    from whetstone.evaluation.engine import EvaluationEngine
    from whetstone.evaluation.preview.anchor import (
        BaselinePreviewTranscript,
        BaselineSweepTranscript,
    )
    from whetstone.experiment.binding import EvaluationBinding
    from whetstone.optimization.copro.ed1_dry_run import (
        Ed1CoproRoundAttempt,
        Ed1CoproSweepPoint,
        Ed1CoproSweepRanges,
    )
    from whetstone.optimization.copro.scoring_preview import (
        CandidateProgress,
        ScoringTranscript,
    )
    from whetstone.optimization.proposal.proposer import (
        ProposerRouteConfig,
        ProposerTransport,
    )
from whetstone.envs.encdec_rollout import (
    EncDecRolloutDefinition,
    build_encdec_rollout_definition,
    build_encoder_provider_call_config,
)
from whetstone.envs.factory import EnvExperiment
from whetstone.envs.rollout_definition import env_candidate_base_ref
from whetstone.envs.sampling import (
    Completeness,
    EnvEvalConfigs,
    EnvSplitSampling,
    derive_split_sampling,
)
from whetstone.evaluation import (
    EvaluationProcedureConfig,
    EvaluationProcedureDefinition,
    MetricExtractionConfig,
    MetricExtractionDefinition,
    MetricQuestionBinding,
    PreprocessingDefinition,
    identity_hash_for,
)
from whetstone.evaluation.aggregate import aggregation_definition
from whetstone.evaluation.metrics.blended import BoundedCompressionBlendConfig
from whetstone.evaluation.preview.persisted import load_aggregate_value
from whetstone.experiment.candidate import (
    Candidate,
    TemplateRenderContract,
    TemplateRenderKind,
)
from whetstone.experiment.reward import (
    MissingDataPolicy,
    Reward,
    RewardPolicy,
    RewardRef,
    RewardTerm,
    apply_reward_policy,
)
from whetstone.experiment.task_selection import (
    TaskSplitRoles,
    resolve_manifest_split,
)
from whetstone.optimization.proposal.mutation import MUTATION_FIELD
from whetstone.provider.policy import ProviderExecutionPolicy

ED1_ENV_NAME = "ed1"

BLENDED_METRIC_ID = "primary_score_with_bounded_compression_penalty"


class BoundedCompressionMetricConfig(BoundedCompressionBlendConfig):
    """ED1 identity-bearing blended-reward configuration."""

    metric_id: Literal["primary_score_with_bounded_compression_penalty"] = (
        BLENDED_METRIC_ID
    )

    def identity_key(self) -> str:
        """Fold ED1 metric identity with the shared blend parameters."""
        return f"{self.metric_id}|{self.blend_identity_key()}"


def ed1_blended_aggregate_values(
    store: ObjectStore,
    reward_ref: RewardRef,
) -> tuple[float | None, float | None]:
    """Load primary and compression aggregate values from a blended reward."""
    if len(reward_ref.record.evidence_refs) != 2:
        raise RuntimeError(
            "ED1 blended Reward must cite primary and compression aggregates"
        )
    return (
        load_aggregate_value(store, reward_ref.record.evidence_refs[0]),
        load_aggregate_value(store, reward_ref.record.evidence_refs[1]),
    )


@verify(UNIQUE)
class Ed1TaskModelKind(StrEnum):
    """Execution route for ED1 encoder and decoder generations."""

    DUMMY = "dummy"
    PROVIDER = "provider"


class Ed1TaskModelConfig(BaseModel):
    """Exact task-model mode, provider request, and execution policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Ed1TaskModelKind
    provider_call_config: ProviderCallConfig
    execution_policy: ProviderExecutionPolicy

    @property
    def model(self) -> str:
        """The exact provider request's model slug for display."""
        return self.provider_call_config.definition.route.model


#: The canonical enc/dec task model (same route plays both encoder + decoder).
#: ``--task-model`` overrides and folds into ``graph_hash``.
ED1_CANONICAL_MODEL = "deepseek/deepseek-v4-flash"
_ED1_CANONICAL_PROVIDER_CALL_CONFIG = build_encoder_provider_call_config(
    ED1_CANONICAL_MODEL
)

ED1_DEFAULT_BUDGET_RATIO = 0.5
ED1_DEFAULT_BLEND_CONFIG = BoundedCompressionMetricConfig()

ED1_DATASET_ID = "evalplus/humanevalplus"
ED1_DATASET_REVISION = identity_hash_for(
    schema="whetstone.humaneval.dataset_coordinate",
    payload={
        "dataset_id": ED1_DATASET_ID,
        "upstream_revision": HF_REVISION,
        "override_set": HUMANEVAL_OVERRIDE_SET.model_dump(mode="json"),
    },
)

#: The per-row metric, aggregate, and Reward-term identity for ED1 correctness.
#: Each row projects dr-code's typed submission outcome to 0.0 or 1.0; the
#: aggregate is the unweighted task mean of those values.
ED1_SUBMISSION_SCORE_NAME = "humaneval_submission_score"

ED1_BLENDED_REWARD_NAME = "blended_reward"
ED1_COMPRESSED_DESCRIPTION_LENGTH_NAME = "compressed_description_length"
ED1_COMPRESSION_NAME = "compression_ratio"

_ED1_STRATUM = "humaneval_plus"

_DEFINITION_VERSION = "1"

# --- Encoder prompt: an IMMUTABLE FRAME + a mutable instruction body ---------
#
# The encoder Mutation Surface is limited to the instruction body: the
# proposer/probe vary ONLY the instruction body; the input code, optional
# budget suffix, and terminal punctuation are a FIXED frame every candidate
# keeps by construction. The Mutation Surface payload is the BODY string;
# rendering composes ``ENCODER_FRAME.format(body=, max_budget=, input_code=)``
# so the code and budget suffix can never be dropped or mutated. Intake
# validation applies to the BODY before evaluation: a body carrying a
# ``{placeholder}`` or a code fence is a TYPED rejection.

#: The immutable encoder frame WITH a budget clause. ``{body}`` is the ONLY
#: mutable region (the instruction); the budget line and fenced Python code
#: are fixed. The ``{max_budget}`` / ``{input_code}`` placeholders live in the
#: frame, so a body never needs (or is allowed) placeholders of its own.
ENCODER_FRAME = (
    "{body}\n"
    "Use at most {max_budget} characters.\n"
    "```python\n{input_code}\n```"
)

#: The no-budget frame omits the budget line. Used when ``budget_ratio is
#: None`` to optimize compression without listing a bound; the blended
#: reward's compression term carries the pressure. Identity-folded (a
#: no-budget rollout is a distinct graph variant).
ENCODER_FRAME_NO_BUDGET = "{body}\n```python\n{input_code}\n```"

#: Encoder body A -- "concise description" (the naive floor instruction).
ENCODER_BODY_A = "Provide a concise description of the following code."

#: Encoder body B -- "compress for reconstruction by another agent" (the
#: ceiling-ish informative instruction).
ENCODER_BODY_B = (
    "Please compress the following code into a description another agent can "
    "use to reconstruct a function that behaves the same as the following "
    "code."
)


def render_encoder_frame(
    body: str, *, input_code: str, max_budget: int | None
) -> str:
    """Compose the immutable encoder frame around a mutable instruction body.

    The body is the ONLY mutable region; the input code is fixed by the frame,
    so EVERY candidate keeps it by construction. When ``max_budget`` is
    ``None`` the NO-BUDGET frame omits the budget line; otherwise the budget
    line is included. The body must NOT carry
    ``{placeholder}`` tokens (the frame owns them) -- the intake validator
    rejects such bodies before this is ever called.
    """
    if max_budget is None:
        return ENCODER_FRAME_NO_BUDGET.format(body=body, input_code=input_code)
    return ENCODER_FRAME.format(
        body=body, input_code=input_code, max_budget=max_budget
    )


#: The typed reason recorded when an ed1 encoder BODY violates the mutation
#: surface (carries a ``{placeholder}`` the frame owns, or a code fence).
ED1_INVALID_BODY = "ed1_invalid_encoder_body"

_ED1_BODY_RENDER_CONTRACT = TemplateRenderContract(
    kind=TemplateRenderKind.PYTHON_FORMAT_V1,
    available_fields=(),
)


class Ed1BodyError(ValueError):
    """A mutable ED1/D1 body violated the environment-owned frame contract."""

    code = ED1_INVALID_BODY

    def __init__(self, offending: tuple[str, ...]) -> None:
        self.offending = offending
        super().__init__(
            f"{self.code}: body contains forbidden tokens {list(offending)}"
        )


def ed1_body_rejection(body: str) -> tuple[str, ...]:
    """The offending tokens that make an ed1 encoder body invalid, else empty.

    The narrowed ed1 Mutation Surface is the INSTRUCTION only: the body
    must carry NO ``{placeholder}`` tokens (the frame owns ``{max_budget}`` /
    ``{input_code}``) and NO code fence (the body is instruction text only).
    Returns the ordered, de-duplicated offending tokens (a ``{field}`` name or
    a triple-backtick code fence); an empty tuple means the body is a clean
    instruction the frame can wrap.

    A MALFORMED brace (``'Explain {code'``) is an offending token too, not a
    bare parse error: the contract's parser raises ``ValueError`` on it, and
    letting that escape would surface as an untyped crash at eval start instead
    of the promised typed :class:`Ed1BodyError` / ``ED1_INVALID_BODY``
    rejection.
    """
    offending: list[str] = []
    seen: set[str] = set()
    try:
        placeholder_fields = _ED1_BODY_RENDER_CONTRACT.placeholder_fields(body)
    except ValueError:
        return ("{",) if "{" in body else ("}",)
    for field_name in placeholder_fields:
        token = "{" + field_name + "}"
        if token not in seen:
            seen.add(token)
            offending.append(token)
    if "```" in body and "```" not in seen:
        offending.append("```")
    return tuple(offending)


def validate_ed1_body(body: str) -> None:
    """Reject invalid body text before any provider call can be made."""
    offending = ed1_body_rejection(body)
    if offending:
        raise Ed1BodyError(offending)


#: The decoder user template (fixed; not the Mutation Surface).
DECODER_TEMPLATE = (
    "Decode the description into functional Python code. Output only Python "
    "code.\n\n{encoder_output}"
)


@dataclass(frozen=True, slots=True)
class Ed1Instance:
    """A HumanEval+ task packed for the runner as a whetstone Instance.

    ``instance`` carries the encoder INPUT_CODE (``gt_code_wo_comments``) plus
    every HumanEval field the code-eval Eval Node needs (task_id, prompt,
    canonical_solution, entry_point, test) in ``prompt_inputs``, and the
    ground-truth (with comments) in ``gold``. ``humaneval_task`` is the fully
    parsed HumanEval task (kept for the eval drive).
    """

    instance: Instance
    humaneval_task: HumanEvalTask

    @property
    def input_code(self) -> str:
        return self.instance.prompt_inputs["input_code"]

    @property
    def gt_code_wo_comments(self) -> str:
        return self.input_code


def ed1_instance_from_task(task: HumanEvalTask) -> Ed1Instance:
    """Pack one parsed HumanEval task into an ``Ed1Instance``.

    ``INPUT_CODE`` = ``task.gt_code_wo_comments`` (design's strong default:
    ``input_code = task.gt_code_wo_comments``); the compression reference is
    the
    same bytes. The HumanEval fields the sandbox needs ride in
    ``prompt_inputs``
    (all strings), so the ed1 eval drive can reconstruct the ``HumanEvalTask``
    without re-loading the dataset.
    """
    gt_wo = task.ground_truth_code_without_comments or task.ground_truth_code
    instance = Instance(
        id=task.task_id,
        seed=0,
        strata=(_ED1_STRATUM,),
        prompt_inputs={
            "input_code": gt_wo,
            "task_id": task.task_id,
            "prompt": task.prompt,
            "canonical_solution": task.canonical_solution,
            "entry_point": task.entry_point,
            "test": task.test,
        },
        gold=task.ground_truth_code,
    )
    return Ed1Instance(instance=instance, humaneval_task=task)


def humaneval_task_from_instance(instance: Instance) -> HumanEvalTask:
    """Reconstruct the parsed ``HumanEvalTask`` from an ed1 ``Instance``.

    The ed1 eval drive calls this to get the sandbox-runnable task (auto-parses
    on construction, so ``parsed``/``parsed_tests`` are populated).
    """
    pi = instance.prompt_inputs
    return HumanEvalTask(
        task_id=pi["task_id"],
        prompt=pi["prompt"],
        canonical_solution=pi["canonical_solution"],
        entry_point=pi["entry_point"],
        test=pi["test"],
    )


def load_ed1_tasks(
    *,
    snapshot_path: Path | None = None,
    limit: int | None = None,
) -> tuple[Ed1Instance, ...]:
    """Load the pinned live dataset or an explicit Whetstone snapshot."""

    tasks = load_humaneval_plus(
        prefer_snapshot=snapshot_path is not None,
        snapshot_path=snapshot_path,
    )
    if limit is not None:
        tasks = tasks[:limit]
    instances: list[Ed1Instance] = []
    for plus in tasks:
        ht = HumanEvalTask(
            task_id=plus.task_id,
            prompt=plus.prompt,
            canonical_solution=plus.canonical_solution,
            entry_point=plus.entry_point,
            test=plus.test,
        )
        instances.append(ed1_instance_from_task(ht))
    return tuple(instances)


def build_code_eval_procedure_config(
    *,
    env_name: str,
    primary_metric_name: str,
    primary_metric_settings: tuple[tuple[str, str], ...],
    zero_denominator: str = "not_applicable",
) -> EvaluationProcedureConfig:
    """Build one enc-dec code-eval procedure with a concrete primary metric.

    ED1 and ED1M share the compression question and procedure shape, but their
    primary row metrics are different contracts. ``env_name`` keeps their
    preprocessing, extraction, procedure, and resolved-operator identities
    distinct while ``primary_metric_name`` makes the row metric explicit.
    """
    definition = MetricExtractionDefinition(
        definition_id=f"whetstone.{env_name}.code_eval",
        version=_DEFINITION_VERSION,
        questions=(
            MetricQuestionBinding(
                metric=primary_metric_name,
                on="submission",
                settings=primary_metric_settings,
            ),
            MetricQuestionBinding(
                metric="whetstone.ed1.compression_ratio",
                on="description",
                settings=(
                    ("zstd_level", "19"),
                    ("reference", "task.gt_code_wo_comments"),
                ),
            ),
        ),
    )
    metric_extraction = MetricExtractionConfig._create(
        definition=definition,
        assignment={},
        resolved_operators=(
            (f"whetstone.{env_name}.code_eval_operator", "1"),
        ),
    )
    preprocessing = PreprocessingDefinition(
        definition_id=f"whetstone.{env_name}.preprocess",
        version=_DEFINITION_VERSION,
        steps=(),
    ).materialize()
    return EvaluationProcedureDefinition(
        definition_id=f"whetstone.{env_name}.procedure",
        version=_DEFINITION_VERSION,
    ).materialize(
        preprocessing=preprocessing,
        metric_extraction=metric_extraction,
        assignment={"zero_denominator": zero_denominator},
    )


def build_ed1_procedure_config(
    *, zero_denominator: str = "not_applicable"
) -> EvaluationProcedureConfig:
    """The canonical ED1 HumanEval-submission evaluation procedure."""
    return build_code_eval_procedure_config(
        env_name=ED1_ENV_NAME,
        primary_metric_name=ED1_SUBMISSION_SCORE_NAME,
        primary_metric_settings=(
            ("dataset", ED1_DATASET_ID),
            ("dataset_coordinate", ED1_DATASET_REVISION),
            ("upstream_revision", HF_REVISION),
            ("scorer", "dr_code.humaneval.score_humaneval_submission"),
            ("scoring_profile_id", ED1_SCORING_PROFILE_ID),
            ("scoring_profile_version", ED1_SCORING_PROFILE_VERSION),
            ("completed_outcome_projection", "definitive_score"),
        ),
        zero_denominator=zero_denominator,
    )


def _ed1_split(
    *,
    env_name: str = ED1_ENV_NAME,
    dataset_revision: str,
    split_role: str,
    instances: tuple[Instance, ...],
    procedure: EvaluationProcedureConfig,
    completeness: Completeness,
    max_skip_fraction: float,
    repeats: int,
    manifest_tag: str | None = None,
) -> EnvSplitSampling:
    policy = completeness.to_policy(max_skip_fraction=max_skip_fraction)
    aggregation = aggregation_definition(
        f"whetstone.{env_name}.aggregation"
    ).materialize(
        {
            "reduction": "mean",
            "missing_data": policy.missing_data,
            "zero_denominator": "not_applicable",
            "max_skip_fraction": policy.skip_fraction_token(),
        }
    )
    namespace = f"whetstone.{env_name}"
    if manifest_tag is not None:
        namespace = f"{namespace}.{manifest_tag}"
    return derive_split_sampling(
        namespace=namespace,
        dataset_revision=dataset_revision,
        split_role=split_role,
        instances=instances,
        task_identity_of=lambda instance: str(instance.id),
        repeats=repeats,
        procedure=procedure,
        aggregation=aggregation,
    )


def _ed1_candidate(*, candidate_id: str, body: str) -> Candidate:
    # The Mutation Surface payload is the INSTRUCTION BODY only; the code,
    # budget suffix, and punctuation are composed by the immutable frame.
    return Candidate(
        candidate_id=candidate_id,
        base_ref=env_candidate_base_ref(ED1_ENV_NAME),
        payload={MUTATION_FIELD: body},
    )


def ed1_initial_candidate() -> Candidate:
    return _ed1_candidate(
        candidate_id=f"{ED1_ENV_NAME}-naive", body=ENCODER_BODY_A
    )


def ed1_ceiling_candidate() -> Candidate:
    return _ed1_candidate(
        candidate_id=f"{ED1_ENV_NAME}-ceiling", body=ENCODER_BODY_B
    )


def reward_from_primary_score(
    policy: RewardPolicy,
    *,
    primary_score: float | None,
    evidence_refs: tuple[TypedRef, ...],
) -> Reward:
    """Apply a one-term environment policy to its internal primary score."""
    from whetstone.envs.reward import CandidateEvaluationFailure

    if len(policy.terms) != 1:
        raise ValueError(
            "primary-score Reward Policy must have exactly one term"
        )
    metric_name = policy.terms[0].name
    try:
        return apply_reward_policy(
            policy,
            aggregates={metric_name: primary_score},
            evidence_role=EvaluationRole.INTERNAL,
            evidence_refs=evidence_refs,
        )
    except ValueError as exc:
        raise CandidateEvaluationFailure(
            "internal candidate has no computable Reward: the "
            f"{metric_name!r} aggregate is missing/incomplete under the "
            f"FAIL missing-data policy (primary_score={primary_score!r})"
        ) from exc


def build_ed1_blended_reward_policy(
    blend_config: BoundedCompressionMetricConfig,
    *,
    env_name: str = ED1_ENV_NAME,
) -> RewardPolicy:
    """An ED1-family blended Reward Policy with one blended-reward term.

    A single unit-weight, maximize term over the pre-computed per-task-blended
    aggregate (:data:`ED1_BLENDED_REWARD_NAME`). The blend config's id key
    and concrete environment fold into the policy name, so ED1 and ED1M never
    share a policy identity.
    """
    return RewardPolicy(
        policy_name=(
            f"whetstone.env.{env_name}.blended_reward"
            f"|{blend_config.identity_key()}"
        ),
        reward_name="reward",
        terms=(
            RewardTerm(
                name=ED1_BLENDED_REWARD_NAME, weight=1.0, maximize=True
            ),
        ),
        missing_data=MissingDataPolicy.FAIL,
    )


def ed1_reward_from_blended(
    blend_config: BoundedCompressionMetricConfig,
    *,
    env_name: str,
    blended: float | None,
    evidence_refs: tuple[TypedRef, ...],
) -> Reward:
    """Apply the blended Reward Policy to the mean per-task blended reward.

    ``blended`` is the count-weighted mean of the per-task blended rewards (the
    aggregate certification value). A missing value under FAIL surfaces as a
    typed :class:`CandidateEvaluationFailure` (candidate marked failed).
    """
    from whetstone.envs.reward import CandidateEvaluationFailure

    policy = build_ed1_blended_reward_policy(
        blend_config,
        env_name=env_name,
    )
    try:
        return apply_reward_policy(
            policy,
            aggregates={ED1_BLENDED_REWARD_NAME: blended},
            evidence_role=EvaluationRole.INTERNAL,
            evidence_refs=evidence_refs,
        )
    except ValueError as exc:
        raise CandidateEvaluationFailure(
            "ed1 internal candidate has no computable blended Reward: the "
            f"{ED1_BLENDED_REWARD_NAME!r} aggregate is missing under FAIL "
            f"(blended={blended!r})"
        ) from exc


@dataclass(frozen=True, slots=True)
class Ed1Experiment(EnvExperiment):
    """An ``EnvExperiment`` carrying the ed1-specific enc-dec rollout + tasks.

    Adds the enc-dec :class:`EncDecRolloutDefinition` (a 3-node graph, with the
    ``budget_ratio`` folded into ``graph_hash``) on top of the base experiment
    shape the runner reads. ``rollout_definition`` (the base field) is set to
    the
    same enc-dec rollout so ``experiment.rollout_definition.graph_hash`` etc.
    resolve for the runner.
    """

    encdec_rollout: EncDecRolloutDefinition | None = None
    #: The per-task Character Budget ratio, or ``None`` for the no-budget frame
    #: without a "Use at most N characters" line or MAX_BUDGET.
    #: ``None`` is the default for ed1 optimizer cells to optimize compression
    #: without listing a budget at all; the reward's
    #: compression term carries the pressure instead.
    budget_ratio: float | None = ED1_DEFAULT_BUDGET_RATIO
    dataset_revision: str = ED1_DATASET_REVISION
    #: The injectable code scorer (raw_submission, task) -> CodeScore. The
    #: scorer is INJECTED by the caller that drives rows; the production
    #: injection is :func:`whetstone.envs.ed1_scoring.score_ed1_submission`,
    #: which runs candidate code through the caller's explicit dr-exec
    #: executor.
    scorer: Callable[..., CodeScore] | None = None
    #: ED1 always uses this per-task blend for internal selection and the
    #: official comparison vector; primary score + compression are still
    #: reported separately. The optional type is required only because ED1M
    #: shares this runtime model and retains its independent reward behavior.
    blend_config: BoundedCompressionMetricConfig | None = field(
        default_factory=BoundedCompressionMetricConfig
    )

    def __post_init__(self) -> None:
        if self.env_name == ED1_ENV_NAME and self.blend_config is None:
            raise ValueError("ED1 requires a bounded compression blend config")


def build_ed1_experiment(
    *,
    provider_call_config: ProviderCallConfig = (
        _ED1_CANONICAL_PROVIDER_CALL_CONFIG
    ),
    budget_ratio: float | None = ED1_DEFAULT_BUDGET_RATIO,
    scorer: Callable[..., CodeScore] | None = None,
    snapshot_path: Path | None = None,
    limit: int | None = None,
    internal_n: int | None = None,
    official_n: int | None = None,
    completeness: Completeness = Completeness.PROPAGATE,
    max_skip_fraction: float = 0.0,
    repeats: int = 3,
    tasks: tuple[Ed1Instance, ...] | None = None,
    exclude_task_ids: frozenset[str] | None = None,
    blend_config: BoundedCompressionMetricConfig = ED1_DEFAULT_BLEND_CONFIG,
    split_manifest: TaskSplitRoles | None = None,
) -> Ed1Experiment:
    """Build the ed1 enc-dec experiment the runner cell consumes.

    Loads the pinned HumanEval+ pool (or uses injected ``tasks`` for tests),
    splits it into internal/official (first-N ordered), builds the 3-node
    enc-dec rollout at ``budget_ratio`` (folded into ``graph_hash``), the naive
    (A) + ceiling (B) encoder candidates, and the two Eval Configs sharing the
    code-eval Procedure identity. ED1 always advertises and applies the
    per-task bounded-compression blend; callers may configure its weight and
    bounds but cannot disable it.

    ``exclude_task_ids`` drops those task ids from the pool before the split:
    excluded tasks are removed from the train / eval / test (internal /
    official / held-out) pools. The exclusion applies to the
    ordered pool, so the filtered Task Set is deterministic; because each
    split's Task Set identity folds its task ids, a filtered pool yields a
    DISTINCT ``eval_config_hash`` per split -- the exclusion folds into the id
    by construction. The caller passes the exclusion list for the model the
    cell actually runs.

    ``split_manifest`` overrides the first-N slice with role-true
    train/val/test semantics: the internal split = the manifest's
    ``train + val`` ids (by MEMBERSHIP, in manifest order -- the internal
    machinery has no val sub-split, so val folds into internal alongside
    train); the official split = the manifest's ``test`` ids EXACTLY
    (membership, NOT a first-N slice).
    ``official_n`` then caps WITHIN the test set. Mutually exclusive with
    ``exclude_task_ids`` (the caller enforces the CLI refusal). The manifest's
    content hash + pool folds into each split's Task Set identity.
    """
    if not isinstance(blend_config, BoundedCompressionMetricConfig):
        raise TypeError("ED1 requires a bounded compression blend config")
    pool = (
        tasks
        if tasks is not None
        else load_ed1_tasks(snapshot_path=snapshot_path, limit=limit)
    )
    if exclude_task_ids:
        pool = tuple(
            t for t in pool if str(t.instance.id) not in exclude_task_ids
        )
    if not pool:
        raise ValueError("ed1 task pool is empty")
    procedure = build_ed1_procedure_config()
    rollout = build_encdec_rollout_definition(
        ED1_ENV_NAME,
        provider_call_config=provider_call_config,
        procedure_config_hash=procedure.config_identity_hash,
        budget_ratio=budget_ratio,
    )
    manifest_tag: str | None = None
    if split_manifest is not None:
        resolved = resolve_manifest_split(
            roles=split_manifest,
            items=pool,
            id_of=lambda t: str(t.instance.id),
            official_n=official_n,
        )
        internal_instances = tuple(t.instance for t in resolved.internal)
        official_instances = tuple(t.instance for t in resolved.official)
        manifest_tag = resolved.manifest_tag
        if resolved.official_capped:
            print(f"[ed1] {resolved.official_capped}")
    else:
        all_instances = tuple(t.instance for t in pool)
        n = len(all_instances)
        # First-N ordered split: internal then official (disjoint, contiguous).
        # A small pool may put all tasks in the official split.
        i_n = internal_n if internal_n is not None else min(max(1, n // 2), n)
        internal_instances = all_instances[:i_n]
        rest = all_instances[i_n:]
        o_n = official_n if official_n is not None else len(rest)
        official_instances = (
            rest[:o_n] if rest else internal_instances[: o_n or n]
        )
        if not official_instances:
            official_instances = internal_instances
    internal_split = _ed1_split(
        dataset_revision=ED1_DATASET_REVISION,
        split_role="internal_eval",
        instances=internal_instances,
        procedure=procedure,
        completeness=completeness,
        max_skip_fraction=max_skip_fraction,
        repeats=repeats,
        manifest_tag=manifest_tag,
    )
    official_split = _ed1_split(
        dataset_revision=ED1_DATASET_REVISION,
        split_role="official",
        instances=official_instances,
        procedure=procedure,
        completeness=completeness,
        max_skip_fraction=max_skip_fraction,
        repeats=repeats,
        manifest_tag=manifest_tag,
    )
    eval_configs = EnvEvalConfigs(
        env_name=ED1_ENV_NAME,
        procedure_config_hash=procedure.config_identity_hash,
        internal=internal_split,
        official=official_split,
        held_out_task_identities=(),
    )
    return Ed1Experiment(
        env_name=ED1_ENV_NAME,
        rollout_definition=rollout,  # type: ignore[arg-type]
        initial_candidate=ed1_initial_candidate(),
        ceiling_candidate=ed1_ceiling_candidate(),
        eval_configs=eval_configs,
        reward_policy=build_ed1_blended_reward_policy(
            blend_config, env_name=ED1_ENV_NAME
        ),
        completeness_policy=completeness.to_policy(
            max_skip_fraction=max_skip_fraction
        ),
        encdec_rollout=rollout,
        budget_ratio=budget_ratio,
        dataset_revision=ED1_DATASET_REVISION,
        scorer=scorer,
        blend_config=blend_config,
    )


def build_ed1_preview_engine(
    *,
    store: ObjectStore,
    experiment: Ed1Experiment,
    task_model: Ed1TaskModelConfig,
    batch_scorer: CodeBatchScorer,
    concurrency: int = 1,
    partial_log: PartialLog | None = None,
    prompt_cache: PromptResultCache | None = None,
) -> EvaluationEngine:
    """Build one ED1 preview evaluation engine for matrix and debug scripts."""

    from whetstone.evaluation.drivers.ed1_row_jobs import (
        ed1_task_model_row_job,
    )
    from whetstone.evaluation.engine import EvaluationEngine

    return EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.internal,
        execution_policy=task_model.execution_policy,
        row_job_factory=ed1_task_model_row_job(task_model),
        concurrency=concurrency,
        partial_log=partial_log,
        prompt_cache=prompt_cache,
        batch_scorer=batch_scorer,
    )


def ed1_preview_metadata(
    *,
    task_model: Ed1TaskModelConfig,
    runtime: Ed1ScoringRuntimeSummary,
    blend_config: BoundedCompressionMetricConfig,
) -> PreviewMetadata:
    """Persist ED1-specific preview fields alongside generic transcripts."""

    return PreviewMetadata.model_validate(
        {
            "task_model": task_model.model_dump(mode="json"),
            "runtime": runtime.model_dump(mode="json"),
            "blend_config": blend_config.model_dump(mode="json"),
        }
    )


def ed1_task_model_from_metadata(
    metadata: PreviewMetadata,
) -> Ed1TaskModelConfig:
    return Ed1TaskModelConfig.model_validate(
        metadata.model_dump(mode="python")["task_model"]
    )


def ed1_runtime_from_metadata(
    metadata: PreviewMetadata,
) -> Ed1ScoringRuntimeSummary:
    return Ed1ScoringRuntimeSummary.model_validate(
        metadata.model_dump(mode="python")["runtime"]
    )


def ed1_blend_config_from_metadata(
    metadata: PreviewMetadata,
) -> BoundedCompressionMetricConfig:
    return BoundedCompressionMetricConfig.model_validate(
        metadata.model_dump(mode="python")["blend_config"]
    )


def _selected_ed1_tasks(
    tasks: tuple[Ed1Instance, ...],
    task_ids: tuple[str, ...],
) -> tuple[Ed1Instance, ...]:
    if not task_ids:
        raise ValueError("baseline preview requires at least one task ID")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("baseline preview task IDs must be unique")
    by_id = {task.humaneval_task.task_id: task for task in tasks}
    missing = tuple(task_id for task_id in task_ids if task_id not in by_id)
    if missing:
        raise ValueError(f"baseline preview task IDs are unknown: {missing}")
    return tuple(by_id[task_id] for task_id in task_ids)


def run_ed1_anchor_baseline_preview(
    *,
    store: ObjectStore,
    tasks: tuple[Ed1Instance, ...],
    task_ids: tuple[str, ...],
    pool_ceiling: int,
    task_model: Ed1TaskModelConfig,
    batch_scorer: CodeBatchScorer,
    runtime: Ed1ScoringRuntimeSummary,
    task_selection: TaskRoleSelection | None = None,
    preflight_task: Ed1Instance | None = None,
    budget_ratio: float | None = None,
    concurrency: int = 1,
    partial_log: PartialLog | None = None,
    prompt_cache: PromptResultCache | None = None,
    repeats: int = 1,
    blend_config: BoundedCompressionMetricConfig = ED1_DEFAULT_BLEND_CONFIG,
    power_config: PowerConfig | None = None,
    bootstrap_level: float = 0.95,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
    log: Callable[[str], None] | None = None,
) -> BaselinePreviewTranscript:
    """Wire ED1 execution and run the generic anchor baseline preview."""

    from whetstone.evaluation.preview.anchor import run_baseline_preview
    from whetstone.evaluation.preview.binding import preview_evaluation_binding

    selected = _selected_ed1_tasks(tasks, task_ids)
    budget_label = (
        "unbudgeted"
        if budget_ratio is None
        else f"budget ratio {budget_ratio:g}"
    )
    if log is not None:
        log(f"{budget_label}: starting scoring-runtime preflight")
    preflight = run_ed1_scoring_preflight(
        (preflight_task or selected[0],), batch_scorer
    )
    if log is not None:
        log(
            f"{budget_label}: scoring-runtime preflight completed "
            f"({preflight.task_id}: {preflight.outcome})"
        )
    experiment = build_ed1_experiment(
        provider_call_config=task_model.provider_call_config,
        budget_ratio=budget_ratio,
        tasks=tasks,
        internal_n=len(tasks),
        official_n=len(tasks),
        repeats=repeats,
        blend_config=blend_config,
    )
    engine = build_ed1_preview_engine(
        store=store,
        experiment=experiment,
        task_model=task_model,
        batch_scorer=batch_scorer,
        concurrency=concurrency,
        partial_log=partial_log,
        prompt_cache=prompt_cache,
    )
    binding = preview_evaluation_binding(
        engine,
        campaign="ed1-baseline-preview",
        provenance_note=(
            f"{task_model.kind.value}-generation-real-humaneval-scoring"
        ),
        environment_fingerprint=ed1_environment_fingerprint(runtime),
    )
    return run_baseline_preview(
        store=store,
        engine=engine,
        evaluation_binding=binding,
        baseline_candidate=experiment.initial_candidate,
        ceiling_candidate=experiment.ceiling_candidate,
        task_ids=task_ids,
        pool_ceiling=pool_ceiling,
        preflight=preflight,
        metadata=ed1_preview_metadata(
            task_model=task_model,
            runtime=runtime,
            blend_config=blend_config,
        ),
        task_selection=task_selection,
        budget_ratio=budget_ratio,
        concurrency=concurrency,
        baseline_purpose="ed1-calibration-baseline",
        ceiling_purpose="ed1-calibration-ceiling",
        power_config=power_config,
        bootstrap_level=bootstrap_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        log=log,
    )


def run_ed1_anchor_baseline_sweep(
    *,
    store: ObjectStore,
    tasks: tuple[Ed1Instance, ...],
    task_ids: tuple[str, ...],
    pool_ceiling: int,
    task_model: Ed1TaskModelConfig,
    batch_scorer: CodeBatchScorer,
    runtime: Ed1ScoringRuntimeSummary,
    budget_ratios: tuple[float | None, ...],
    task_selection: TaskRoleSelection | None = None,
    preflight_task: Ed1Instance | None = None,
    concurrency: int = 1,
    partial_log: PartialLog | None = None,
    prompt_cache: PromptResultCache | None = None,
    repeats: int = 1,
    blend_config: BoundedCompressionMetricConfig = ED1_DEFAULT_BLEND_CONFIG,
    power_config: PowerConfig | None = None,
    bootstrap_level: float = 0.95,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
    log: Callable[[str], None] | None = None,
) -> BaselineSweepTranscript:
    """Evaluate both ED1 anchors under each requested budget framing."""

    from whetstone.evaluation.preview.anchor import run_baseline_sweep

    return run_baseline_sweep(
        preview_factory=lambda budget_ratio: run_ed1_anchor_baseline_preview(
            store=store,
            tasks=tasks,
            task_ids=task_ids,
            task_selection=task_selection,
            preflight_task=preflight_task,
            pool_ceiling=pool_ceiling,
            task_model=task_model,
            batch_scorer=batch_scorer,
            runtime=runtime,
            budget_ratio=budget_ratio,
            concurrency=concurrency,
            partial_log=partial_log,
            prompt_cache=prompt_cache,
            repeats=repeats,
            blend_config=blend_config,
            power_config=power_config,
            bootstrap_level=bootstrap_level,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            log=log,
        ),
        task_ids=task_ids,
        budget_ratios=budget_ratios,
        task_selection=task_selection,
    )


def run_ed1_copro_scoring_preview(
    *,
    store: ObjectStore,
    tasks: tuple[Ed1Instance, ...],
    sweep: Ed1CoproSweepRanges,
    proposer_kind: str,
    proposer_config: ProposerRouteConfig,
    proposer_transport: ProposerTransport,
    task_model: Ed1TaskModelConfig,
    batch_scorer: CodeBatchScorer,
    runtime: Ed1ScoringRuntimeSummary,
    task_selection: TaskRoleSelection | None = None,
    preflight_task: Ed1Instance | None = None,
    concurrency: int = 1,
    repeats: int = 1,
    blend_config: BoundedCompressionMetricConfig = ED1_DEFAULT_BLEND_CONFIG,
    proposal_observer: Callable[[Ed1CoproRoundAttempt], None] | None = None,
    candidate_observer: Callable[[CandidateProgress], None] | None = None,
) -> ScoringTranscript:
    """Wire ED1 execution and run the generic COPRO scoring preview."""

    from whetstone.evaluation.preview.binding import preview_evaluation_binding
    from whetstone.optimization.copro.ed1_dry_run import Ed1CoproPreviewTask
    from whetstone.optimization.copro.scoring_preview import (
        run_copro_scoring_preview,
    )

    if not tasks:
        raise ValueError("COPRO scoring preview requires at least one task")
    if repeats < 1:
        raise ValueError("COPRO scoring preview repeats must be positive")
    task_ids = tuple(task.humaneval_task.task_id for task in tasks)
    preflight = run_ed1_scoring_preflight(
        (preflight_task or tasks[0],), batch_scorer
    )
    first = tasks[0]
    preview_task = Ed1CoproPreviewTask(
        task_id=first.humaneval_task.task_id,
        input_code=first.input_code,
    )

    def _experiment_for(settings: Ed1CoproSweepPoint) -> Ed1Experiment:
        return build_ed1_experiment(
            provider_call_config=task_model.provider_call_config,
            budget_ratio=settings.budget_ratio,
            tasks=tasks,
            internal_n=len(tasks),
            official_n=len(tasks),
            repeats=repeats,
            blend_config=blend_config,
        )

    def engine_factory(settings: Ed1CoproSweepPoint) -> EvaluationEngine:
        return build_ed1_preview_engine(
            store=store,
            experiment=_experiment_for(settings),
            task_model=task_model,
            batch_scorer=batch_scorer,
            concurrency=concurrency,
        )

    def binding_factory(
        engine: EvaluationEngine,
        settings: Ed1CoproSweepPoint,
    ) -> EvaluationBinding:
        del settings
        return preview_evaluation_binding(
            engine,
            campaign="copro-scoring-preview",
            provenance_note=(
                f"{task_model.kind.value}-generation-real-humaneval-scoring"
            ),
            environment_fingerprint=ed1_environment_fingerprint(runtime),
        )

    def initial_candidate_factory(settings: Ed1CoproSweepPoint) -> Candidate:
        return _experiment_for(settings).initial_candidate

    return run_copro_scoring_preview(
        store=store,
        sweep=sweep,
        preview_task=preview_task,
        proposer_kind=proposer_kind,
        proposer_config=proposer_config,
        proposer_transport=proposer_transport,
        engine_factory=engine_factory,
        binding_factory=binding_factory,
        initial_candidate_factory=initial_candidate_factory,
        aggregate_values_fn=ed1_blended_aggregate_values,
        preflight=preflight,
        metadata=ed1_preview_metadata(
            task_model=task_model,
            runtime=runtime,
            blend_config=blend_config,
        ),
        task_ids=task_ids,
        task_selection=task_selection,
        concurrency=concurrency,
        proposal_observer=proposal_observer,
        candidate_observer=candidate_observer,
    )


#: Callable type for reconstructing a HumanEvalTask (test injection point).
HumanEvalTaskFromInstance = Callable[[Instance], HumanEvalTask]

__all__ = [
    "BLENDED_METRIC_ID",
    "DECODER_TEMPLATE",
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
    "BoundedCompressionMetricConfig",
    "Ed1BodyError",
    "Ed1Experiment",
    "Ed1Instance",
    "Ed1TaskModelConfig",
    "Ed1TaskModelKind",
    "build_code_eval_procedure_config",
    "build_ed1_experiment",
    "build_ed1_preview_engine",
    "build_ed1_procedure_config",
    "ed1_blend_config_from_metadata",
    "ed1_body_rejection",
    "ed1_ceiling_candidate",
    "ed1_initial_candidate",
    "ed1_instance_from_task",
    "ed1_preview_metadata",
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
