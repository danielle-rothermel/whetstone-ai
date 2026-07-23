"""The ``ed1`` encoder-decoder HumanEval compression env binding.

``ed1`` is the runner env for the enc-dec compression experiment (scope:
``reports/encdec-scope.md``). Unlike the QA envs (a deterministic string
oracle),
ed1's rollout is a three-node Encoder -> Decoder -> Eval graph
(:mod:`whetstone.envs.encdec_rollout`) whose Eval Node runs the dr-code
HumanEval
test sandbox (Binary Test Pass Score on the DECODER output) plus the whetstone
zstd-19 compression scoring (Compression Ratio on the ENCODER output vs the
ground-truth code).

This module owns the ed1 experiment binding that plugs into the runner:

* the HumanEval+ task pool (wrapping dr-code's ``load_humaneval_plus``), each
  task wrapped as a whetstone :class:`Instance` carrying ``INPUT_CODE`` (=
  ``task.gt_code_wo_comments``) plus the HumanEval fields the code-eval Eval
  Node
  needs, with the pinned dataset revision recorded;
* the two encoder :class:`ProbeSurface` templates (naive "concise" A and the
  ceiling-ish "compress for reconstruction" B, verbatim from
  ``design/eval-run.html``) -- the encoder ``user_prompt_template`` is the
  Mutation Surface optimizers mutate;
* :func:`build_ed1_experiment`, an
:class:`~whetstone.envs.factory.EnvExperiment`
  the runner cell consumes (its rollout is the enc-dec 3-node graph, its eval
  path is the code-eval drive in :mod:`whetstone.envs.ed1_eval`).

The canonical enc/dec task model is ``deepseek/deepseek-v4-flash`` (same route
plays BOTH encoder and decoder); ``--task-model`` overrides per cell and folds
into ``graph_hash``. ``budget_ratio`` (default 0.5) is CLI-visible
(``--budget-ratio``) and folds into ``graph_hash`` via the Character Budget
rule.

Optimizer Reward = pass-rate ONLY for now; the Mean Compression Ratio is
REPORTED alongside (dual scores in the cell line + traces + sidecars). Full
dual-objective / Pareto selection is a FLAGGED follow-up, not built here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from dr_code.eval import (
    EvalDefinition,
    EvaluationProcedureConfig,
    EvaluationProcedureDefinition,
    MetricExtractionConfig,
    MetricExtractionDefinition,
    MetricQuestionBinding,
    PreprocessingDefinition,
    RepeatPlan,
    SamplingDefinition,
    TaskSet,
)
from dr_code.humaneval import HumanEvalTask
from dr_code.synthetic.humaneval_loader import HF_REVISION, load_humaneval_plus
from whetstone_envs.core import Instance

from whetstone.code_eval.aggregate import aggregation_definition
from whetstone.envs.ed1_scoring import CodeScore
from whetstone.envs.encdec_rollout import (
    EncDecRolloutDefinition,
    build_encdec_rollout_definition,
)
from whetstone.envs.factory import EnvExperiment
from whetstone.envs.sampling import (
    Completeness,
    EnvEvalConfigs,
    EnvSplitSampling,
)
from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization.mutation import (
    MUTATION_FIELD,
    template_placeholder_fields,
)
from whetstone.optimization.reward import (
    MissingDataPolicy,
    Reward,
    RewardPolicy,
    RewardTerm,
    apply_reward_policy,
)
from whetstone.optimization.schema import Candidate

#: The ed1 env id.
ED1_ENV_NAME = "ed1"

#: The canonical enc/dec task model (same route plays both encoder + decoder).
#: Design + directive: ``deepseek/deepseek-v4-flash`` (NOT nano).
#: ``--task-model``
#: overrides and folds into ``graph_hash``.
ED1_CANONICAL_MODEL = "deepseek/deepseek-v4-flash"

#: The default budget ratio (``--budget-ratio``). Folds into ``graph_hash``.
ED1_DEFAULT_BUDGET_RATIO = 0.5

#: The pinned HumanEval+ dataset revision (dr-code ``HF_REVISION``), recorded
#: so
#: the ed1 Task Set is a deterministic, reproducible, offline-loadable
#: identity.
ED1_DATASET_REVISION = HF_REVISION
ED1_DATASET_ID = "evalplus/humanevalplus"

#: The reward-term / internal aggregate name: the Binary Test Pass rate. The
#: optimizer Reward is pass-rate ONLY; compression is reported, not rewarded.
ED1_PASS_RATE_NAME = "binary_test_pass"
#: The compression aggregate name reported alongside (never the Reward).
ED1_COMPRESSION_NAME = "compression_ratio"

#: The ed1 single stratum label (HumanEval+ is not stratified).
_ED1_STRATUM = "humaneval_plus"

_DEFINITION_VERSION = "1"

# --- Encoder prompt: an IMMUTABLE FRAME + a mutable strategy body ------------
#
# The encoder Mutation Surface is NARROWED (user directive, task 17): the
# proposer/probe vary ONLY the leading STRATEGY-SENTENCE body; the budget
# clause and the fenced code block are a FIXED frame every candidate keeps by
# construction. The Mutation Surface payload is the BODY string; rendering
# composes ``ENCODER_FRAME.format(body=, max_budget=, input_code=)`` so the
# budget line + code block can never be dropped or mutated. Intake validation
# applies to the BODY (see runner.optimizers): a body carrying a
# ``{placeholder}`` or a code fence is a TYPED rejection (the frame owns them).

#: The immutable encoder frame. ``{body}`` is the ONLY mutable region (the
#: strategy sentence); the budget clause + fenced code block are fixed. The
#: ``{max_budget}`` / ``{input_code}`` placeholders live in the frame, so a
#: body never needs (or is allowed) placeholders of its own.
ENCODER_FRAME = (
    "{body}\n"
    "Use at most {max_budget} characters.\n"
    "```python\n"
    "{input_code}\n"
    "```"
)

#: Encoder body A -- "concise description" (the naive floor strategy sentence).
ENCODER_BODY_A = "Provide a concise description of the following code."

#: Encoder body B -- "compress for reconstruction by another agent" (the
#: ceiling-ish informative strategy sentence).
ENCODER_BODY_B = (
    "Compress the following code into a description another agent can use\n"
    "to reconstruct a function with the same behavior."
)


def render_encoder_frame(
    body: str, *, input_code: str, max_budget: int
) -> str:
    """Compose the immutable encoder frame around a mutable strategy body.

    The body is the ONLY mutable region; the budget clause + fenced code block
    are fixed by the frame, so EVERY candidate keeps them by construction. The
    body must NOT carry ``{placeholder}`` tokens (the frame owns them) -- the
    intake validator rejects such bodies before this is ever called.
    """
    return ENCODER_FRAME.format(
        body=body, input_code=input_code, max_budget=max_budget
    )


#: The typed reason recorded when an ed1 encoder BODY violates the mutation
#: surface (carries a ``{placeholder}`` the frame owns, or a code fence).
ED1_INVALID_BODY = "ed1_invalid_encoder_body"


def ed1_body_rejection(body: str) -> tuple[str, ...]:
    """The offending tokens that make an ed1 encoder body invalid, else empty.

    The narrowed ed1 Mutation Surface is the STRATEGY SENTENCE only: the body
    must carry NO ``{placeholder}`` tokens (the frame owns ``{max_budget}`` /
    ``{input_code}``) and NO code fence (the frame owns the fenced code block).
    Returns the ordered, de-duplicated offending tokens (a ``{field}`` name or
    a triple-backtick code fence); an empty tuple means the body is a clean
    strategy sentence the frame can wrap.
    """
    offending: list[str] = []
    seen: set[str] = set()
    for field_name in template_placeholder_fields(body):
        token = "{" + field_name + "}"
        if token not in seen:
            seen.add(token)
            offending.append(token)
    if "```" in body and "```" not in seen:
        offending.append("```")
    return tuple(offending)


#: Back-compat: the fully-rendered naive/ceiling templates (frame + body), for
#: any reader that still wants the whole encoder prompt shape. The Mutation
#: Surface itself now carries only the body.
ENCODER_TEMPLATE_A = ENCODER_FRAME.format(
    body=ENCODER_BODY_A, max_budget="{max_budget}", input_code="{input_code}"
)
ENCODER_TEMPLATE_B = ENCODER_FRAME.format(
    body=ENCODER_BODY_B, max_budget="{max_budget}", input_code="{input_code}"
)

#: The decoder user template (fixed; not the Mutation Surface).
DECODER_TEMPLATE = (
    "Decode the description into functional Python code. Output only Python "
    "code.\n"
    "{encoder_output}"
)


@dataclass(frozen=True, slots=True)
class Ed1Instance:
    """A HumanEval+ task packed for the runner as a whetstone Instance.

    ``instance`` carries the encoder INPUT_CODE (``gt_code_wo_comments``) plus
    every HumanEval field the code-eval Eval Node needs (task_id, prompt,
    canonical_solution, entry_point, test) in ``prompt_inputs``, and the
    ground-truth (with comments) in ``gold``. ``humaneval_task`` is the fully
    parsed dr-code task (kept for the eval drive).
    """

    instance: Instance
    humaneval_task: HumanEvalTask

    @property
    def input_code(self) -> str:
        """The encoder INPUT_CODE (``gt_code_wo_comments``)."""
        return self.instance.prompt_inputs["input_code"]

    @property
    def gt_code_wo_comments(self) -> str:
        """The compression-reference ground-truth code (without comments)."""
        return self.input_code


def _humaneval_to_instance(task: HumanEvalTask) -> Ed1Instance:
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
    prefer_snapshot: bool = True,
    limit: int | None = None,
) -> tuple[Ed1Instance, ...]:
    """Load the pinned HumanEval+ pool as ordered, deterministic ed1 instances.

    ``prefer_snapshot`` (default True) loads the committed offline snapshot so
    the pilot + tests need no network. ``limit`` takes the first-N tasks (a
    fixed
    ordered slice) for the minimal pilot. The dataset revision is
    :data:`ED1_DATASET_REVISION`.
    """
    tasks = load_humaneval_plus(prefer_snapshot=prefer_snapshot)
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
        instances.append(_humaneval_to_instance(ht))
    return tuple(instances)


def _ed1_metric_extraction_config() -> MetricExtractionConfig:
    """The ed1 code-eval Metric Extraction Config (folds in the eval wiring).

    Two Metric Questions -- the Binary Test Pass Score on the DECODER output
    and
    the Compression Ratio on the ENCODER output -- naming the ed1 code-eval
    operator. The identity folds the operator name/version so a change of the
    eval wiring is visible in ``eval_config_hash`` / ``graph_hash``.
    """
    definition = MetricExtractionDefinition(
        definition_id="whetstone.ed1.code_eval",
        version=_DEFINITION_VERSION,
        questions=(
            MetricQuestionBinding(
                metric="whetstone.ed1.binary_test_pass",
                on="submission",
                settings=(
                    ("dataset", ED1_DATASET_ID),
                    ("revision", ED1_DATASET_REVISION),
                    ("scorer", "dr_code.humaneval.score_humaneval_submission"),
                ),
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
    return MetricExtractionConfig._create(
        definition=definition,
        assignment={},
        resolved_operators=(
            ("whetstone.ed1.code_eval_operator", "1"),
        ),
    )


def build_ed1_procedure_config(
    *, zero_denominator: str = "not_applicable"
) -> EvaluationProcedureConfig:
    """The ed1 code-eval Evaluation Procedure Config (Eval Node Variable).

    Empty preprocessing (the sandbox owns extraction) + the ed1 code-eval
    Metric
    Extraction Config. Its ``config_identity_hash`` is the Procedure identity
    the
    enc-dec Eval Node references and both Eval Configs fold in.
    """
    preprocessing = PreprocessingDefinition(
        definition_id="whetstone.ed1.preprocess",
        version=_DEFINITION_VERSION,
        steps=(),
    ).materialize()
    metric_extraction = _ed1_metric_extraction_config()
    return EvaluationProcedureDefinition(
        definition_id="whetstone.ed1.procedure",
        version=_DEFINITION_VERSION,
    ).materialize(
        preprocessing=preprocessing,
        metric_extraction=metric_extraction,
        assignment={"zero_denominator": zero_denominator},
    )


def _ed1_task_set(split_role: str, task_ids: tuple[str, ...]) -> TaskSet:
    return TaskSet(
        manifest_id=f"whetstone.ed1.{split_role}",
        version=_DEFINITION_VERSION,
        dataset_revision=ED1_DATASET_REVISION,
        task_identities=task_ids,
    )


def _ed1_split(
    *,
    split_role: str,
    instances: tuple[Instance, ...],
    procedure: EvaluationProcedureConfig,
    completeness: Completeness,
    max_skip_fraction: float,
    repeats: int,
) -> EnvSplitSampling:
    task_ids = tuple(str(inst.id) for inst in instances)
    task_set = _ed1_task_set(split_role, task_ids)
    repeat_plan = RepeatPlan(
        plan_id=f"whetstone.ed1.{split_role}",
        version=_DEFINITION_VERSION,
        task_identities=task_ids,
        repeat_count=repeats,
    )
    sampling = SamplingDefinition(
        definition_id=f"whetstone.ed1.{split_role}.sampling",
        version=_DEFINITION_VERSION,
    ).materialize(
        {
            "task_set_hash": task_set.identity_hash(),
            "repeat_plan_hash": repeat_plan.identity_hash(),
        }
    )
    policy = completeness.to_policy(max_skip_fraction=max_skip_fraction)
    aggregation = aggregation_definition(
        "whetstone.ed1.aggregation"
    ).materialize(
        {
            "reduction": "mean",
            "missing_data": policy.missing_data,
            "zero_denominator": "not_applicable",
            "max_skip_fraction": policy.skip_fraction_token(),
        }
    )
    # Compose the ed1 Eval Config directly from its three component Configs
    # (the QA composer wants an EnvSpec; ed1 is not a QA env, so it builds the
    # same EvalDefinition here). Both Eval Configs share the Procedure
    # identity, so ``graph_hash`` is stable across internal/official.
    eval_config = EvalDefinition(
        definition_id=f"whetstone.{ED1_ENV_NAME}.eval",
        version=_DEFINITION_VERSION,
    ).materialize(
        sampling=sampling,
        evaluation_procedure=procedure,
        aggregation=aggregation,
    )
    return EnvSplitSampling(
        split_role=split_role,
        instances=instances,
        task_set=task_set,
        repeat_plan=repeat_plan,
        sampling_config=sampling,
        eval_config=eval_config,
    )


def _ed1_candidate(*, candidate_id: str, body: str) -> Candidate:
    # The Mutation Surface payload is the STRATEGY-SENTENCE BODY only; the
    # budget clause + code block are the immutable frame composed at render.
    return Candidate(
        candidate_id=candidate_id,
        base_ref=f"whetstone.env.{ED1_ENV_NAME}.base",
        payload={MUTATION_FIELD: body},
    )


def ed1_initial_candidate() -> Candidate:
    """The naive Initial Candidate: strategy body A ("concise description")."""
    return _ed1_candidate(
        candidate_id=f"{ED1_ENV_NAME}-naive", body=ENCODER_BODY_A
    )


def ed1_ceiling_candidate() -> Candidate:
    """The ceiling reference: strategy body B (reconstruction-compress)."""
    return _ed1_candidate(
        candidate_id=f"{ED1_ENV_NAME}-ceiling", body=ENCODER_BODY_B
    )


def build_ed1_reward_policy() -> RewardPolicy:
    """The ed1 Reward Policy: MAXIMIZE the internal Binary Test Pass rate ONLY.

    One unit-weight, maximize term over the ``binary_test_pass`` internal
    aggregate. Compression is REPORTED alongside but is NOT a Reward term (the
    dual-objective / Pareto selection is a flagged follow-up). ``missing_data =
    FAIL`` matches the QA envs (a missing internal aggregate is not scorable).
    """
    return RewardPolicy(
        policy_name=f"whetstone.env.{ED1_ENV_NAME}.reward",
        reward_name="reward",
        terms=(
            RewardTerm(name=ED1_PASS_RATE_NAME, weight=1.0, maximize=True),
        ),
        missing_data=MissingDataPolicy.FAIL,
    )


def ed1_reward_from_pass_rate(
    policy: RewardPolicy, *, pass_rate: float | None
) -> Reward:
    """Apply the ed1 Reward Policy to the internal Binary Test Pass rate.

    Names the aggregate under the ed1 pass-rate term
    (:data:`ED1_PASS_RATE_NAME`)
    -- distinct from the QA ``env_exact_match`` term -- and pins the evidence
    role to ``internal``. A missing pass rate under ``FAIL`` surfaces as a
    typed
    :class:`~whetstone.envs.reward.CandidateEvaluationFailure` the optimizer
    loop
    handles (candidate marked failed), never a bare ``ValueError``.
    """
    from whetstone.envs.reward import CandidateEvaluationFailure

    try:
        return apply_reward_policy(
            policy,
            aggregates={ED1_PASS_RATE_NAME: pass_rate},
            evidence_role=EvaluationRole.INTERNAL,
        )
    except ValueError as exc:
        raise CandidateEvaluationFailure(
            "ed1 internal candidate has no computable Reward: the "
            f"{ED1_PASS_RATE_NAME!r} aggregate is missing/incomplete under "
            f"the FAIL missing-data policy (pass_rate={pass_rate!r})"
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
    budget_ratio: float = ED1_DEFAULT_BUDGET_RATIO
    dataset_revision: str = ED1_DATASET_REVISION
    #: The injectable code scorer (raw_submission, task) -> CodeScore. ``None``
    #: uses the production dr-code container sandbox; tests / dry-runs inject a
    #: local no-container runner so no Docker/network is needed.
    scorer: Callable[..., CodeScore] | None = None


def build_ed1_experiment(
    *,
    model: str = ED1_CANONICAL_MODEL,
    budget_ratio: float = ED1_DEFAULT_BUDGET_RATIO,
    scorer: Callable[..., CodeScore] | None = None,
    prefer_snapshot: bool = True,
    limit: int | None = None,
    internal_n: int | None = None,
    official_n: int | None = None,
    completeness: Completeness = Completeness.PROPAGATE,
    max_skip_fraction: float = 0.0,
    repeats: int = 3,
    tasks: tuple[Ed1Instance, ...] | None = None,
) -> Ed1Experiment:
    """Build the ed1 enc-dec experiment the runner cell consumes.

    Loads the pinned HumanEval+ pool (or uses injected ``tasks`` for tests),
    splits it into internal/official (first-N ordered), builds the 3-node
    enc-dec rollout at ``budget_ratio`` (folded into ``graph_hash``), the naive
    (A) + ceiling (B) encoder candidates, the two Eval Configs (sharing the
    code-eval Procedure identity), and the pass-rate-only Reward Policy.
    """
    pool = tasks if tasks is not None else load_ed1_tasks(
        prefer_snapshot=prefer_snapshot, limit=limit
    )
    if not pool:
        raise ValueError("ed1 task pool is empty")
    procedure = build_ed1_procedure_config()
    rollout = build_encdec_rollout_definition(
        ED1_ENV_NAME,
        model=model,
        procedure_config_hash=procedure.config_identity_hash,
        budget_ratio=budget_ratio,
    )
    all_instances = tuple(t.instance for t in pool)
    n = len(all_instances)
    # First-N ordered split: internal then official (disjoint, contiguous). A
    # tiny pilot pool may put all tasks in the official split.
    i_n = internal_n if internal_n is not None else min(max(1, n // 2), n)
    internal_instances = all_instances[:i_n]
    rest = all_instances[i_n:]
    o_n = official_n if official_n is not None else len(rest)
    official_instances = rest[:o_n] if rest else internal_instances[:o_n or n]
    if not official_instances:
        official_instances = internal_instances
    internal_split = _ed1_split(
        split_role="internal_eval", instances=internal_instances,
        procedure=procedure, completeness=completeness,
        max_skip_fraction=max_skip_fraction, repeats=repeats,
    )
    official_split = _ed1_split(
        split_role="official", instances=official_instances,
        procedure=procedure, completeness=completeness,
        max_skip_fraction=max_skip_fraction, repeats=repeats,
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
        reward_policy=build_ed1_reward_policy(),
        completeness_policy=completeness.to_policy(
            max_skip_fraction=max_skip_fraction
        ),
        encdec_rollout=rollout,
        budget_ratio=budget_ratio,
        dataset_revision=ED1_DATASET_REVISION,
        scorer=scorer,
    )


#: Callable type for reconstructing a HumanEvalTask (test injection point).
HumanEvalTaskFromInstance = Callable[[Instance], HumanEvalTask]

_ = field  # keep the dataclass field import referenced


__all__ = [
    "DECODER_TEMPLATE",
    "ED1_CANONICAL_MODEL",
    "ED1_COMPRESSION_NAME",
    "ED1_DATASET_ID",
    "ED1_DATASET_REVISION",
    "ED1_DEFAULT_BUDGET_RATIO",
    "ED1_ENV_NAME",
    "ED1_INVALID_BODY",
    "ED1_PASS_RATE_NAME",
    "ENCODER_BODY_A",
    "ENCODER_BODY_B",
    "ENCODER_FRAME",
    "ENCODER_TEMPLATE_A",
    "ENCODER_TEMPLATE_B",
    "Ed1Experiment",
    "Ed1Instance",
    "build_ed1_experiment",
    "build_ed1_procedure_config",
    "build_ed1_reward_policy",
    "ed1_body_rejection",
    "ed1_ceiling_candidate",
    "ed1_initial_candidate",
    "humaneval_task_from_instance",
    "load_ed1_tasks",
    "render_encoder_frame",
]
