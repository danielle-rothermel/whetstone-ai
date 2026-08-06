from __future__ import annotations

import keyword
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from dr_code.humaneval import HumanEvalTask
from dr_graph import GraphConfig, GraphDefinition, graph_hash
from dr_providers import ProviderCallConfig, openrouter_chat_config
from whetstone_envs.core import Instance

from whetstone.envs.ed1 import (
    ED1_DATASET_REVISION,
    ED1_SUBMISSION_SCORE_NAME,
    Ed1Instance,
    build_ed1_procedure_config,
    load_ed1_tasks,
)
from whetstone.envs.ed1_scoring import CodeScore
from whetstone.envs.factory import EnvExperiment, RolloutDefinitionLike
from whetstone.envs.rollout_definition import (
    EVAL_NODE_ID,
    LLM_NODE_ID,
    PROMPT_EXTERNAL_INPUT,
    PROVIDER_CALL_CONFIG_SCHEMA,
    env_candidate_base_ref,
)
from whetstone.envs.sampling import (
    Completeness,
    EnvEvalConfigs,
    EnvSplitSampling,
    derive_split_sampling,
)
from whetstone.envs.task_selection import (
    TaskSplitRoles,
    resolve_manifest_split,
)
from whetstone.evaluation.code.aggregate import aggregation_definition
from whetstone.experiment.candidate import Candidate
from whetstone.experiment.graph.nodes import (
    eval_node_definition,
    eval_variable_assignment,
    llm_call_node_definition,
    llm_call_variable_assignment,
)
from whetstone.experiment.reward import (
    MissingDataPolicy,
    RewardPolicy,
    RewardTerm,
)
from whetstone.optimization.proposal.mutation import MUTATION_FIELD

D1_ENV_NAME = "d1"

#: The canonical d1 task model. d1's science pairs a clean model against
#: deepseek (the contamination axis), so the CLI ``--task-model`` selects the
#: model per cell; the matrix default mirrors ed1's deepseek enc/dec model so a
#: d1 anchor pairs with the corresponding ed1 anchor on the same model family.
D1_CANONICAL_MODEL = "deepseek/deepseek-v4-flash"

#: The per-row metric, aggregate, and Reward-term identity for D1 correctness.
D1_SUBMISSION_SCORE_NAME = ED1_SUBMISSION_SCORE_NAME

#: The d1 procedure config schema for the direct code-eval Eval Node.
D1_PROCEDURE_CONFIG_SCHEMA = "whetstone.d1_code_eval_procedure"

_DEFINITION_VERSION = "1"

# --- The frozen input arms ---------------------------------------------------

#: The five frozen input arms d1 can pin. ``renamed`` additionally scrubs every
#: canonical-name occurrence (signature and doctests) and scores against the
#: renamed entry point.
#: The one arm that reads ``rename_token`` (and so folds it into identity).
D1_RENAMED_ARM = "renamed"

D1_INPUT_ARMS: tuple[str, ...] = (
    "original",
    "docstring",
    "signature",
    "name",
    D1_RENAMED_ARM,
)

D1_DEFAULT_RENAME_TOKEN = "target_fxn"


# --- The mutable surrounding wrapper (the Mutation Surface) ------------------
#
# d1's Mutation Surface is a leading strategy BODY the optimizer mutates; an
# immutable frame composes it around the FROZEN input arm. The frame owns the
# ``{body}`` / ``{input_arm}`` placeholders, so a body carries NONE of its own
# (body validation reuses ed1's ``ed1_body_rejection``).

#: The immutable d1 wrapper frame: a mutable strategy ``{body}`` followed by
#: the FROZEN ``{input_arm}`` text. ``{body}`` is the ONLY mutable region.
D1_WRAPPER_FRAME = "{body}\n{input_arm}"

#: The naive wrapper body reproduces the canonical direct prompt instruction.
D1_WRAPPER_BODY_NAIVE = (
    "Write a complete, correct Python implementation for the following. "
    "Output only Python code."
)

#: A ceiling-reference wrapper body (the headroom probe): a more explicit
#: instruction. Distinct from the naive body -> a distinct rendered prompt.
D1_WRAPPER_BODY_CEILING = (
    "You are an expert Python engineer. Implement the following completely "
    "and correctly, handling all edge cases. Output only the Python function."
)


def render_d1_frame(body: str, *, input_arm: str) -> str:
    """Compose the immutable d1 wrapper frame around a mutable strategy body.

    ``body`` is the Mutation-Surface payload (the strategy sentence ONLY);
    ``input_arm`` is the frozen input-arm text. A body carrying a
    ``{placeholder}`` would raise here -- but intake validation rejects such
    bodies first (the frame owns every placeholder).
    """
    return D1_WRAPPER_FRAME.format(body=body, input_arm=input_arm)


def _d1_candidate(*, candidate_id: str, body: str) -> Candidate:
    # The Mutation Surface payload is the wrapper BODY only; the frame + the
    # frozen input arm are composed at render.
    return Candidate(
        candidate_id=candidate_id,
        base_ref=env_candidate_base_ref(D1_ENV_NAME),
        payload={MUTATION_FIELD: body},
    )


def d1_initial_candidate() -> Candidate:
    return _d1_candidate(
        candidate_id=f"{D1_ENV_NAME}-naive", body=D1_WRAPPER_BODY_NAIVE
    )


def d1_ceiling_candidate() -> Candidate:
    """The ceiling reference: the explicit-instruction wrapper body."""
    return _d1_candidate(
        candidate_id=f"{D1_ENV_NAME}-ceiling", body=D1_WRAPPER_BODY_CEILING
    )


def build_d1_reward_policy() -> RewardPolicy:
    """The D1 Reward Policy: maximize HumanEval Submission Score only."""
    return RewardPolicy(
        policy_name=f"whetstone.env.{D1_ENV_NAME}.reward",
        reward_name="reward",
        terms=(
            RewardTerm(
                name=D1_SUBMISSION_SCORE_NAME,
                weight=1.0,
                maximize=True,
            ),
        ),
        missing_data=MissingDataPolicy.FAIL,
    )


# --- The single-LLM-call direct rollout definition --------------------------


@dataclass(frozen=True, slots=True)
class D1RolloutDefinition:
    """The d1 direct Rollout Definition graph + the config references it binds.

    A single LLM Call Node -> terminal Eval Node (the SAME two-node shape the
    QA envs use), with the code-eval Evaluation Procedure on the Eval Node. The
    FROZEN ``input_arm`` folds into ``graph_hash`` (a distinct arm is a
    distinct graph variant), so a d1 cell on ``renamed`` is identity-distinct
    from one on ``original``.
    """

    env_name: str
    definition: GraphDefinition
    provider_call_config: ProviderCallConfig
    procedure_config_hash: str
    input_arm: str
    graph_config: GraphConfig

    @property
    def graph_hash(self) -> str:
        """The native dr-graph Graph Config Identity Hash."""
        return graph_hash(self.graph_config)


def d1_graph_definition() -> GraphDefinition:
    """The d1 direct LLM Call -> terminal Eval Graph Definition.

    The SAME two-node shape as the QA graph, but the LLM Call Node DECLARES the
    input-arm control Variable (reusing the ``character_budget_rule`` slot to
    carry the FROZEN input-arm token) so a distinct input arm yields a distinct
    ``graph_hash`` -- the arm is an output-affecting knob that MUST fold into
    graph identity, exactly as ed1 folds its budget ratio.
    """
    llm = llm_call_node_definition(
        LLM_NODE_ID,
        prompt_source=PROMPT_EXTERNAL_INPUT,
        declares_character_budget=True,
    )
    ev = eval_node_definition(
        EVAL_NODE_ID,
        upstream_sources={"generation": LLM_NODE_ID},
    )
    return GraphDefinition(nodes=(llm, ev), terminal_node_id=EVAL_NODE_ID)


def d1_arm_token(input_arm: str, rename_token: str) -> str:
    """The identity-bearing control token for one (arm, rename token) pair.

    ``rename_token`` folds in ONLY for the ``renamed`` arm -- it is the text
    substituted for every canonical name in that arm, so two ``renamed`` cells
    with different tokens are different experiments. The other arms never read
    the token, so folding it there would churn their identities for a value
    they ignore.
    """
    if input_arm == D1_RENAMED_ARM:
        return f"d1_input_arm:{input_arm}|rename={rename_token}"
    return f"d1_input_arm:{input_arm}"


def build_d1_graph_config(
    *,
    provider_call_config_hash: str,
    evaluation_procedure_config_hash: str,
    input_arm: str,
    rename_token: str = D1_DEFAULT_RENAME_TOKEN,
) -> GraphConfig:
    """Materialize the d1 Graph Config binding the route, procedure, and arm.

    The LLM Call Node carries the Provider Call Config reference AND the FROZEN
    input-arm control token (in the declared budget-variable slot); the Eval
    Node carries the code-eval Procedure reference. A distinct arm -- or, on
    the ``renamed`` arm, a distinct ``rename_token`` -- yields a distinct
    ``graph_hash`` (identity-folded by construction).
    """
    definition = d1_graph_definition()
    assignments = {
        LLM_NODE_ID: llm_call_variable_assignment(
            provider_call_config_schema=PROVIDER_CALL_CONFIG_SCHEMA,
            provider_call_config_hash=provider_call_config_hash,
            character_budget_rule=d1_arm_token(input_arm, rename_token),
        ),
        EVAL_NODE_ID: eval_variable_assignment(
            evaluation_procedure_config_schema=D1_PROCEDURE_CONFIG_SCHEMA,
            evaluation_procedure_config_hash=(
                evaluation_procedure_config_hash
            ),
        ),
    }
    return definition.materialize(assignments)


def build_d1_rollout_definition(
    *,
    model: str,
    procedure_config_hash: str,
    input_arm: str,
    rename_token: str = D1_DEFAULT_RENAME_TOKEN,
) -> D1RolloutDefinition:
    """Build the d1 direct Rollout Definition for one (model, input arm)."""
    provider_call_config = openrouter_chat_config(model=model)
    graph_config = build_d1_graph_config(
        provider_call_config_hash=provider_call_config.identity_hash,
        evaluation_procedure_config_hash=procedure_config_hash,
        input_arm=input_arm,
        rename_token=rename_token,
    )
    return D1RolloutDefinition(
        env_name=D1_ENV_NAME,
        definition=d1_graph_definition(),
        provider_call_config=provider_call_config,
        procedure_config_hash=procedure_config_hash,
        input_arm=input_arm,
        graph_config=graph_config,
    )


# --- The split builder (arm folds into Task Set identity) --------------------


def _d1_split(
    *,
    split_role: str,
    instances: tuple[Instance, ...],
    procedure,
    completeness: Completeness,
    max_skip_fraction: float,
    repeats: int,
    input_arm: str,
    rename_token: str = D1_DEFAULT_RENAME_TOKEN,
    manifest_tag: str | None = None,
) -> EnvSplitSampling:
    """A d1 split whose Task Set + sampling fold in the FROZEN input arm.

    Mirrors ``ed1._ed1_split`` but adds the input arm to the manifest id so a
    ``renamed`` cell and an ``original`` cell over the SAME task ids have
    DISTINCT ``eval_config_hash`` values (the arm is identity-bearing). On the
    ``renamed`` arm the ``rename_token`` folds in too: it changes the scored
    entry point, so two tokens are two experiments.

    ``manifest_tag`` (a task-split manifest's content hash and pool)
    folds in ALONGSIDE the input arm so a manifest-driven split is a DISTINCT
    eval_config_hash from both a first-N slice and a same-arm non-manifest
    cell. ``None`` leaves the ids byte-identical to a first-N slice cell.
    """
    policy = completeness.to_policy(max_skip_fraction=max_skip_fraction)
    aggregation = aggregation_definition(
        "whetstone.d1.aggregation"
    ).materialize(
        {
            "reduction": "mean",
            "missing_data": policy.missing_data,
            "zero_denominator": "not_applicable",
            "max_skip_fraction": policy.skip_fraction_token(),
        }
    )
    namespace = f"whetstone.d1.{input_arm}"
    if input_arm == D1_RENAMED_ARM:
        namespace = f"{namespace}.{rename_token}"
    if manifest_tag is not None:
        namespace = f"{namespace}.{manifest_tag}"
    return derive_split_sampling(
        namespace=namespace,
        dataset_revision=ED1_DATASET_REVISION,
        split_role=split_role,
        instances=instances,
        task_identity_of=lambda instance: str(instance.id),
        repeats=repeats,
        procedure=procedure,
        aggregation=aggregation,
    )


@dataclass(frozen=True, slots=True)
class D1Experiment(EnvExperiment):
    """An ``EnvExperiment`` for the d1 direct-generation env.

    Adds the FROZEN ``input_arm`` + the ``rename_token`` and the direct
    :class:`D1RolloutDefinition` on top of the base experiment shape. The arm
    is identity-bearing unconditionally; the ``rename_token`` is identity-
    bearing on the ``renamed`` arm, the only arm that reads it (see
    :func:`d1_arm_token`). ``rollout_definition`` (the base field) is the same
    direct rollout so ``experiment.rollout_definition.graph_hash`` resolves for
    the runner. The
    per-task HumanEval map (``humaneval_by_id``) lets the direct drive rebuild
    the frozen input-arm prompt + the (possibly renamed) scoring task.
    """

    input_arm: str = "original"
    rename_token: str = D1_DEFAULT_RENAME_TOKEN
    dataset_revision: str = ""
    #: The injectable code scorer (raw_submission, task) -> CodeScore. The
    #: production injection runs candidate code through the caller's explicit
    #: dr-exec executor; tests may inject a controlled scorer.
    scorer: Callable[..., CodeScore] | None = None
    #: Per-Instance-id parsed HumanEval task (for the frozen input-arm render +
    #: the renamed-arm scoring task); empty for a bare shape.
    humaneval_by_id: dict[str, HumanEvalTask] = field(default_factory=dict)

    def humaneval_for(self, instance: Instance) -> HumanEvalTask:
        """The parsed HumanEval task for one d1 Instance."""
        return self.humaneval_by_id[str(instance.id)]


def build_d1_experiment(
    *,
    model: str = D1_CANONICAL_MODEL,
    input_arm: str = "original",
    rename_token: str = D1_DEFAULT_RENAME_TOKEN,
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
    split_manifest: TaskSplitRoles | None = None,
) -> D1Experiment:
    """Build the d1 direct-generation experiment the runner cell consumes.

    Loads the pinned HumanEval+ pool (shared with ed1; ``tasks`` injects a test
    pool), pins the FROZEN ``input_arm``, splits internal/official (first-N
    ordered), builds the single-LLM-call direct rollout (arm folded into
    ``graph_hash``), the naive + ceiling wrapper candidates, the two Eval
    Configs (sharing the code-eval Procedure identity; arm folded into each
    ``eval_config_hash``), and the unblended HumanEval Submission Score Reward
    Policy.

    ``exclude_task_ids`` drops those ids from the ordered pool before the
    split, exactly as ed1 does.

    ``split_manifest`` overrides the first-N slice with role-true
    train/val/test semantics: internal = the manifest's ``train + val`` ids (by
    MEMBERSHIP, manifest order -- no val sub-split exists, so val folds into
    internal alongside train), official = the manifest's ``test`` ids EXACTLY
    (membership, NOT first-N). ``official_n`` then caps WITHIN the test set.
    The manifest's content hash + pool folds into each split's Task Set
    identity ALONGSIDE the input arm.
    """
    if input_arm not in D1_INPUT_ARMS:
        raise ValueError(
            f"unknown d1 input arm {input_arm!r} "
            f"(choose one of {D1_INPUT_ARMS})"
        )
    # The rename token is substituted into rendered SOURCE, so an invalid
    # identifier is a per-row SyntaxError at drive time (after the pool load
    # and every provider call). Reject it at build time instead.
    if not rename_token.isidentifier() or keyword.iskeyword(rename_token):
        raise ValueError(
            f"d1 rename_token {rename_token!r} is not a valid, "
            "non-keyword Python identifier"
        )
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
        raise ValueError("d1 task pool is empty")
    procedure = build_d1_procedure_config()
    rollout = build_d1_rollout_definition(
        model=model,
        procedure_config_hash=procedure.config_identity_hash,
        input_arm=input_arm,
        rename_token=rename_token,
    )
    humaneval_by_id = {str(t.instance.id): t.humaneval_task for t in pool}
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
            print(f"[d1] {resolved.official_capped}")
    else:
        all_instances = tuple(t.instance for t in pool)
        n = len(all_instances)
        i_n = internal_n if internal_n is not None else min(max(1, n // 2), n)
        internal_instances = all_instances[:i_n]
        rest = all_instances[i_n:]
        o_n = official_n if official_n is not None else len(rest)
        official_instances = (
            rest[:o_n] if rest else internal_instances[: o_n or n]
        )
        if not official_instances:
            official_instances = internal_instances
    internal_split = _d1_split(
        split_role="internal_eval",
        instances=internal_instances,
        procedure=procedure,
        completeness=completeness,
        max_skip_fraction=max_skip_fraction,
        repeats=repeats,
        input_arm=input_arm,
        rename_token=rename_token,
        manifest_tag=manifest_tag,
    )
    official_split = _d1_split(
        split_role="official",
        instances=official_instances,
        procedure=procedure,
        completeness=completeness,
        max_skip_fraction=max_skip_fraction,
        repeats=repeats,
        input_arm=input_arm,
        rename_token=rename_token,
        manifest_tag=manifest_tag,
    )
    eval_configs = EnvEvalConfigs(
        env_name=D1_ENV_NAME,
        procedure_config_hash=procedure.config_identity_hash,
        internal=internal_split,
        official=official_split,
        held_out_task_identities=(),
    )
    return D1Experiment(
        env_name=D1_ENV_NAME,
        rollout_definition=rollout,  # type: ignore[arg-type]
        initial_candidate=d1_initial_candidate(),
        ceiling_candidate=d1_ceiling_candidate(),
        eval_configs=eval_configs,
        reward_policy=build_d1_reward_policy(),
        completeness_policy=completeness.to_policy(
            max_skip_fraction=max_skip_fraction
        ),
        input_arm=input_arm,
        rename_token=rename_token,
        dataset_revision=ED1_DATASET_REVISION,
        scorer=scorer,
        humaneval_by_id=humaneval_by_id,
    )


def build_d1_procedure_config():
    """The d1 direct code-eval Evaluation Procedure Config.

    d1 reuses ed1's code-eval Procedure (the same HumanEval sandbox + zstd
    compression Metric Questions); d1 does not use the compression metric (its
    Reward is submission-score only), but sharing the Procedure keeps the
    identity domain common with ed1 so a d1 vs ed1 comparison is on the same
    eval wiring.
    """
    return build_ed1_procedure_config()


_ROLLOUT_LIKE: type[RolloutDefinitionLike] = D1RolloutDefinition  # type check


__all__ = [
    "D1_CANONICAL_MODEL",
    "D1_DEFAULT_RENAME_TOKEN",
    "D1_ENV_NAME",
    "D1_INPUT_ARMS",
    "D1_RENAMED_ARM",
    "D1_SUBMISSION_SCORE_NAME",
    "D1_WRAPPER_BODY_CEILING",
    "D1_WRAPPER_BODY_NAIVE",
    "D1_WRAPPER_FRAME",
    "D1Experiment",
    "D1RolloutDefinition",
    "build_d1_experiment",
    "build_d1_procedure_config",
    "build_d1_reward_policy",
    "build_d1_rollout_definition",
    "d1_ceiling_candidate",
    "d1_graph_definition",
    "d1_initial_candidate",
    "render_d1_frame",
]
