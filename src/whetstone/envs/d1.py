"""The ``d1`` direct-optimization precursor env (task 23).

``d1`` is the DIRECT-generation counterpart to the enc-dec ``ed1`` family: a
single LLM Call renders one prompt and the model's output is scored directly
through the SAME dr-code HumanEval sandbox ``ed1`` uses. There is no encoder or
decoder -- the model generates the function implementation in one call.

d1's two-part prompt has a FROZEN input arm and a MUTABLE surrounding wrapper:

* the FROZEN ``--input-arm`` is one of the five screen DIRECT arms
  (:data:`whetstone.runner.task_screen.DIRECT_ARMS`): ``original`` /
  ``docstring`` / ``signature`` / ``name`` / ``renamed``. The construction
  REUSES the screen driver verbatim (``split_prompt`` / ``_direct_body`` /
  ``renamed_task`` / ``rename_identifier``), including the RENAMED arm's
  all-occurrence canonical-name scrub (signature AND doctests -> a neutral
  token) and the amendment-2 scoring trap (a renamed arm scores against the
  RENAMED entry point, never the leaked canonical name). The chosen input arm
  is a per-experiment CONSTANT (identity-bearing): a d1 cell is pinned to one
  arm and folds it into the split Task Set / graph identity.
* the MUTABLE surrounding wrapper is the Mutation Surface the optimizer varies
  -- a leading strategy/instruction BODY (:data:`D1_WRAPPER_BODY_NAIVE`) an
  immutable frame (:data:`D1_WRAPPER_FRAME`) composes around the frozen input
  arm. The frame owns the ``{body}`` / ``{input_arm}`` placeholders, so a body
  never needs (or is allowed) placeholders of its own -- body validation reuses
  ed1's :func:`whetstone.envs.ed1.ed1_body_rejection`.

The naive d1 wrapper reproduces the screen's own direct-arm prompt BYTE FOR
BYTE (:data:`D1_WRAPPER_BODY_NAIVE` == the screen ``_direct_prompt``
instruction sentence), so a d1 EVAL anchor on a given input arm reproduces the
corresponding screen arm's pass numbers -- the naive probe IS the screen
wrapper.

Reward is a PLAIN pass-rate (the Average Binary Test Pass Rate), NOT the
weighted blend (blended reward is an ed1/ed1m standing rule ONLY -- task 22).
The full runner stack (power analysis / optimization traces / telemetry /
reasoning-effort / temperature / partials / resume) rides the SHARED seams.

Science framing (respected in naming/docs, not implemented here): d1 is the
information-floor control (clean models x ablated input arms) and the
contamination-exploitation trap (deepseek x ablated arms; the accepted prompt
text is the evidence). The old "enforcement impl" idea is DEAD -- superseded by
the ed1 blended reward; d1 builds NO budget enforcement.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dr_code.eval import (
    EvalDefinition,
    RepeatPlan,
    SamplingDefinition,
)
from dr_code.humaneval import HumanEvalTask
from dr_graph import GraphConfig, GraphDefinition, graph_hash
from dr_providers import ProviderCallConfig, openrouter_chat_config
from whetstone_envs.core import Instance

from whetstone.code_eval.aggregate import aggregation_definition
from whetstone.envs.ed1 import (
    Ed1Instance,
    _ed1_task_set,
    build_ed1_procedure_config,
    build_ed1_reward_policy,
    load_ed1_tasks,
)
from whetstone.envs.ed1_scoring import CodeScore
from whetstone.envs.factory import EnvExperiment, RolloutDefinitionLike
from whetstone.envs.rollout_definition import (
    EVAL_NODE_ID,
    LLM_NODE_ID,
    PROMPT_EXTERNAL_INPUT,
    PROVIDER_CALL_CONFIG_SCHEMA,
)
from whetstone.envs.sampling import (
    Completeness,
    EnvEvalConfigs,
    EnvSplitSampling,
)
from whetstone.graph.nodes import (
    eval_node_definition,
    eval_variable_assignment,
    llm_call_node_definition,
    llm_call_variable_assignment,
)
from whetstone.optimization.mutation import MUTATION_FIELD
from whetstone.optimization.schema import Candidate

if TYPE_CHECKING:
    from whetstone.runner.task_split_manifest import TaskSplitRoles

#: The d1 env id.
D1_ENV_NAME = "d1"

#: The canonical d1 task model. d1's science pairs a clean model against
#: deepseek (the contamination axis), so the CLI ``--task-model`` selects the
#: model per cell; the matrix default mirrors ed1's deepseek enc/dec model so a
#: d1 anchor pairs with the corresponding ed1 anchor on the same model family.
D1_CANONICAL_MODEL = "deepseek/deepseek-v4-flash"

#: The reward-term / aggregate name: the Average Binary Test Pass Rate. d1's
#: Reward is pass-rate ONLY (NOT blended -- blended is ed1/ed1m per task 22).
D1_PASS_RATE_NAME = "binary_test_pass"

#: The d1 procedure config schema for the direct code-eval Eval Node.
D1_PROCEDURE_CONFIG_SCHEMA = "whetstone.d1_code_eval_procedure"

_DEFINITION_VERSION = "1"

# --- The frozen input arms (REUSED from the screen) --------------------------

#: The five frozen input arms d1 can pin, in screen order. Each maps to a
#: screen DIRECT arm's slice of the canonical HumanEval prompt; ``renamed``
#: additionally scrubs EVERY canonical-name occurrence (signature + doctests)
#: and scores against the renamed entry point (the amendment-2 ablation).
D1_INPUT_ARMS: tuple[str, ...] = (
    "original",
    "docstring",
    "signature",
    "name",
    "renamed",
)

#: The default rename token for the renamed input arm (matches the screen's).
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

#: The naive wrapper body -- BYTE-IDENTICAL to the screen ``_direct_prompt``
#: instruction sentence, so a d1 naive (eval-anchor) prompt reproduces the
#: screen's direct-arm prompt exactly and hence the screen arm's pass numbers.
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
        base_ref=f"whetstone.env.{D1_ENV_NAME}.base",
        payload={MUTATION_FIELD: body},
    )


def d1_initial_candidate() -> Candidate:
    """The naive Initial Candidate: the screen-identical wrapper body."""
    return _d1_candidate(
        candidate_id=f"{D1_ENV_NAME}-naive", body=D1_WRAPPER_BODY_NAIVE
    )


def d1_ceiling_candidate() -> Candidate:
    """The ceiling reference: the explicit-instruction wrapper body."""
    return _d1_candidate(
        candidate_id=f"{D1_ENV_NAME}-ceiling", body=D1_WRAPPER_BODY_CEILING
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
    graph identity (the c23-era rule), exactly as ed1 folds its budget ratio.
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


def build_d1_graph_config(
    *,
    provider_call_config_hash: str,
    evaluation_procedure_config_hash: str,
    input_arm: str,
) -> GraphConfig:
    """Materialize the d1 Graph Config binding the route, procedure, and arm.

    The LLM Call Node carries the Provider Call Config reference AND the FROZEN
    input-arm control token (in the declared budget-variable slot); the Eval
    Node carries the code-eval Procedure reference. A distinct arm yields a
    distinct ``graph_hash`` (identity-folded by construction).
    """
    definition = d1_graph_definition()
    assignments = {
        LLM_NODE_ID: llm_call_variable_assignment(
            provider_call_config_schema=PROVIDER_CALL_CONFIG_SCHEMA,
            provider_call_config_hash=provider_call_config_hash,
            character_budget_rule=f"d1_input_arm:{input_arm}",
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
) -> D1RolloutDefinition:
    """Build the d1 direct Rollout Definition for one (model, input arm)."""
    provider_call_config = openrouter_chat_config(model=model)
    graph_config = build_d1_graph_config(
        provider_call_config_hash=provider_call_config.identity_hash,
        evaluation_procedure_config_hash=procedure_config_hash,
        input_arm=input_arm,
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
    manifest_tag: str | None = None,
) -> EnvSplitSampling:
    """A d1 split whose Task Set + sampling fold in the FROZEN input arm.

    Mirrors ``ed1._ed1_split`` but adds the input arm to the manifest id so a
    ``renamed`` cell and an ``original`` cell over the SAME task ids have
    DISTINCT ``eval_config_hash`` values (the arm is identity-bearing).

    ``manifest_tag`` (a task-split-manifest's content-hash + pool, task 29)
    folds in ALONGSIDE the input arm so a manifest-driven split is a DISTINCT
    eval_config_hash from both a first-N slice and a same-arm non-manifest
    cell. ``None`` leaves the ids byte-identical to a first-N slice cell.
    """
    task_ids = tuple(str(inst.id) for inst in instances)
    manifest_role = f"{split_role}.{input_arm}"
    if manifest_tag is not None:
        manifest_role = f"{manifest_role}.{manifest_tag}"
    task_set = _ed1_task_set(manifest_role, task_ids)
    repeat_plan = RepeatPlan(
        plan_id=f"whetstone.d1.{manifest_role}",
        version=_DEFINITION_VERSION,
        task_identities=task_ids,
        repeat_count=repeats,
    )
    sampling = SamplingDefinition(
        definition_id=f"whetstone.d1.{manifest_role}.sampling",
        version=_DEFINITION_VERSION,
    ).materialize(
        {
            "task_set_hash": task_set.identity_hash(),
            "repeat_plan_hash": repeat_plan.identity_hash(),
        }
    )
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
    eval_config = EvalDefinition(
        definition_id=f"whetstone.{D1_ENV_NAME}.eval.{input_arm}",
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


@dataclass(frozen=True, slots=True)
class D1Experiment(EnvExperiment):
    """An ``EnvExperiment`` for the d1 direct-generation env.

    Adds the FROZEN ``input_arm`` + the ``rename_token`` (both identity-
    bearing) and the direct :class:`D1RolloutDefinition` on top of the base
    experiment shape. ``rollout_definition`` (the base field) is the same
    direct rollout so ``experiment.rollout_definition.graph_hash`` resolves for
    the runner. The
    per-task HumanEval map (``humaneval_by_id``) lets the direct drive rebuild
    the frozen input-arm prompt + the (possibly renamed) scoring task.
    """

    input_arm: str = "original"
    rename_token: str = D1_DEFAULT_RENAME_TOKEN
    dataset_revision: str = ""
    #: The injectable code scorer (raw_submission, task) -> CodeScore. ``None``
    #: uses the production dr-code local subprocess sandbox; tests/dry-runs
    #: inject a fast no-subprocess scorer.
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
    prefer_snapshot: bool = True,
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
    ``eval_config_hash``), and the PLAIN pass-rate Reward Policy.

    ``exclude_task_ids`` DROPS those ids from the ordered pool before the split
    (the per-model screen's always-pass exclusion list), exactly as ed1 does.

    ``split_manifest`` (task 29) OVERRIDES the first-N slice with role-true
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
    pool = tasks if tasks is not None else load_ed1_tasks(
        prefer_snapshot=prefer_snapshot, limit=limit
    )
    if exclude_task_ids:
        pool = tuple(
            t for t in pool if str(t.instance.id) not in exclude_task_ids
        )
    if not pool:
        raise ValueError("d1 task pool is empty")
    from whetstone.envs.ed1 import ED1_DATASET_REVISION

    procedure = build_d1_procedure_config()
    rollout = build_d1_rollout_definition(
        model=model,
        procedure_config_hash=procedure.config_identity_hash,
        input_arm=input_arm,
    )
    humaneval_by_id = {
        str(t.instance.id): t.humaneval_task for t in pool
    }
    manifest_tag: str | None = None
    if split_manifest is not None:
        from whetstone.runner.task_split_manifest import resolve_manifest_split
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
            rest[:o_n] if rest else internal_instances[:o_n or n]
        )
        if not official_instances:
            official_instances = internal_instances
    internal_split = _d1_split(
        split_role="internal_eval", instances=internal_instances,
        procedure=procedure, completeness=completeness,
        max_skip_fraction=max_skip_fraction, repeats=repeats,
        input_arm=input_arm, manifest_tag=manifest_tag,
    )
    official_split = _d1_split(
        split_role="official", instances=official_instances,
        procedure=procedure, completeness=completeness,
        max_skip_fraction=max_skip_fraction, repeats=repeats,
        input_arm=input_arm, manifest_tag=manifest_tag,
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
        reward_policy=build_ed1_reward_policy(),
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
    compression Metric Questions); d1 does not USE the compression metric (its
    Reward is pass-only), but sharing the Procedure keeps the identity domain
    common with ed1 so a d1 vs ed1 comparison is on the same eval wiring.
    """
    return build_ed1_procedure_config()


_ROLLOUT_LIKE: type[RolloutDefinitionLike] = D1RolloutDefinition  # type check


__all__ = [
    "D1_CANONICAL_MODEL",
    "D1_DEFAULT_RENAME_TOKEN",
    "D1_ENV_NAME",
    "D1_INPUT_ARMS",
    "D1_PASS_RATE_NAME",
    "D1_WRAPPER_BODY_CEILING",
    "D1_WRAPPER_BODY_NAIVE",
    "D1_WRAPPER_FRAME",
    "D1Experiment",
    "D1RolloutDefinition",
    "build_d1_experiment",
    "build_d1_procedure_config",
    "build_d1_rollout_definition",
    "d1_ceiling_candidate",
    "d1_graph_definition",
    "d1_initial_candidate",
    "render_d1_frame",
]
