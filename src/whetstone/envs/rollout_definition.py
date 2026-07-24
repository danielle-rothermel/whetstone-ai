"""The per-env Rollout Definition role graph and its candidates.

Rubric criterion 1 (validation plan cell definition): one LLM Call Node plus
one terminal Eval Node.

* **LLM Call Node** (``whetstone.llm-call/v1``): its Provider Call Config is a
  static Variable (in Graph Config identity); its prompt is the only Node
  Input Source, bound to the env task's Graph External Inputs. The
  prompt-template is the **Mutation Surface** (the encoder
  ``user_prompt_template``), rendered against ``task.<field>`` external
  inputs. The naive probe is the **Initial Candidate**; the ceiling probe is
  the **reference candidate** (headroom line).
* **Eval Node** (``whetstone.eval/v1``, terminal): its Evaluation Procedure
  Config is a static Variable; when executed it invokes the env oracle via
  the whetstone metric-extraction operator, emitting the ``env_exact_match``
  Metric Fact / Score.

The prompt Node Input Source binds to a single Graph External Input,
``task.prompt`` -- the *rendered* prompt for the selected candidate against a
task's external inputs. Rendering is env-owned (the env ``ProbePair``); this
adapter renders the chosen candidate's template against the task's public
prompt inputs and supplies the result as ``task.prompt`` at runtime.

The Provider Call Config is a native dr-providers Config; its Identity Hash
is the static Variable value on the LLM Call Node, so a model/route/control
change changes ``graph_hash``. The Evaluation Procedure Config identity is
the Eval Node's static Variable, so a Procedure change changes ``graph_hash``
too. Both are references (schema + Identity Hash), never Node Input Sources.
"""

from __future__ import annotations

from dataclasses import dataclass

from dr_graph import GraphConfig, GraphDefinition, graph_hash
from dr_providers import ProviderCallConfig, openrouter_chat_config
from whetstone_envs.core import Instance

from whetstone.envs.procedure import (
    EVALUATION_PROCEDURE_CONFIG_SCHEMA,
    env_procedure_config,
)
from whetstone.envs.registry import EnvSpec
from whetstone.envs.task import EnvTask
from whetstone.graph.nodes import (
    eval_node_definition,
    eval_variable_assignment,
    llm_call_node_definition,
    llm_call_variable_assignment,
)
from whetstone.optimization.mutation import (
    MUTATION_FIELD,
    invalid_template_placeholders,
    template_placeholder_fields,
)
from whetstone.optimization.schema import Candidate

#: The Provider Call Config schema name (referenced by the LLM Call Node's
#: static Variable typed reference).
PROVIDER_CALL_CONFIG_SCHEMA = "dr_providers.provider_call_config"

#: The single Graph External Input the LLM Call Node's prompt binds to: the
#: rendered prompt for the selected candidate against a task's external
#: inputs. The env task's ``task.<field>`` prompt inputs feed the render.
PROMPT_EXTERNAL_INPUT = "task.prompt"

#: Node ids for the two-node graph.
LLM_NODE_ID = "generate"
EVAL_NODE_ID = "evaluate"

#: The Eval Node's declared upstream input: the LLM Call Node's generation.
_EVAL_INPUT_FIELD = "generation"


@dataclass(frozen=True, slots=True)
class EnvRolloutDefinition:
    """The Rollout Definition role graph plus the config references it binds.

    ``definition`` is the native dr-graph :class:`GraphDefinition` playing the
    Rollout Definition role. ``provider_call_config`` and ``procedure_config``
    are the native configs whose Identity Hashes are the Nodes' static
    Variables; ``graph_config`` is the materialized Graph Config for this env.
    """

    env_name: str
    definition: GraphDefinition
    provider_call_config: ProviderCallConfig
    procedure_config_hash: str
    graph_config: GraphConfig

    @property
    def graph_hash(self) -> str:
        """The native dr-graph Graph Config Identity Hash."""
        return graph_hash(self.graph_config)


def build_provider_call_config(model: str) -> ProviderCallConfig:
    """The native OpenRouter Provider Call Config for a task model.

    A minimal chat Config over ``model``; its Identity Hash is the LLM Call
    Node's Provider Call Config static Variable. Controls are left at their
    defaults (temperature/token-limit are folded into the Config identity
    when set by a caller); the validation runner supplies concrete controls.
    """
    return openrouter_chat_config(model=model)


def llm_eval_graph_definition() -> GraphDefinition:
    """The minimal LLM Call -> terminal Eval Graph Definition.

    One ``whetstone.llm-call/v1`` Node whose prompt binds the
    ``task.prompt`` Graph External Input, and one terminal
    ``whetstone.eval/v1`` Node consuming the generation.
    """
    llm = llm_call_node_definition(
        LLM_NODE_ID,
        prompt_source=PROMPT_EXTERNAL_INPUT,
    )
    ev = eval_node_definition(
        EVAL_NODE_ID,
        upstream_sources={_EVAL_INPUT_FIELD: LLM_NODE_ID},
    )
    return GraphDefinition(nodes=(llm, ev), terminal_node_id=EVAL_NODE_ID)


def build_graph_config(
    *,
    provider_call_config_hash: str,
    evaluation_procedure_config_hash: str,
) -> GraphConfig:
    """Materialize the env Graph Config binding both config references.

    The Provider Call Config reference is the LLM Call Node's static
    Variable; the Evaluation Procedure Config reference is the Eval Node's.
    Both are typed ``{schema_name, identity_hash}`` references in Graph Config
    identity, so changing either changes ``graph_hash``.
    """
    definition = llm_eval_graph_definition()
    assignments = {
        LLM_NODE_ID: llm_call_variable_assignment(
            provider_call_config_schema=PROVIDER_CALL_CONFIG_SCHEMA,
            provider_call_config_hash=provider_call_config_hash,
        ),
        EVAL_NODE_ID: eval_variable_assignment(
            evaluation_procedure_config_schema=(
                EVALUATION_PROCEDURE_CONFIG_SCHEMA
            ),
            evaluation_procedure_config_hash=(
                evaluation_procedure_config_hash
            ),
        ),
    }
    return definition.materialize(assignments)


def build_rollout_definition(
    env: EnvSpec,
    *,
    model: str,
) -> EnvRolloutDefinition:
    """Build the env's Rollout Definition role graph.

    Wires one LLM Call Node (Provider Call Config = ``model``) to one terminal
    Eval Node (Evaluation Procedure Config = the env oracle procedure), and
    materializes the resulting Graph Config.
    """
    provider_call_config = build_provider_call_config(model)
    procedure = env_procedure_config(env)
    graph_config = build_graph_config(
        provider_call_config_hash=provider_call_config.identity_hash,
        evaluation_procedure_config_hash=procedure.config_identity_hash,
    )
    return EnvRolloutDefinition(
        env_name=env.name,
        definition=llm_eval_graph_definition(),
        provider_call_config=provider_call_config,
        procedure_config_hash=procedure.config_identity_hash,
        graph_config=graph_config,
    )


def _probe_candidate(
    env: EnvSpec, *, candidate_id: str, template: str
) -> Candidate:
    """A prompt-template Mutation-Surface candidate for env ``env``.

    The candidate's ``base_ref`` is the env-scoped base binding; its payload
    assigns exactly the Mutation Surface field (the encoder
    ``user_prompt_template``) to ``template``.
    """
    return Candidate(
        candidate_id=candidate_id,
        base_ref=f"whetstone.env.{env.name}.base",
        payload={MUTATION_FIELD: template},
    )


def initial_candidate(env: EnvSpec) -> Candidate:
    """The naive-probe Initial Candidate (the floor prompt template)."""
    return _probe_candidate(
        env,
        candidate_id=f"{env.name}-naive",
        template=env.surface.naive_template,
    )


def ceiling_candidate(env: EnvSpec) -> Candidate:
    """The ceiling-probe reference candidate (the headroom prompt template)."""
    return _probe_candidate(
        env,
        candidate_id=f"{env.name}-ceiling",
        template=env.surface.ceiling_template,
    )


def render_prompt(
    env: EnvSpec, candidate: Candidate, instance: Instance
) -> str:
    """Render a candidate's template against a task's external inputs.

    The candidate's Mutation-Surface template text is rendered by the adapter
    probe surface (content-driven, never object identity) against the
    instance's public prompt inputs, producing the ``task.prompt`` Graph
    External Input. Rendering restricts to public inputs -- gold/oracle state
    can never be interpolated -- so a mutated or JSON-round-tripped template
    still renders (the c19 fidelity fix).
    """
    template = str(candidate.payload[MUTATION_FIELD])
    return env.surface.render(template, instance)


def valid_prompt_input_keys(
    env: EnvSpec, instance: Instance
) -> frozenset[str]:
    """The keyword fields a candidate template may reference for ``env``.

    Derived (never hardcoded per-env) from two authoritative sources already
    available at the optimizer seam:

    * ``instance.prompt_inputs`` keys -- the exact keyword inputs the render
      binds (``env.surface.render`` formats against these), and
    * the placeholder fields of the env's OWN naive/ceiling probe templates --
      known-good renders. This covers envs whose render translates keys (c19's
      surface renders ``{fact_line}`` from the public ``fact_type`` input, and
      c11's ceiling template carries example-JSON braces its literal-replace
      render tolerates), so a legitimate candidate mimicking the env's own
      templates is never spuriously rejected.

    A candidate placeholder outside this set (e.g. c22's ``{question}``) cannot
    be filled by the render and would raise the probe surface's loud
    ``KeyError`` -- so the intake validator rejects it before any eval spend.
    """
    keys: set[str] = set(dict(instance.prompt_inputs).keys())
    keys.update(template_placeholder_fields(env.surface.naive_template))
    keys.update(template_placeholder_fields(env.surface.ceiling_template))
    return frozenset(keys)


class PromptInputError(ValueError):
    """A QA candidate references prompt inputs unavailable to its tasks."""

    def __init__(self, offending: tuple[str, ...]) -> None:
        self.offending = offending
        super().__init__(
            "candidate template contains unavailable placeholders: "
            + ", ".join(offending)
        )


def validate_candidate_prompt(
    env: EnvSpec,
    candidate: Candidate,
    instances: tuple[Instance, ...],
) -> None:
    """Validate all placeholders before any provider call can be made."""
    template = str(candidate.payload[MUTATION_FIELD])
    offending: list[str] = []
    seen: set[str] = set()
    for instance in instances:
        invalid = invalid_template_placeholders(
            template,
            valid_prompt_input_keys(env, instance),
        )
        for field_name in invalid:
            if field_name not in seen:
                seen.add(field_name)
                offending.append(field_name)
    if offending:
        raise PromptInputError(tuple(offending))


def env_task_for(env: EnvSpec, instance: Instance) -> EnvTask:
    """Wrap an env instance as an :class:`EnvTask` (Graph External Inputs +
    evaluation inputs)."""
    return EnvTask.from_instance(env.name, instance)


__all__ = [
    "EVAL_NODE_ID",
    "LLM_NODE_ID",
    "PROMPT_EXTERNAL_INPUT",
    "PROVIDER_CALL_CONFIG_SCHEMA",
    "EnvRolloutDefinition",
    "PromptInputError",
    "build_graph_config",
    "build_provider_call_config",
    "build_rollout_definition",
    "ceiling_candidate",
    "env_task_for",
    "initial_candidate",
    "llm_eval_graph_definition",
    "render_prompt",
    "valid_prompt_input_keys",
    "validate_candidate_prompt",
]
