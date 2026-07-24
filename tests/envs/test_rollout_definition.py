"""The Rollout Definition role graph: one LLM Call Node -> one Eval Node."""

from __future__ import annotations

import pytest
from dr_graph import graph_hash

from whetstone.envs.registry import ENV_NAMES, env_spec
from whetstone.envs.rollout_definition import (
    EVAL_NODE_ID,
    LLM_NODE_ID,
    PROMPT_EXTERNAL_INPUT,
    build_provider_call_config,
    build_rollout_definition,
    ceiling_candidate,
    initial_candidate,
    render_prompt,
)
from whetstone.graph.nodes import (
    EVAL_NODE_TYPE,
    LLM_CALL_NODE_TYPE,
    eval_node_procedure_hash,
)
from whetstone.optimization.mutation import MUTATION_FIELD
from whetstone.optimization.schema import Candidate

_MODEL = "openai/gpt-5-nano"


def test_graph_has_one_llm_call_and_one_terminal_eval_node() -> None:
    rd = build_rollout_definition(env_spec("c22"), model=_MODEL)
    nodes = rd.graph_config.nodes
    types = [n.node_type for n in nodes]
    assert types.count(LLM_CALL_NODE_TYPE) == 1
    assert types.count(EVAL_NODE_TYPE) == 1
    assert rd.definition.terminal_node_id == EVAL_NODE_ID


def test_prompt_is_the_only_llm_input_source() -> None:
    rd = build_rollout_definition(env_spec("c22"), model=_MODEL)
    llm = next(n for n in rd.graph_config.nodes if n.node_id == LLM_NODE_ID)
    # The prompt Graph External Input is the sole Node Input Source.
    assert list(llm.input_sources) == ["prompt"]
    source = llm.input_sources["prompt"]
    assert source.kind.value == "graph_external"
    assert source.field == "prompt"
    assert PROMPT_EXTERNAL_INPUT == "task.prompt"


def test_eval_node_carries_the_procedure_config_reference() -> None:
    rd = build_rollout_definition(env_spec("c22"), model=_MODEL)
    ev = next(n for n in rd.graph_config.nodes if n.node_id == EVAL_NODE_ID)
    assert eval_node_procedure_hash(ev.variables) == rd.procedure_config_hash


def test_provider_config_change_changes_graph_hash() -> None:
    rd = build_rollout_definition(env_spec("c22"), model="model-a")
    other = build_rollout_definition(env_spec("c22"), model="model-b")
    assert rd.graph_hash != other.graph_hash


def test_procedure_change_changes_graph_hash() -> None:
    # Two different envs => different Evaluation Procedure Config identity =>
    # different Eval Node static Variable => different graph_hash.
    a = build_rollout_definition(env_spec("c22"), model=_MODEL)
    b = build_rollout_definition(env_spec("c11"), model=_MODEL)
    assert a.procedure_config_hash != b.procedure_config_hash
    assert a.graph_hash != b.graph_hash


def test_graph_hash_matches_native_dr_graph() -> None:
    rd = build_rollout_definition(env_spec("c22"), model=_MODEL)
    assert rd.graph_hash == graph_hash(rd.graph_config)


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_initial_and_ceiling_candidates_are_the_probe_templates(
    env_name: str,
) -> None:
    env = env_spec(env_name)
    ic = initial_candidate(env)
    cc = ceiling_candidate(env)
    # The Mutation Surface is the encoder user_prompt_template only.
    assert set(ic.payload) == {MUTATION_FIELD}
    assert ic.payload[MUTATION_FIELD] == env.surface.naive_template
    assert cc.payload[MUTATION_FIELD] == env.surface.ceiling_template
    assert ic.base_ref == cc.base_ref


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_render_uses_public_inputs_and_never_leaks_gold(
    env_name: str,
) -> None:
    env = env_spec(env_name)
    pool = env.generate_pool(n_per_stratum=1)
    inst = pool.instances[0]
    naive = render_prompt(env, initial_candidate(env), inst)
    ceiling = render_prompt(env, ceiling_candidate(env), inst)
    assert naive
    assert ceiling
    # Rendering is restricted to the instance's public prompt inputs: a
    # template that referenced ``{gold}`` would raise KeyError rather than
    # silently interpolate the oracle-only state. Prove that structurally by
    # rendering a gold-referencing template and expecting a loud failure.
    with pytest.raises(KeyError):
        env.surface.render("{gold}", inst)


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_template_survives_json_round_trip(env_name: str) -> None:
    # The Mutation Surface must be a genuinely serialization-stable template:
    # a Candidate persisted + reloaded via the Result Store
    # (model_validate_json -> value-equal but a fresh object) must still
    # render. This is the c19 fidelity fix (its env ProbePair dispatched by
    # object identity, so a round-tripped template raised KeyError); the check
    # runs for every env so no future env regresses to identity dispatch.
    env = env_spec(env_name)
    inst = env.generate_pool(n_per_stratum=1).instances[0]
    for candidate in (initial_candidate(env), ceiling_candidate(env)):
        reloaded = Candidate.model_validate_json(candidate.model_dump_json())
        assert (
            reloaded.payload[MUTATION_FIELD]
            == (candidate.payload[MUTATION_FIELD])
        )
        assert render_prompt(env, reloaded, inst) == (
            render_prompt(env, candidate, inst)
        )


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_mutated_template_still_renders(env_name: str) -> None:
    # An optimizer mutation of the surfaced template must render (the whole
    # point of the Mutation Surface). Prepend a marker to the naive template
    # and assert the rendered prompt carries it -- content-driven rendering,
    # never object identity.
    env = env_spec(env_name)
    inst = env.generate_pool(n_per_stratum=1).instances[0]
    naive = initial_candidate(env)
    marker = "OPTIMIZER-EDIT-MARKER\n"
    mutated = Candidate(
        candidate_id=f"{env.name}-mutated",
        base_ref=naive.base_ref,
        payload={MUTATION_FIELD: marker + naive.payload[MUTATION_FIELD]},
    )
    rendered = render_prompt(env, mutated, inst)
    assert marker.strip() in rendered


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_surface_render_matches_env_probe_bytes(env_name: str) -> None:
    # The adapter surface must render byte-for-byte identically to the env's
    # own committed ProbePair renderer across a whole pool (oracle fidelity is
    # unchanged by the c19 template rebinding). c19's env renderer is reached
    # via its ProbePair.render_naive / render_ceiling.
    env = env_spec(env_name)
    pool = env.generate_pool(n_per_stratum=1)
    for inst in pool.instances:
        assert render_prompt(env, initial_candidate(env), inst) == (
            env.probes.render_naive(inst)
        )
        assert render_prompt(env, ceiling_candidate(env), inst) == (
            env.probes.render_ceiling(inst)
        )


def test_provider_config_identity_is_the_llm_node_variable() -> None:
    config = build_provider_call_config(_MODEL)
    rd = build_rollout_definition(env_spec("c22"), model=_MODEL)
    llm = next(n for n in rd.graph_config.nodes if n.node_id == LLM_NODE_ID)
    ref = llm.variables["provider_call_config_ref"]
    assert ref["identity_hash"] == config.identity_hash
