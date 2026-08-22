"""Literal spellings of the Codex wire and environment formats.

These strings cross a process boundary into a foreign agent and into the
MCP server's own environment. Nothing derives them from Python field
names, so only a golden literal test catches silent drift.
"""

from __future__ import annotations

import json

from whetstone.core.identity import ImmutableJsonObject, TerminalFailure
from whetstone.optim.codex.adapter import (
    CODEX_ADAPTER_KEY,
    CODEX_ARTIFACT_RUN_MISMATCH_CODE,
    CODEX_LEASE_TOKEN_MISMATCH_CODE,
    CODEX_MCP_HOST_FAILED_CODE,
    CODEX_OUTPUT_ARTIFACT_SCHEMA,
    CODEX_SELECTION_CONTRACT_CODE,
    CODEX_SELECTION_UNEVALUATED_CODE,
    CODEX_SELECTION_UNSCORED_CODE,
    CODEX_UNREPORTED_EVALUATION_CODE,
    CODEX_WALL_BUDGET_EXCEEDED_CODE,
    CodexOutputArtifact,
)
from whetstone.optim.codex.mcp_bridge import (
    CODEX_EVAL_INPUT_FIELDS,
    CODEX_EVAL_OUTPUT_FIELDS,
    CODEX_EVAL_TOOL_NAME,
    McpResultKey,
)
from whetstone.optim.codex.control import CodexControl
from whetstone.optim.codex.mcp_environment import McpEnvironmentKey
from whetstone.optim.codex.containment import CODEX_DENIED_FEATURES
from whetstone.optim.codex.adapter import (
    CODEX_ARTIFACT_AGENT_FIELDS,
    codex_output_schema,
)
from whetstone.optim.codex.runner import build_codex_command
from whetstone.optim.codex.step_contract import (
    CODEX_TOOL_CALLS_BUDGET_LABEL,
)
from whetstone.optim.tools.contracts import (
    RefusalClass,
    ToolCall,
    ToolCallRef,
    ToolRefusal,
    ToolResult,
    tool_call_reference,
)
from whetstone.optim.tools.evaluator import (
    TOOL_EVAL_FAILURE_EVIDENCE_CODE,
    TOOL_EVAL_UNEXPECTED_RESULT_CODE,
)
from whetstone.optim.tools.execution import TOOL_EVALUATION_REJECTED_CODE


def test_the_mcp_environment_keys_are_pinned() -> None:
    assert {member.name: member.value for member in McpEnvironmentKey} == {
        "SQLITE_PATH": "WS_MCP_SQLITE_PATH",
        "TOOL_CONFIG": "WS_MCP_TOOL_CONFIG",
        "CAPACITY_BINDING": "WS_MCP_CAPACITY_BINDING",
        "RUNTIME_CONFIG": "WS_MCP_RUNTIME_CONFIG",
        "RUNTIME_CONFIG_CLASS": "WS_MCP_RUNTIME_CONFIG_CLASS",
        "REWARD_POLICY": "WS_MCP_REWARD_POLICY",
        "RUN_LEASE_TOKEN": "WS_MCP_RUN_LEASE_TOKEN",
        "RUN_LEASE_BINDING": "WS_MCP_RUN_LEASE_BINDING",
    }


def test_the_mcp_result_payload_keys_are_pinned() -> None:
    assert {member.name: member.value for member in McpResultKey} == {
        "REFUSED": "refused",
        "CALL_ID": "call_id",
        "REFUSAL_CLASS": "refusal_class",
        "REASON": "reason",
        "TERMINAL_FAILURE": "terminal_failure",
        "OUTPUT": "output",
        "REWARD": "reward",
    }


def test_the_tool_name_and_field_tuples_are_pinned() -> None:
    assert CODEX_EVAL_TOOL_NAME == "evaluate_candidate"
    assert CODEX_EVAL_INPUT_FIELDS == (
        "base_ref",
        "model_route",
        "template",
    )
    assert CODEX_EVAL_OUTPUT_FIELDS == (
        "evaluation_evidence_ref",
        "output_artifact_ref",
        "per_task_values",
        "per_task_counts",
        "row_accounting",
    )


def test_the_persisted_schema_and_code_literals_are_pinned() -> None:
    assert CODEX_ADAPTER_KEY == "codex"
    assert CODEX_OUTPUT_ARTIFACT_SCHEMA == "whetstone.codex_output_artifact"
    assert CODEX_SELECTION_UNEVALUATED_CODE == "codex_selection_unevaluated"
    assert CODEX_LEASE_TOKEN_MISMATCH_CODE == "codex_lease_token_mismatch"
    assert (
        CODEX_ARTIFACT_RUN_MISMATCH_CODE == "codex_artifact_run_mismatch"
    )
    assert CODEX_MCP_HOST_FAILED_CODE == "codex_mcp_host_failed"
    assert CODEX_SELECTION_CONTRACT_CODE == "codex_selection_contract"
    assert CODEX_SELECTION_UNSCORED_CODE == "codex_selection_unscored"
    assert (
        CODEX_UNREPORTED_EVALUATION_CODE
        == "codex_unreported_evaluation"
    )
    assert (
        CODEX_WALL_BUDGET_EXCEEDED_CODE == "codex_wall_budget_exceeded"
    )
    assert TOOL_EVAL_FAILURE_EVIDENCE_CODE == "tool_eval_failure_evidence"
    assert TOOL_EVAL_UNEXPECTED_RESULT_CODE == "tool_eval_unexpected_result"
    assert TOOL_EVALUATION_REJECTED_CODE == "tool_evaluation_rejected"


def test_the_budget_label_the_ledger_reads_is_pinned() -> None:
    # _IssuedToolCallLedger keys its hard limit and its budget_delta
    # injection on this exact literal; any other spelling silently
    # disables both.
    assert CODEX_TOOL_CALLS_BUDGET_LABEL == "tool_calls"


def test_the_output_artifact_field_set_is_pinned() -> None:
    assert set(CodexOutputArtifact.model_fields) == {
        "run_id",
        "evaluated_call_ids",
        "selected_call_id",
        "lease_token_hash",
        "conversation_evidence",
        "control_cost",
    }
    # The artifact carries no candidate body: a template that was never
    # evaluated through the tool cannot be returned.
    assert "proposals" not in CodexOutputArtifact.model_fields


def _refusal_result(call_ref: ToolCallRef) -> ToolResult:
    return ToolResult(
        call=call_ref,
        refusal=ToolRefusal(
            refusal_class=RefusalClass.CAPACITY,
            reason="Tool Capacity exhausted",
        ),
    )


def test_a_refusal_payload_uses_the_pinned_keys(codex_tool_config) -> None:
    from whetstone.optim.codex.mcp_bridge import tool_result_to_mcp_result
    from whetstone.optim.tools.contracts import (
        ToolCapacityScope,
        tool_capacity_binding,
        tool_config_reference,
    )

    config, subject_ref = codex_tool_config
    call = ToolCall(
        call_id="c1",
        tool_config=tool_config_reference(config),
        capacity_binding=tool_capacity_binding(
            ToolCapacityScope.RUN, subject_ref
        ),
        args=ImmutableJsonObject(
            {
                "base_ref": {
                    "schema_name": "whetstone.candidate",
                    "content_hash": "0" * 64,
                },
                "model_route": "route",
                "template": "hello {prompt}",
            }
        ),
    )
    result = _refusal_result(tool_call_reference(call))

    mcp_result = tool_result_to_mcp_result(result)

    assert mcp_result.is_error is True
    assert set(mcp_result.structured_content) == {
        "refused",
        "call_id",
        "refusal_class",
        "reason",
    }
    assert mcp_result.structured_content["refused"] is True
    assert mcp_result.structured_content["refusal_class"] == "capacity"
    # The text content is the same payload, canonically ordered.
    assert json.loads(mcp_result.content[0].text) == (
        mcp_result.structured_content
    )


def test_a_terminal_failure_payload_uses_the_pinned_keys(
    codex_tool_config,
) -> None:
    from whetstone.optim.codex.mcp_bridge import tool_result_to_mcp_result
    from whetstone.optim.tools.contracts import (
        ToolCapacityScope,
        tool_capacity_binding,
        tool_config_reference,
    )

    config, subject_ref = codex_tool_config
    call = ToolCall(
        call_id="c2",
        tool_config=tool_config_reference(config),
        capacity_binding=tool_capacity_binding(
            ToolCapacityScope.RUN, subject_ref
        ),
        args=ImmutableJsonObject(
            {
                "base_ref": {
                    "schema_name": "whetstone.candidate",
                    "content_hash": "0" * 64,
                },
                "model_route": "route",
                "template": "hello {prompt}",
            }
        ),
    )
    result = ToolResult(
        call=tool_call_reference(call),
        terminal_failure=TerminalFailure(
            code=TOOL_EVAL_UNEXPECTED_RESULT_CODE,
            message="Tool evaluation produced an unrecognized Eval Result",
        ),
        provenance_ordinal=1,
    )

    mcp_result = tool_result_to_mcp_result(result)

    assert mcp_result.is_error is True
    assert set(mcp_result.structured_content) == {
        "refused",
        "call_id",
        "terminal_failure",
    }
    assert mcp_result.structured_content["refused"] is False
    assert (
        mcp_result.structured_content["terminal_failure"]["code"]
        == TOOL_EVAL_UNEXPECTED_RESULT_CODE
    )


def test_the_reasoning_effort_reaches_the_cli_as_a_config_override() -> None:
    """Every identity-bearing control field must reach the invocation.

    The Codex CLI has no reasoning-effort flag; it is a config key, and
    the run passes ``--strict-config``, so a misspelling fails the launch
    rather than being silently dropped. The literal is pinned here
    because nothing derives it and it crosses into a foreign binary.
    """
    argv = build_codex_command(
        prompt="go",
        codex_binary="/usr/bin/codex",
        model="toy-model",
        reasoning_effort="high",
        mcp_endpoint=None,
        output_schema_path="/tmp/schema.json",
        output_artifact_path="/tmp/last.json",
        working_directory="/tmp/work",
    )

    assert "--strict-config" in argv
    override = 'model_reasoning_effort="high"'
    assert override in argv
    assert argv[argv.index(override) - 1] == "-c"


def test_an_empty_reasoning_effort_adds_no_override() -> None:
    argv = build_codex_command(
        prompt="go",
        codex_binary="/usr/bin/codex",
        model="toy-model",
        reasoning_effort="",
        mcp_endpoint=None,
        output_schema_path="/tmp/schema.json",
        output_artifact_path="/tmp/last.json",
        working_directory="/tmp/work",
    )

    assert not any("model_reasoning_effort" in entry for entry in argv)


def test_web_search_is_disabled_by_config_key_not_by_feature_flags() -> None:
    """Codex 0.148 enables web search by default; the flags no longer gate it.

    ``web_search_cached`` and ``web_search_request`` were in the deny list
    as ``--disable`` flags. Against the real 0.148 CLI they are
    *deprecated*: they disable nothing, and each one makes the CLI emit a
    deprecation ``error`` item into the JSONL transcript the adapter
    parses. Web search stayed on, so a contained agent could still reach
    the open web.

    The top-level ``web_search`` config key is what actually turns it off,
    and ``--strict-config`` makes a misspelling fatal rather than silent.
    Both halves are pinned: the key is present, and the deprecated flags
    are gone.
    """
    argv = build_codex_command(
        prompt="go",
        codex_binary="/usr/bin/codex",
        model="toy-model",
        reasoning_effort="",
        mcp_endpoint=None,
        output_schema_path="/tmp/schema.json",
        output_artifact_path="/tmp/last.json",
        working_directory="/tmp/work",
    )

    assert 'web_search="disabled"' in argv
    assert argv[argv.index('web_search="disabled"') - 1] == "-c"
    # The deprecated flags must not come back: passing them is what
    # produced the transcript noise and the false sense of containment.
    assert "web_search_cached" not in argv
    assert "web_search_request" not in argv
    assert "web_search_cached" not in CODEX_DENIED_FEATURES
    assert "web_search_request" not in CODEX_DENIED_FEATURES


def test_the_output_schema_satisfies_the_structured_output_validator() -> None:
    """The schema handed to the CLI is stricter than plain JSON Schema.

    The OpenAI structured-output validator requires every object to set
    ``additionalProperties: false`` and every property to appear in
    ``required``. The schema derived from ``CodexOutputArtifact`` did not:
    its two ``dict[str, Any]`` fields emitted
    ``additionalProperties: true``, and the real API rejected the whole
    run with ``invalid_json_schema`` before the agent produced a token.

    The fake CLI reads the schema for its two constants and never
    validates it, so only a real run surfaced this. These assertions are
    the validator's rules, checked locally.
    """
    schema = codex_output_schema(run_id="r1", lease_token_hash="h1")

    def check(node: dict) -> None:
        if node.get("type") != "object":
            return
        assert node.get("additionalProperties") is False, (
            f"nested object without additionalProperties=false: {node}"
        )
        properties = node.get("properties", {})
        assert set(node.get("required", [])) == set(properties), (
            "every property must be required; optional fields are "
            f"expressed as nullable types: {node}"
        )
        for child in properties.values():
            check(child)

    check(schema)
    # The agent is asked only for the fields it actually decides.
    assert set(schema["properties"]) == set(CODEX_ARTIFACT_AGENT_FIELDS)
    # conversation_evidence is overwritten by the runner with whetstone's
    # own process evidence, and nothing reads control_cost, so asking the
    # model for either invites invented evidence.
    assert "conversation_evidence" not in schema["properties"]
    assert "control_cost" not in schema["properties"]
    assert schema["properties"]["run_id"]["const"] == "r1"
    assert schema["properties"]["lease_token_hash"]["const"] == "h1"
    # Null selection is how the agent retains the seed, so it must be a
    # nullable type rather than an omitted property.
    assert "null" in schema["properties"]["selected_call_id"]["type"]


def test_mcp_tools_are_pre_approved_for_the_server_whetstone_hosts() -> None:
    """``codex exec`` approves nothing interactively.

    Its session approval policy is ``never``. Under the previous
    ``auto`` mode the CLI still routed each MCP tool call through
    approval, so every call failed with "MCP tool call requires approval,
    but approval policy is never" and the agent could not evaluate at
    all. The scripted fake CLI calls the endpoint directly and never
    consults an approval policy, so it could not catch this.

    ``approve`` pre-approves this one server's tools. The evaluation tool
    is admission-gated and capacity-capped server-side, so the CLI's
    prompt adds no safety -- only a deadlock.
    """
    from whetstone.optim.codex.mcp_host import CodexMcpEndpoint

    argv = build_codex_command(
        prompt="go",
        codex_binary="/usr/bin/codex",
        model="toy-model",
        reasoning_effort="",
        mcp_endpoint=CodexMcpEndpoint(
            url="http://127.0.0.1:4242/mcp", auth_token="run-token"
        ),
        output_schema_path="/tmp/schema.json",
        output_artifact_path="/tmp/last.json",
        working_directory="/tmp/work",
    )

    assert (
        'mcp_servers.whetstone.default_tools_approval_mode="approve"' in argv
    )


def test_code_mode_host_is_not_denied_because_it_routes_mcp_calls() -> None:
    """Denying it made the evaluation tool invisible to the agent.

    ``code_mode_host`` reads like an agent capability, but from Codex
    0.148 it is the host that routes MCP tool calls. With it disabled the
    real agent is told "Code Mode is unavailable" and reports that no
    server-specific MCP tools exist -- so ``evaluate_candidate`` cannot be
    called and the Step can only return an empty artifact.

    Code Mode itself stays denied through the other two flags.
    """
    assert "code_mode_host" not in CODEX_DENIED_FEATURES
    assert "code_mode" in CODEX_DENIED_FEATURES
    assert "code_mode_only" in CODEX_DENIED_FEATURES


def test_the_control_carries_no_field_the_cli_cannot_honor() -> None:
    """A turn cap and a sampling seed are not Codex CLI concepts.

    ``codex exec`` exposes neither, and ``--strict-config`` rejects
    ``max_turns`` and ``seed`` as unknown configuration fields. Carrying
    them would mean two runs with different identities and different
    recorded hyperparameters executing byte-identical invocations.
    """
    fields = set(CodexControl.model_fields)

    assert "max_turns" not in fields
    assert "seed" not in fields
    assert "reasoning_effort" in fields


def test_the_agent_is_given_an_endpoint_url_and_never_the_store() -> None:
    """The agent connects to a server; it never spawns one.

    Spawning the server as a child of the sandboxed agent gave the agent
    the server's profile, and the server must write the whetstone store
    -- the durable ledger, and the admission-capacity rows that cap paid
    evaluations. So the CLI is configured with a URL, and the store path
    and the server's own configuration never enter its argv at all.
    """
    from whetstone.optim.codex.mcp_host import CodexMcpEndpoint
    from whetstone.optim.codex.runner import CODEX_MCP_TOKEN_ENV

    argv = build_codex_command(
        prompt="go",
        codex_binary="/usr/bin/codex",
        model="toy-model",
        reasoning_effort="low",
        mcp_endpoint=CodexMcpEndpoint(
            url="http://127.0.0.1:4242/mcp", auth_token="run-token"
        ),
        output_schema_path="/tmp/schema.json",
        output_artifact_path="/tmp/last.json",
        working_directory="/tmp/work",
    )
    joined = " ".join(argv)

    assert 'mcp_servers.whetstone.url="http://127.0.0.1:4242/mcp"' in argv
    assert (
        f'mcp_servers.whetstone.bearer_token_env_var="{CODEX_MCP_TOKEN_ENV}"'
        in argv
    )
    # No command, no server module, and no environment block: the agent
    # is given nothing it could use to start a store-writing process.
    assert "mcp_servers.whetstone.command" not in joined
    assert "mcp_servers.whetstone.env." not in joined
    # The token travels in the environment, not in a world-readable argv.
    assert "run-token" not in joined


def test_the_bearer_token_variable_is_pinned() -> None:
    from whetstone.optim.codex.runner import CODEX_MCP_TOKEN_ENV

    assert CODEX_MCP_TOKEN_ENV == "WS_MCP_BEARER_TOKEN"


def test_the_mcp_endpoint_path_and_auth_scheme_are_pinned() -> None:
    from whetstone.optim.codex.mcp_host import (
        CODEX_MCP_AUTH_HEADER,
        CODEX_MCP_AUTH_SCHEME,
        CODEX_MCP_HTTP_PATH,
    )

    assert CODEX_MCP_HTTP_PATH == "/mcp"
    assert CODEX_MCP_AUTH_HEADER == "authorization"
    assert CODEX_MCP_AUTH_SCHEME == "Bearer"


def test_the_prompt_names_the_fixed_tool_arguments_the_agent_cannot_guess(
    tmp_path,
) -> None:
    """Two arguments are fixed values that appear nowhere else.

    ``model_route`` must equal the engine's exact Provider Call Config
    route, and ``base_ref`` must be the run's seed candidate reference.
    Neither appears in the serialized step request the prompt embeds, so a
    real agent assembled a route object out of the model/reasoning-effort
    fields and invented a base hash. Both calls were refused *after*
    admission, so the run paid capacity for calls that could never score.

    The scripted fake CLI is handed both values in its transcript, so it
    could never catch this. Here the prompt itself is the assertion.
    """
    from dr_store.sync import open_sqlite

    from tests.codex_support import (
        toy_codex_control,
        toy_codex_run,
        toy_codex_step_request,
    )
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
    from whetstone.experiment.candidate import candidate_reference
    from whetstone.optim.codex.runner import _default_prompt

    sqlite_path = str((tmp_path / "prompt.sqlite").resolve())
    with open_sqlite(sqlite_path) as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(store)
        control = toy_codex_control(engine=engine, max_tool_calls=2)
        run, config, candidate = toy_codex_run(
            control=control, engine=engine
        )
        request = toy_codex_step_request(
            control=control, run=run, candidate=candidate
        )
        route = engine.expected_model_route()
        base_ref = candidate_reference(candidate).record_ref

        prompt = _default_prompt(
            request,
            tool_name=config.tool_name,
            lease_token_hash="h" * 64,
            max_tool_calls=2,
            model_route=route,
            base_ref=json.dumps(
                {
                    "schema_name": base_ref.schema_name,
                    "content_hash": base_ref.content_hash,
                },
                sort_keys=True,
            ),
        )

        assert route in prompt, (
            "the agent is never told the one model route the evaluator "
            "accepts, so it must guess and every guess is refused"
        )
        assert base_ref.content_hash in prompt
        assert base_ref.schema_name in prompt

        # The seed-retention clause, pinned literally.
        #
        # The artifact accepts two statements of "the seed is still
        # best": a null selection, and selecting an evaluated call whose
        # template is the seed's own. A real agent that evaluated the
        # seed reached for the second, and the prompt documented only the
        # first -- never saying the two were the same statement. The
        # adapter now reads both as seed-retained, so the wrong one no
        # longer discards the Step's paid evaluations; this clause is
        # what stops the agent from picking it in the first place.
        #
        # Substring assertions on ``model_route`` and ``base_ref`` above
        # cannot catch drift here: this clause names no value the test
        # can rebuild, so only its literal text pins it.
        assert (
            "Set selected_call_id to null to keep the run's seed "
            "candidate.\n"
            "If your evaluations leave the seed candidate best, set "
            "selected_call_id to null. Do not re-select a call whose "
            "template is the seed's own template; null is how this "
            "contract says the seed was retained.\n"
        ) in prompt


def test_the_advertised_tool_schema_pins_the_model_route(tmp_path) -> None:
    """The prompt says it and the tool contract enforces the shape.

    ``model_route`` is advertised as a string ``const``, so an agent
    reading the schema cannot send the route object it otherwise invents.
    The evaluator still validates -- this narrows what a well-behaved
    agent sends, it does not replace the check.
    """
    import asyncio
    from datetime import timedelta

    from dr_store.sync import open_sqlite

    from tests.codex_support import (
        toy_capacity_binding,
        toy_codex_control,
        toy_codex_run,
    )
    from whetstone.core.leasing import EffectLeaseAuthority, ReplayPolicy
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
    from whetstone.optim.codex.mcp_bridge import EvaluateCandidateServer
    from whetstone.optim.tools.evaluator import EngineToolEvaluator
    from whetstone.optim.tools.execution import EvaluatingToolExecutor
    from whetstone.optim.tools.facade import (
        ToolAdmissionAuthority,
        ToolCallStore,
    )

    sqlite_path = str((tmp_path / "schema.sqlite").resolve())
    with open_sqlite(sqlite_path) as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(store)
        control = toy_codex_control(engine=engine, max_tool_calls=2)
        run, config, _candidate = toy_codex_run(
            control=control, engine=engine
        )
        authority = EffectLeaseAuthority.sqlite(sqlite_path)
        tool_store = ToolCallStore(
            store, ToolAdmissionAuthority.sqlite(sqlite_path), authority
        )
        executor = EvaluatingToolExecutor(
            EngineToolEvaluator(engine),
            engine.reward_policy,
            authority,
            owner_id="wire-golden-owner",
            replay_policy=ReplayPolicy.IDEMPOTENT,
            lease_duration=timedelta(minutes=5),
        )
        server = EvaluateCandidateServer(
            handle=executor.runtime_handle(
                config, tool_store, toy_capacity_binding(run)
            ),
            expected_model_route=engine.expected_model_route(),
        )

        tools = asyncio.run(server.list_tools())

        # The tool must still be registered: building the schema override
        # once silently dropped it, and every call became "Unknown tool".
        assert [tool.name for tool in tools] == [str(config.tool_name)]
        schema = tools[0].input_schema
        assert schema["additionalProperties"] is False
        route_schema = schema["properties"]["model_route"]
        assert route_schema["const"] == engine.expected_model_route()
        assert route_schema["type"] == "string"
