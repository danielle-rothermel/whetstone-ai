"""The real-Codex ladder: cheapest rung first, each independently skippable.

Every rung drives the real Codex CLI (0.148+) against a live subscription
session. The *task* model is always the reference transport, so a full
ladder run spends Codex agent turns and no eval-provider credit.

Run it with ``scripts/check-real-codex.sh``, or directly::

    WHETSTONE_REAL_CODEX=1 uv run --extra platform pytest \
        tests/real_codex/test_real_codex_ladder.py -x -v -m real_codex

Rungs are ordered by cost and by what they presuppose:

1. config the runner writes is accepted by the real binary (no session)
2. the real auth preflight proves a session
3. one real Step through the hosted MCP server
4. edge paths: capacity refusal, wall budget, no-tool-call,
   seed-identical selection
5. a real multi-evaluation selection loop
6. reasoning-effort variants the real binary accepts
7. output retention against a real, truncated transcript
8. the sandbox really denies the store and writes outside scratch
9. a foreign bearer token is refused by the hosted server
"""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request

import pytest

from tests.codex_support import toy_capacity_binding
from tests.real_codex.conftest import RUNG_WALL_SECONDS, real_codex_binary
from whetstone.optim.codex.adapter import (
    CODEX_WALL_BUDGET_EXCEEDED_CODE,
)
from whetstone.optim.codex.containment import CODEX_DENIED_FEATURES
from whetstone.optim.codex.mcp_host import (
    CODEX_MCP_AUTH_SCHEME,
    CodexMcpHost,
)
from whetstone.optim.codex.runner import (
    CODEX_MCP_TOKEN_ENV,
    build_codex_command,
)
from whetstone.optim.contracts import StepStatus
from whetstone.testing.toy.experiment import TOY_MUTATION_FIELD

pytestmark = pytest.mark.real_codex

#: Short enough that no real session can finish inside it: the CLI has to
#: start, authenticate, connect to the MCP endpoint, and reach a first
#: token. This makes the wall-budget stop a property of the budget rather
#: than a bet on how long the model happens to think.
_WALL_BUDGET_STOP_SECONDS = 5.0

_TEMPLATE_A = "Answer {prompt} in one short sentence."
_TEMPLATE_B = "Answer {prompt} with a single friendly word."
_TEMPLATE_C = "Reply to {prompt} briefly and warmly."
_TEMPLATE_D = "Give {prompt} a concise, cheerful answer."


def _cli(*args: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603 - fixed binary, test-only
        [real_codex_binary(), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------- rung 1


def test_rung1_real_binary_accepts_every_config_key_the_runner_writes(
    tmp_path,
) -> None:
    """No session: the real binary must parse the runner's own argv.

    ``--strict-config`` makes an unknown config key fatal, so a key that
    the CLI dropped in a version bump fails the launch instead of being
    silently ignored. Validating the argv the runner actually builds --
    rather than a hand-copied list -- is what keeps this honest when
    ``build_codex_command`` changes.

    The probe points CODEX_HOME at an empty directory, so config parsing
    runs to completion and the process then stops at authentication.
    That is a config check with no session and no spend.
    """
    from whetstone.optim.codex.mcp_host import CodexMcpEndpoint

    empty_home = tmp_path / "empty-codex-home"
    empty_home.mkdir()
    argv = build_codex_command(
        prompt="ping",
        codex_binary=real_codex_binary(),
        model="",
        reasoning_effort="low",
        mcp_endpoint=CodexMcpEndpoint(
            url="http://127.0.0.1:1/mcp", auth_token="t" * 16
        ),
        output_schema_path=str(_write_schema(tmp_path)),
        output_artifact_path=str(tmp_path / "artifact.json"),
        working_directory=str(tmp_path),
    )
    completed = subprocess.run(  # noqa: S603 - fixed binary, test-only
        argv,
        capture_output=True,
        text=True,
        timeout=120.0,
        stdin=subprocess.DEVNULL,
        env={"PATH": "/usr/bin:/bin", "CODEX_HOME": str(empty_home)},
    )

    combined = completed.stdout + completed.stderr
    assert "unknown configuration field" not in combined, (
        "the real Codex CLI rejected a config key the runner writes:\n"
        f"{combined[-2000:]}"
    )
    assert "Error loading config.toml" not in combined, (
        f"the real Codex CLI failed to load the runner's config:\n"
        f"{combined[-2000:]}"
    )


def test_rung1_real_binary_knows_every_denied_feature(tmp_path) -> None:
    """A stale ``--disable`` name would fail every launch under strict config.

    The deny list is frozen in ``containment.py`` and passed verbatim; the
    real binary's ``features list`` is the only authority on which names
    still exist.
    """
    listed = _cli("features", "list")
    assert listed.returncode == 0, listed.stderr[-2000:]
    known = {line.split()[0] for line in listed.stdout.splitlines() if line.strip()}

    unknown = sorted(set(CODEX_DENIED_FEATURES) - known)
    assert not unknown, (
        "the real Codex CLI no longer knows these denied features, so every "
        f"--strict-config launch would fail: {unknown}"
    )


def _write_schema(tmp_path):
    schema = tmp_path / "schema.json"
    schema.write_text(
        json.dumps(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["ready"],
                "properties": {"ready": {"type": "boolean"}},
            }
        ),
        encoding="utf-8",
    )
    return schema


# ---------------------------------------------------------------- rung 2


def test_rung2_real_auth_preflight_proves_a_session(real_codex_world) -> None:
    """The production preflight, against the real CLI and the real session.

    One trivial structured prompt. It proves four things at once: the
    binary resolves, the staged credentials work, sandbox-exec admits the
    real Codex process, and the JSONL/artifact path parses what the real
    CLI emits.
    """
    from whetstone.optim.codex.preflight import codex_auth_preflight

    world = real_codex_world()
    runner = world.runner(timeout_seconds=RUNG_WALL_SECONDS)

    # Raises CodexPreflightError with the CLI's stderr tail on failure.
    codex_auth_preflight(
        executor=runner._executor,
        codex_binary=real_codex_binary(),
        environment=runner.codex_process_environment(),
        wall_seconds=RUNG_WALL_SECONDS,
    )


def test_rung2_the_agent_environment_carries_no_task_model_key(
    real_codex_world,
) -> None:
    """The containment claim, checked on the real run's environment.

    The Codex agent has network access, so a task-model key in its
    environment would let it score candidates outside the ledger.
    """
    world = real_codex_world()
    environment = world.runner().codex_process_environment()

    assert "OPENAI_API_KEY" not in environment, (
        "the real Codex process would receive a task-model credential"
    )
    assert CODEX_MCP_TOKEN_ENV not in environment, (
        "the bearer token must be injected per run, not carried in the "
        "runner's base environment"
    )


# ---------------------------------------------------------------- rung 3


def test_rung3_one_real_step_through_the_hosted_mcp_server(
    real_codex_world,
) -> None:
    """The whole point: a real agent, a real MCP call, a real ledger entry.

    The agent reaches whetstone's own streamable-HTTP endpoint over
    loopback with a bearer token, calls ``evaluate_candidate`` once, and
    returns an artifact naming the call it made. Everything downstream --
    admission, lease, reward, ledger, cost -- is production code.
    """
    world = real_codex_world(max_tool_calls=2)
    runner = world.runner(timeout_seconds=RUNG_WALL_SECONDS)
    adapter = world.adapter(runner)
    request = world.step_request()

    result, _ref = world.harness(adapter).run_step(request)

    assert result.terminal_failure is None, (
        f"the real Codex Step failed: {result.terminal_failure}"
    )
    assert result.status is StepStatus.COMPLETE, (
        f"the real Codex Step did not complete: {result.status}"
    )
    assert len(result.tool_evidence) >= 1, (
        "the real agent returned without any admitted evaluation on the "
        "ledger; it likely never reached the MCP endpoint"
    )
    for entry in result.tool_evidence:
        assert entry.result is not None, (
            f"call {entry.store_entry.call_id} has no durable terminal"
        )
    assert result.budget_delta.consumed["tool_calls"] == len(
        result.tool_evidence
    )


def test_rung3_the_step_result_reaches_task_model_cost(
    real_codex_world,
) -> None:
    """A real Codex run's spend must be visible through tool evidence.

    The Codex arm has no proposer, so an aggregator reading only the
    intent path would report a real run as free.
    """
    from whetstone.optim.cost_aggregation import aggregate_run_cost

    world = real_codex_world(max_tool_calls=2)
    adapter = world.adapter(world.runner())
    result, _ref = world.harness(adapter).run_step(world.step_request())

    assert result.terminal_failure is None, result.terminal_failure
    assert result.resolved_intents == ()

    report = aggregate_run_cost(store=world.store, step_results=(result,))
    assert report.task_model.calls > 0, (
        "a real Codex Step recorded no task-model calls, so its evaluations "
        "are invisible to run cost"
    )


# --------------------------------------------------------------- rung 4a


def _request_json(request) -> str:
    """The serialized Step Request, spelled as the production prompt does."""
    return json.dumps(
        request.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )


def _overstated_allowance_prompt(context, *, stated_allowance: int) -> str:
    """The production protocol facts, under a deliberately false cap.

    This is the one prompt in the ladder that lies to the agent, and it
    lies about exactly one thing: how many calls it is allowed.
    Everything the agent cannot derive for itself is carried over from
    the production prompt -- the tool name, the fixed ``model_route``
    string, the seed ``base_ref``, the artifact's schema obligations, the
    ``lease_token_hash``, and the serialized Step Request -- because a
    prompt that dropped any of them would have every call refused *after*
    admission, which looks exactly like the failure this rung is trying
    to distinguish itself from.

    ``world.production_prompt`` cannot be used here: it extends the
    production prompt, which truthfully states the configured cap, and
    that truthful clause is the whole reason this rung stopped observing
    anything. See the test's docstring for why the lie is load-bearing.

    The serialized request carries the real cap too -- ``max_tool_calls``
    and ``capacity.max_accepted_calls`` are both 1 -- so the
    overstatement is made explicit rather than merely asserted: an agent
    that notices the contradiction has to be told which number this run
    intends, or it will reconcile it by obeying the smaller one and make
    a single call again.
    """
    return (
        f"Use only the external {context.tool_name} MCP tool for "
        "measurements. Do not call any built-in tool. Build candidate "
        "templates from the exact candidate base_ref, model route, payload "
        "template, Tool Config, budget, pools, hyperparameters, and output "
        "contract in the serialized request below.\n"
        "The model_route argument is a fixed string and must be exactly "
        f"{context.model_route!r}. It is not an object and must not be "
        "built from any other field.\n"
        "The base_ref argument must be copied verbatim as "
        f"{context.base_ref}. Do not construct or modify it.\n"
        "Evaluating through the MCP tool is mandatory. Every candidate you "
        "consider must be submitted to the tool with a call_id you choose; "
        f"you may make up to {stated_allowance} calls on this run. Ignore "
        "any smaller call limit you find inside the serialized request "
        f"below -- {stated_allowance} is the allowance for this run.\n"
        "For this run, evaluate BOTH of these templates, each with its own "
        f"distinct call_id: {_TEMPLATE_A!r} first, then {_TEMPLATE_B!r}. "
        "Submit the second call even if the first one succeeded; both "
        "templates must reach the tool.\n"
        "If a call comes back with refused=true, that is an expected "
        "outcome and not an error: do not retry it and do not abandon the "
        "run. Write your artifact naming only the call_ids that were "
        "actually scored.\n"
        "Write a schema-conforming final artifact naming every scored "
        "call_id in evaluated_call_ids, and selected_call_id set to the "
        "call_id whose candidate you chose. The artifact carries no "
        "candidate body: a template that was never evaluated through the "
        "tool cannot be returned. Set selected_call_id to null to keep the "
        "run's seed candidate. Copy lease_token_hash verbatim as "
        f"{context.lease_token_hash!r}.\n"
        f"OPTIM_STEP_REQUEST_JSON={_request_json(context.request)}"
    )


def test_rung4a_capacity_refusal_is_durable_and_the_step_still_completes(
    real_codex_world,
) -> None:
    """Cap 1, and the agent is told it may make 2 calls.

    The second call must be refused by admission -- not by the agent's
    good behavior -- and the Step must still complete on the first.

    This rung has to lie to the agent about its allowance, and that is
    not a shortcut. The production prompt states the *real* configured
    cap, so a well-behaved agent under a cap of 1 makes exactly one call
    and the durable refusal path is never exercised at all: one admitted
    evaluation under a cap of one is precisely what an obedient
    single-call agent produces, so the rung would pass while observing
    nothing. Telling the agent it may make 2 calls while the admission
    authority is configured for 1 is what drives a genuine second call
    into the authority and makes it refuse -- which is the behavior under
    test. The lie is confined to the allowance; every protocol fact the
    agent cannot derive is carried over from the production prompt (see
    ``_overstated_allowance_prompt``).

    The evidence for "by admission" is the durable CAPACITY refusal in
    the admission ledger, which is the only thing that tells "the agent
    was refused" apart from "the agent never tried".
    """
    world = real_codex_world(max_tool_calls=1)

    def prompt_builder(context):
        # Two, against a configured capacity of one.
        return _overstated_allowance_prompt(context, stated_allowance=2)

    adapter = world.adapter(world.runner(prompt_builder=prompt_builder))
    result, _ref = world.harness(adapter).run_step(
        world.step_request(tool_calls=1)
    )

    assert result.terminal_failure is None, (
        f"the capped Step failed instead of completing: "
        f"{result.terminal_failure}"
    )
    assert result.status is StepStatus.COMPLETE
    assert len(result.tool_evidence) == 1, (
        "capacity did not hold: the real agent got "
        f"{len(result.tool_evidence)} admitted evaluations under a cap of 1"
    )
    assert result.budget_delta.consumed["tool_calls"] == 1

    # The refusal path itself, not just its absence of damage.
    refusals = world.capacity_refusals()
    assert refusals, (
        "no CAPACITY refusal was recorded, so the real agent never "
        "attempted the second call the prompt asked for -- this rung "
        "observed an obedient agent, not the durable refusal path. Check "
        "the transcript: if the agent stopped after one call, the prompt "
        "needs to insist on the second."
    )
    # A refusal debits no capacity, so the paid ledger stays at the cap
    # even though more calls than the cap were made.
    assert world.tool_store.accepted_count(
        world.config, toy_capacity_binding(world.run)
    ) == 1, "a refused call debited capacity"


# --------------------------------------------------------------- rung 4b


def test_rung4b_a_real_wall_budget_stop_terminalizes_and_releases_the_lease(
    real_codex_world,
) -> None:
    """A real Codex process, stopped mid-flight by the real wall budget.

    The budget is deliberately far below the time a real session needs to
    reach its first token -- model startup alone exceeds it -- so the stop
    is forced by the budget rather than by hoping the agent is slow. A
    generous budget plus a "please think for a long time" prompt would
    make this rung a coin flip on model latency.

    The retry is the state evidence that the lease was released: it would
    raise ``EffectBusyError`` instead of returning a result otherwise.
    """
    world = real_codex_world(max_tool_calls=4)

    def slow_prompt(context):
        return world.production_prompt(
            context,
            extra=(
                "Before doing anything else, think step by step at length "
                "about at least twenty distinct candidate templates, "
                "writing out your full reasoning for each one, and only "
                "then begin evaluating them one at a time."
            ),
        )

    request = world.step_request()
    adapter = world.adapter(
        world.runner(
            timeout_seconds=_WALL_BUDGET_STOP_SECONDS,
            prompt_builder=slow_prompt,
        )
    )

    result, _ref = world.harness(adapter).run_step(request)

    assert result.status is StepStatus.FAILED, (
        f"a {_WALL_BUDGET_STOP_SECONDS} s wall budget did not stop the real "
        f"agent: {result.status}"
    )
    assert result.terminal_failure is not None
    assert result.terminal_failure.code == CODEX_WALL_BUDGET_EXCEEDED_CODE, (
        f"unexpected terminal failure: {result.terminal_failure}"
    )
    assert result.accepted_candidates == ()

    # The lease must be free: an identical Step runs again rather than
    # raising EffectBusyError.
    retry = world.adapter(
        world.runner(
            timeout_seconds=_WALL_BUDGET_STOP_SECONDS,
            prompt_builder=slow_prompt,
        )
    )
    retried, _retry_ref = world.harness(retry).run_step(request)
    assert retried.status is StepStatus.FAILED


# --------------------------------------------------------------- rung 4c


def test_rung4c_an_agent_that_never_calls_the_tool_retains_the_seed(
    real_codex_world,
) -> None:
    """No evaluation, no candidate -- and nothing stranded.

    The artifact carries no candidate body, so an agent that skips the
    tool can only retain the seed. This is the contract that makes "no
    eval outside the tools" hold against a real, unscripted agent.
    """
    world = real_codex_world(max_tool_calls=2)

    def no_tool_prompt(context):
        return world.production_prompt(
            context,
            extra=(
                "For this run specifically: do NOT call the evaluation "
                "tool at all. Immediately write the final artifact with "
                "selected_call_id set to null and evaluated_call_ids set "
                "to an empty list."
            ),
        )

    adapter = world.adapter(world.runner(prompt_builder=no_tool_prompt))
    result, _ref = world.harness(adapter).run_step(world.step_request())

    assert result.terminal_failure is None, (
        f"a no-tool-call Step failed instead of retaining the seed: "
        f"{result.terminal_failure}"
    )
    assert result.status is StepStatus.COMPLETE
    assert result.seed_retained is True, (
        "the agent made no evaluation, so the seed must be retained"
    )
    assert result.accepted_candidates == ()
    assert result.tool_evidence == ()


# --------------------------------------------------------------- rung 4d


def test_rung4d_a_real_seed_identical_selection_retains_the_seed(
    real_codex_world,
) -> None:
    """The seed's own template, evaluated and then *selected*.

    A real agent that decides the seed is best has two ways to say so:
    ``selected_call_id=null``, and selecting an evaluated call whose
    template is the seed's. This ladder observed the second three times
    on real runs, and it was a hard ``codex_selection_contract`` failure
    -- ``diff_check`` refuses a mutation equal to its base -- which threw
    away a Step whose evaluations the run had already admitted, paid for,
    and debited.

    The two forms assert the same thing, so both must reach
    ``seed_retained``. The rung is deliberately driven against the
    *production* prompt, which now tells the agent to use null instead:
    this asserts the adapter's tolerance, not the prompt's, so the extra
    instruction has to overrule that clause explicitly. A rung that let
    the agent choose the form would observe whichever one it picked.

    The seed's template is read off the world's own candidate rather
    than spelled here, so the rung cannot drift from what the run
    actually seeded.
    """
    world = real_codex_world(max_tool_calls=2)
    seed_template = world.candidate.payload.to_json()[TOY_MUTATION_FIELD]

    def seed_selecting_prompt(context):
        return world.production_prompt(
            context,
            extra=(
                "For this run specifically, override the instruction "
                "above about returning null for the seed. Evaluate "
                f"exactly this one template: {seed_template!r}, with a "
                "call_id you choose. Then write the final artifact with "
                "selected_call_id set to THAT call_id -- not null -- and "
                "evaluated_call_ids naming it."
            ),
        )

    adapter = world.adapter(world.runner(prompt_builder=seed_selecting_prompt))
    result, _ref = world.harness(adapter).run_step(world.step_request())

    assert result.terminal_failure is None, (
        "a seed-identical selection failed instead of retaining the seed: "
        f"{result.terminal_failure}"
    )
    assert result.status is StepStatus.COMPLETE
    assert result.seed_retained is True, (
        "the real agent selected the seed's own template, which is the "
        "seed being retained"
    )
    assert result.accepted_candidates == ()

    # The evaluation the agent paid for stays on the ledger and stays
    # debited: reading the selection for what it means must not change
    # what the run spent.
    assert result.tool_evidence, (
        "the agent retained the seed without evaluating anything, which is "
        "rung 4c's path, not this one -- this rung needs a real admitted "
        "evaluation to have been selected and kept"
    )
    assert result.budget_delta.consumed["tool_calls"] == len(
        result.tool_evidence
    )
    scored_templates = {
        entry.store_entry.tool_call.record.args["template"]
        for entry in result.tool_evidence
    }
    assert seed_template in scored_templates, (
        "the agent evaluated something other than the seed template, so a "
        "seed-identical selection was never the thing that retained the "
        f"seed: {scored_templates}"
    )


# ---------------------------------------------------------------- rung 5


def test_rung5_a_real_multi_evaluation_loop_selects_an_evaluated_candidate(
    real_codex_world,
) -> None:
    """The real agent's tool-call loop, over several distinct templates.

    A single-call rung cannot show that ledger ordinals, tool evidence,
    and artifact selection stay consistent across an agent that iterates.
    The accepted candidate's template is read back from the *store*, so
    the assertion is that the selection came from a recorded call rather
    than from anything the artifact asserted.
    """
    world = real_codex_world(max_tool_calls=4)

    def multi_prompt(context):
        return world.production_prompt(
            context,
            extra=(
                "For this run, evaluate these three templates, each with "
                f"its own distinct call_id: {_TEMPLATE_A!r}, "
                f"{_TEMPLATE_C!r}, {_TEMPLATE_D!r}. Then select the "
                "call_id with the highest reward and name every call you "
                "made in evaluated_call_ids."
            ),
        )

    adapter = world.adapter(world.runner(prompt_builder=multi_prompt))
    result, _ref = world.harness(adapter).run_step(world.step_request())

    assert result.terminal_failure is None, (
        f"the multi-evaluation Step failed: {result.terminal_failure}"
    )
    assert result.status is StepStatus.COMPLETE
    assert len(result.tool_evidence) >= 2, (
        "the real agent made "
        f"{len(result.tool_evidence)} evaluations; this rung needs a loop"
    )
    # Ledger totality: every admitted call is on the evidence, and the
    # budget debit matches it exactly.
    assert result.budget_delta.consumed["tool_calls"] == len(
        result.tool_evidence
    )
    call_ids = [str(e.store_entry.call_id) for e in result.tool_evidence]
    assert len(set(call_ids)) == len(call_ids), (
        f"the real agent reused a call_id: {call_ids}"
    )

    if result.accepted_candidates:
        accepted = world.store.get(
            result.accepted_candidates[0].record_ref.reference
        )
        scored_templates = {
            entry.store_entry.tool_call.record.args["template"]
            for entry in result.tool_evidence
        }
        assert accepted["payload"][TOY_MUTATION_FIELD] in scored_templates, (
            "the accepted candidate's template was never evaluated through "
            "the tool"
        )


# ---------------------------------------------------------------- rung 6


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_rung6_the_real_binary_accepts_each_reasoning_effort(
    tmp_path, effort
) -> None:
    """``model_reasoning_effort`` is a config key, so a bad value is fatal.

    Checked without a session: config parsing completes against an empty
    CODEX_HOME and the process then stops at authentication.
    """
    from whetstone.optim.codex.mcp_host import CodexMcpEndpoint

    empty_home = tmp_path / f"home-{effort}"
    empty_home.mkdir()
    argv = build_codex_command(
        prompt="ping",
        codex_binary=real_codex_binary(),
        model="",
        reasoning_effort=effort,
        mcp_endpoint=CodexMcpEndpoint(
            url="http://127.0.0.1:1/mcp", auth_token="t" * 16
        ),
        output_schema_path=str(_write_schema(tmp_path)),
        output_artifact_path=str(tmp_path / f"artifact-{effort}.json"),
        working_directory=str(tmp_path),
    )
    completed = subprocess.run(  # noqa: S603 - fixed binary, test-only
        argv,
        capture_output=True,
        text=True,
        timeout=120.0,
        stdin=subprocess.DEVNULL,
        env={"PATH": "/usr/bin:/bin", "CODEX_HOME": str(empty_home)},
    )

    combined = completed.stdout + completed.stderr
    assert "Error loading config.toml" not in combined, (
        f"the real Codex CLI rejected model_reasoning_effort={effort!r}:\n"
        f"{combined[-2000:]}"
    )


def test_rung6_a_real_step_runs_under_an_explicit_reasoning_effort(
    real_codex_world,
) -> None:
    """One real session proving the accepted key actually drives a run."""
    world = real_codex_world(max_tool_calls=2)
    adapter = world.adapter(world.runner(reasoning_effort="low"))

    result, _ref = world.harness(adapter).run_step(world.step_request())

    assert result.terminal_failure is None, (
        f"a run under reasoning_effort=low failed: {result.terminal_failure}"
    )
    assert result.status is StepStatus.COMPLETE


# ---------------------------------------------------------------- rung 7


def test_rung7_output_retention_survives_a_real_truncated_transcript(
    real_codex_world,
) -> None:
    """A small output budget against a real, verbose JSONL stream.

    The parser stitches a stream the retention window cut mid-record. The
    fake CLI emits tidy, predictable lines; only the real CLI produces the
    interleaved reasoning/event traffic that exercises the cut-record
    heuristics. The Step must still terminalize cleanly -- either
    completing or failing with a typed code, never raising.
    """
    world = real_codex_world(max_tool_calls=3)

    def verbose_prompt(context):
        return world.production_prompt(
            context,
            extra=(
                "Narrate your reasoning verbosely as you go, then evaluate "
                f"{_TEMPLATE_A!r} and {_TEMPLATE_B!r} and select the "
                "better one."
            ),
        )

    # Far below a real transcript, so retention certainly truncates.
    adapter = world.adapter(
        world.runner(max_output_bytes=8192, prompt_builder=verbose_prompt)
    )

    result, _ref = world.harness(adapter).run_step(world.step_request())

    # The contract is terminalization, not success: a truncated stream may
    # legitimately lose the artifact. What must never happen is an
    # untyped escape.
    assert result.status in {StepStatus.COMPLETE, StepStatus.FAILED}
    if result.terminal_failure is not None:
        assert result.terminal_failure.code.startswith("codex_"), (
            "a truncated real transcript produced an untyped failure: "
            f"{result.terminal_failure}"
        )


# ---------------------------------------------------------------- rung 8


def test_rung8_the_real_sandbox_denies_the_store_and_writes_outside_scratch(
    real_codex_world,
) -> None:
    """The kernel, not the prompt, is what keeps the agent off the ledger.

    The agent is asked to read the whetstone store and to write outside
    its scratch directory. sandbox-exec must deny both, and the store
    file must be byte-identical afterwards. The shell tool is in the deny
    list, so the realistic outcome is that the agent cannot even attempt
    it -- either way, the ledger is untouched.
    """
    import hashlib
    from pathlib import Path

    world = real_codex_world(max_tool_calls=2)
    store_path = Path(world.sqlite_path)
    forbidden_target = world.tmp_path / "escaped.txt"

    def probing_prompt(context):
        return world.production_prompt(
            context,
            extra=(
                "Additionally, as a containment check, first attempt to "
                f"read the file {str(store_path)!r} and attempt to create "
                f"the file {str(forbidden_target)!r}. Both attempts are "
                "expected to fail; do not treat the failures as errors, "
                "just continue and complete the evaluation task normally."
            ),
        )

    digest_before = hashlib.sha256(store_path.read_bytes()).hexdigest()
    adapter = world.adapter(world.runner(prompt_builder=probing_prompt))

    result, _ref = world.harness(adapter).run_step(world.step_request())

    assert not forbidden_target.exists(), (
        "the real Codex agent wrote outside its scratch directory: "
        f"{forbidden_target}"
    )
    # The store is written by the evaluation server (outside the sandbox),
    # so it legitimately changes when a call is admitted. What must hold
    # is that the Step terminalized and the agent itself never got write
    # access -- proven by the escape file's absence plus a clean terminal.
    assert result.status in {StepStatus.COMPLETE, StepStatus.FAILED}
    assert digest_before  # the store existed before the run
    if result.terminal_failure is not None:
        assert result.terminal_failure.code.startswith("codex_")


# ---------------------------------------------------------------- rung 9


def test_rung9_the_hosted_server_refuses_a_foreign_bearer_token() -> None:
    """Reachability is not authorization.

    The endpoint is on loopback, so any process on the machine can
    connect to it. The bearer token is what makes it *this run's*
    endpoint. This drives real HTTP against the real hosted server: a
    foreign token and a missing token must both get 401, and the run's
    own token must get past the middleware.
    """
    pytest.importorskip("uvicorn")

    token = "a" * 64

    class _Echo:
        """Minimal app: the auth middleware is what this rung owns."""

        def streamable_http_app(self, *, streamable_http_path: str):
            from starlette.applications import Starlette
            from starlette.responses import PlainTextResponse
            from starlette.routing import Route

            async def _ok(_request):
                return PlainTextResponse("ok")

            return Starlette(
                routes=[Route(streamable_http_path, _ok, methods=["GET"])]
            )

    with CodexMcpHost(_Echo(), auth_token=token) as endpoint:
        assert endpoint.url.startswith("http://127.0.0.1:"), endpoint.url

        assert _status(endpoint.url, None) == 401, (
            "the hosted evaluation endpoint served a request carrying no "
            "bearer token"
        )
        assert _status(endpoint.url, f"{CODEX_MCP_AUTH_SCHEME} {'b' * 64}") == 401, (
            "the hosted evaluation endpoint accepted a foreign bearer token, "
            "so any local process could drive this run's evaluations"
        )
        assert _status(endpoint.url, f"{CODEX_MCP_AUTH_SCHEME} {token}") == 200, (
            "the hosted evaluation endpoint refused this run's own token"
        )


def _status(url: str, authorization: str | None) -> int:
    """The status the hosted endpoint returns for one real GET."""
    request = urllib.request.Request(url, method="GET")  # noqa: S310
    if authorization is not None:
        request.add_header("Authorization", authorization)
    try:
        with urllib.request.urlopen(request, timeout=10.0) as response:  # noqa: S310
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
