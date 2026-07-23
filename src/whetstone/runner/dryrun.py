"""Fake-transport dry-run harness for the validation ``cell`` CLI path.

This is the *program* seam the ``--dry-run-fake`` CLI flag drives so the whole
cell plumbing -- ``build_env_experiment`` -> baseline/ceiling/best official
evals -> ``run_optimize`` internal-split search -> delta + bootstrap CI ->
``cells.jsonl`` / ``spend.jsonl`` ledger append -- can be exercised end-to-end
as a program (not only under pytest), while making **no live paid LLM call**.

It builds the same scripted doubles the runner tests use, but in production
source so the CLI can import them without reaching into ``tests/``:

* a scripted *rollout* transport (a ``TransportCall``) whose reply is a pure
  function of the rendered prompt -- no network, no DBOS;
* the shared :class:`whetstone.optimization.proposer.FakeProposerTransport`
  scripted proposer;
* a scripted credits fetcher (a static OpenRouter credits snapshot pair) so the
  ``lane=openrouter`` before/after spend accounting runs with no HTTP.

Two scripts mirror the two proven cell paths:

* ``eval`` (the identity optimizer, breadth/depth = 0x0): the naive probe
  itself renders the gold answer -> baseline == best, a faithful eval cell.
* ``copro`` (and any proposing optimizer): only the injected *winning* template
  renders the gold answer; the naive baseline scores 0 -> the optimizer's best
  candidate beats the baseline (``status=improved``, ``delta=1.0``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from dr_providers import (
    ProviderCallRequest,
    ProviderInvocationEvidence,
    ProviderTransportPolicy,
    ProviderTransportResponse,
    RawHttpRequest,
    policy_for,
)
from whetstone_envs.core import Instance

from whetstone.envs.ed1 import ED1_ENV_NAME
from whetstone.envs.factory import EnvExperiment, build_env_experiment
from whetstone.envs.registry import env_spec
from whetstone.envs.rollout_definition import (
    ceiling_candidate,
    initial_candidate,
    render_prompt,
)
from whetstone.envs.sampling import SamplingOverrides
from whetstone.optimization.codex_proposer import codex_proposer_ref
from whetstone.optimization.mutation import MUTATION_FIELD
from whetstone.optimization.proposer import (
    FakeProposerTransport,
    ProposerConfig,
)
from whetstone.optimization.schema import Candidate
from whetstone.provider.policy import ProviderExecutionPolicy
from whetstone.runner.budget import CreditsSnapshot
from whetstone.runner.cell import CellConfig, CellOutcome, run_cell
from whetstone.runner.execution_mode import ExecutionMode
from whetstone.runner.ledger import Ledger

__all__ = [
    "DRYRUN_PROPOSER_MODEL",
    "DRYRUN_TASK_MODEL",
    "DRYRUN_WINNING_SUFFIX",
    "ScriptedRolloutTransport",
    "run_dry_cell",
]

#: The stand-in task/proposer model slugs recorded on a dry-run cell record.
#: They mirror the canonical route slugs so the ledger line looks like a real
#: cell, but every transport is a scripted fake (no live paid call).
DRYRUN_TASK_MODEL = "openai/gpt-5-nano"
DRYRUN_PROPOSER_MODEL = "openai/gpt-5.4-nano"
#: The ed1 dry-run enc/dec task model (recorded on the dry cell line).
ED1_DRY_TASK_MODEL = "deepseek/deepseek-v4-flash"

#: A brace-free suffix appended to the env's own naive template to build the
#: winning template. Appending plain text keeps every ``str.format``
#: placeholder intact (so the template renders for EVERY env) while producing
#: a rendered prompt distinct from the naive probe -- so only the winning
#: candidate maps to the gold answer.
DRYRUN_WINNING_SUFFIX = "\n(Answer precisely.)"


def _winning_template(experiment: EnvExperiment) -> str:
    """The winning template: the env's naive template + a brace-free suffix.

    Deriving from the naive template guarantees it renders for every env
    (all placeholders preserved) yet renders a distinct prompt.
    """
    naive = initial_candidate(env_spec(experiment.env_name))
    return str(naive.payload[MUTATION_FIELD]) + DRYRUN_WINNING_SUFFIX

_MISS = "definitely-not-a-label"


def _dryrun_transport_policy() -> ProviderTransportPolicy:
    return policy_for(
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://example.test/v1",
        native_retry_count=0,
    )


@dataclass
class ScriptedRolloutTransport:
    """A scripted rollout transport: rendered prompt -> reply text.

    ``reply`` is a pure function of the request's user-message content (the
    rendered prompt), so the fake answers the gold for the target templates
    and a fixed miss otherwise. Records every request served. No network.
    """

    reply: Callable[[str], str]
    policy: ProviderTransportPolicy = field(
        default_factory=_dryrun_transport_policy
    )
    served: list[ProviderCallRequest] = field(default_factory=list)

    def __call__(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        self.served.append(request)
        messages = request.transcript.messages
        prompt = messages[-1].content if messages else ""
        text = self.reply(prompt)
        raw_request = RawHttpRequest.build(
            url="https://example.test/v1/chat/completions",
            headers={"Authorization": "Bearer k", "content-type": "json"},
            body={"model": DRYRUN_TASK_MODEL},
        )
        response = ProviderTransportResponse(
            text=text,
            raw_body={"choices": [{"message": {"content": text}}]},
            response_id="dryrun-resp-1",
            model=DRYRUN_TASK_MODEL,
            finish_reason="stop",
        )
        return ProviderInvocationEvidence.build(
            request=request,
            policy=self.policy,
            raw_request=raw_request,
            outcome=response,
        )


def _all_instances(experiment: EnvExperiment) -> tuple[Instance, ...]:
    cfgs = experiment.eval_configs
    return tuple(cfgs.internal.instances) + tuple(cfgs.official.instances)


def _correct_reply(experiment: EnvExperiment) -> Callable[[str], str]:
    """Naive AND ceiling probes render the gold answer (the eval-cell path)."""
    env = env_spec(experiment.env_name)
    naive = initial_candidate(env)
    ceiling = ceiling_candidate(env)
    by_prompt: dict[str, str] = {}
    for inst in _all_instances(experiment):
        by_prompt[render_prompt(env, naive, inst)] = inst.gold
        by_prompt[render_prompt(env, ceiling, inst)] = inst.gold
    return lambda prompt: by_prompt.get(prompt, _MISS)


def _improvement_reply(
    experiment: EnvExperiment, winning_template: str
) -> Callable[[str], str]:
    """Only the winning template renders the gold answer (the improved path).

    The naive/ceiling probes and every other proposal score 0; the candidate
    built from ``winning_template`` scores the gold -> the optimizer's best
    candidate beats the naive baseline on the official split.
    """
    env = env_spec(experiment.env_name)
    winner = Candidate(
        candidate_id="dryrun-winner",
        base_ref=initial_candidate(env).base_ref,
        payload={MUTATION_FIELD: winning_template},
    )
    winning_prompts: dict[str, str] = {}
    for inst in _all_instances(experiment):
        winning_prompts[render_prompt(env, winner, inst)] = inst.gold
    return lambda prompt: winning_prompts.get(prompt, _MISS)


def _dryrun_proposer_config() -> ProposerConfig:
    return ProposerConfig(
        provider_call_config_ref=f"pcc://{DRYRUN_PROPOSER_MODEL}",
        provider_call_config_hash="f" * 64,
        temperature=1.0,
    )


def _pool_n_per_stratum(env: str) -> int:
    """The smallest per-stratum pool size that fits a tiny (2,2,2) split.

    The dry run only needs the plumbing to execute over a real (small) pool;
    it does not need the canonical pool sizes.
    """
    n = 1
    while n <= 40:
        try:
            build_env_experiment(
                env, model=DRYRUN_TASK_MODEL, pool_n_per_stratum=n,
                split_sizes=(2, 2, 2),
            )
        except Exception:  # probing for a fitting pool size
            n += 1
            continue
        return n
    msg = f"could not size a tiny pool for {env!r}"
    raise RuntimeError(msg)


def run_dry_cell(
    *,
    env: str,
    optimizer: str,
    root,
    attempt: int = 0,
    lane: str = "openrouter",
    execution_mode: ExecutionMode = ExecutionMode.IN_PROCESS,
    overrides: SamplingOverrides | None = None,
    budget_ratio: float = 0.5,
) -> CellOutcome:
    """Run one fake-transport cell end-to-end and append its ledger line.

    Mirrors the two proven cell paths: ``eval`` drives the correct-reply script
    (naive == best), any proposing optimizer drives the improvement script (the
    injected winning template beats the naive baseline). Returns the
    :class:`CellOutcome`; the ledger line lands under ``root``.

    ``overrides`` (a :class:`SamplingOverrides`) exercises the reduced-sampling
    path end-to-end (``--official-n`` / ``--official-repeats``): it folds into
    the official Eval Config identity so the dry cell records the overrides and
    keys the env cache by the reduced hash.
    """
    overrides = overrides or SamplingOverrides()
    # ed1 (enc-dec) has a distinct fake-cell path: the 3-node
    # encoder->decoder->
    # code-eval graph + a local (no-Docker) sandbox scorer. The QA path below
    # is
    # unchanged.
    if env == ED1_ENV_NAME:
        return _run_dry_ed1_cell(
            optimizer=optimizer, root=root, attempt=attempt, lane=lane,
            execution_mode=execution_mode, budget_ratio=budget_ratio,
        )
    pool_n = _pool_n_per_stratum(env)
    experiment = build_env_experiment(
        env, model=DRYRUN_TASK_MODEL, pool_n_per_stratum=pool_n,
        split_sizes=(2, 2, 2), overrides=overrides,
    )
    if optimizer == "eval":
        reply = _correct_reply(experiment)
        proposer = FakeProposerTransport(script={}, default=())
    else:
        winning = _winning_template(experiment)
        reply = _improvement_reply(experiment, winning)
        proposer = FakeProposerTransport(script={}, default=(winning,))
    rollout = ScriptedRolloutTransport(reply=reply)
    config = CellConfig(
        optimizer=optimizer,
        env=env,
        lane=lane,
        attempt=attempt,
        task_model=DRYRUN_TASK_MODEL,
        proposer_model=(
            # Match the live path's recorded codex proposer id (task 5): the
            # codex optimizer drafts through the local codex CLI.
            codex_proposer_ref("gpt-5.6") if optimizer == "codex"
            else DRYRUN_PROPOSER_MODEL
        ),
        canonical=False,
        proposer_config=_dryrun_proposer_config(),
        proposer_transport=proposer,
        rollout_transport=rollout,
        execution_policy=ProviderExecutionPolicy(
            transport_policy=_dryrun_transport_policy(),
            max_attempts=1,
        ),
        repeats=3,
        official_repeats=5,
        pool_n_per_stratum=pool_n,
        split_sizes=(2, 2, 2),
        execution_mode=execution_mode,
        window_notes="dry-run-fake (no live paid call)",
        sampling_overrides=overrides,
    )
    ledger = Ledger(root=root)
    ledger.load()
    # A static credits pair: identical before/after so the reserve guard clears
    # (a non-canonical dry cell) and the recorded spend is $0 (no real spend).
    return run_cell(
        config,
        ledger=ledger,
        credits_fetcher=_static_credits(),
    )


def _static_credits() -> Callable[[], CreditsSnapshot]:
    def fetch() -> CreditsSnapshot:
        return CreditsSnapshot(total_credits=710.0, total_usage=616.0)

    return fetch


#: The fixed ed1 dry-run task slice (small + offline).
_ED1_DRY_TASK_LIMIT = 4


def _ed1_dry_reply(tasks) -> Callable[[str], str]:
    """A scripted enc/dec reply: encoder emits a per-task marker; the decoder
    returns that task's canonical solution (so the sandbox scores PASS)."""
    by_entry = {
        t.humaneval_task.entry_point: t.humaneval_task.ground_truth_code
        for t in tasks
    }

    def reply(prompt: str) -> str:
        if prompt.startswith("Provide") or prompt.startswith("Compress"):
            for ep in by_entry:
                if f"def {ep}(" in prompt:
                    return f"REBUILD:{ep}"
            return "REBUILD:unknown"
        for ep, gt in by_entry.items():
            if f"REBUILD:{ep}" in prompt:
                return gt
        return "def _x():\n    return None\n"

    return reply


def _run_dry_ed1_cell(
    *,
    optimizer: str,
    root,
    attempt: int,
    lane: str,
    execution_mode: ExecutionMode,
    budget_ratio: float = 0.5,
) -> CellOutcome:
    """Run one ed1 (enc-dec) fake cell end-to-end (no network, no Docker).

    Builds the 3-node encoder->decoder->code-eval experiment over a small
    offline HumanEval+ slice, a scripted enc/dec transport (encoder marker ->
    decoder canonical), a LOCAL subprocess sandbox scorer (no container), and
    any
    optimizer's proposer script (a valid improved encoder template for the
    proposing optimizers). Records BOTH scores (pass rate + Mean Compression
    Ratio) on the ledger line.
    """
    from whetstone.envs.ed1 import ENCODER_TEMPLATE_B, load_ed1_tasks

    tasks = load_ed1_tasks(prefer_snapshot=True, limit=_ED1_DRY_TASK_LIMIT)
    reply = _ed1_dry_reply(tasks)
    # The default ed1 scorer runs the test suite in a LOCAL subprocess (no
    # container) -- no Docker needed for the dry-run wiring validation.
    if optimizer == "eval":
        proposer = FakeProposerTransport(script={}, default=())
    else:
        # A valid alternate encoder template (still a real Mutation-Surface
        # template) so a proposing optimizer drafts + scores a candidate.
        proposer = FakeProposerTransport(
            script={}, default=(ENCODER_TEMPLATE_B,)
        )
    config = CellConfig(
        optimizer=optimizer,
        env=ED1_ENV_NAME,
        lane=lane,
        attempt=attempt,
        task_model=ED1_DRY_TASK_MODEL,
        proposer_model=(
            codex_proposer_ref("gpt-5.6") if optimizer == "codex"
            else DRYRUN_PROPOSER_MODEL
        ),
        canonical=False,
        proposer_config=_dryrun_proposer_config(),
        proposer_transport=proposer,
        rollout_transport=ScriptedRolloutTransport(reply=reply),
        execution_policy=ProviderExecutionPolicy(
            transport_policy=_dryrun_transport_policy(),
            max_attempts=1,
        ),
        repeats=1,
        official_repeats=1,
        execution_mode=execution_mode,
        window_notes="dry-run-fake (no live paid call)",
        budget_ratio=budget_ratio,
        ed1_task_limit=_ED1_DRY_TASK_LIMIT,
    )
    ledger = Ledger(root=root)
    ledger.load()
    return run_cell(config, ledger=ledger, credits_fetcher=_static_credits())
