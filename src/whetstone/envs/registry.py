"""The five whetstone-envs task families bound to a uniform env spec.

Each quick-test candidate in ``whetstone-envs`` exposes the same three
model-call-free surfaces behind its ``whetstone_envs.<env>`` package: a
seeded generator (``generate.generate_pool`` -> a
``whetstone_envs.core.TaskPool``), an independent scoring oracle
(``oracle.score_gold(prediction, gold) -> int``, which applies the env's
*shared* normalization first), and a naive/ceiling ``ProbePair``
(``prompts.PROBES``). This module names those surfaces once per env so the
rest of the adapter is env-agnostic.

The oracle contract is deliberately uniform: every env's ``score_gold``
takes the model generation as its first argument and the instance's public
``gold`` field as its second, and returns ``0`` / ``1`` after the shared
:func:`whetstone_envs.core.normalize`. For the four re-derive-the-answer
envs (c11, c19, c18, c23) ``gold`` is the expected answer string; for c22
``gold`` is the serialized constraint stack the oracle re-runs its checkers
against. Either way, ``score_gold(generation, task.gold)`` is the single
call the whetstone metric-extraction operator makes.

Split sizes and the repeat count are the env's committed spec defaults where
the env commits them (``generate.default_split_sizes`` for c11/c18/c19/c23),
and whetstone-side spec defaults where the env does not (c22 commits no
split call; see :data:`_C22_SPLIT`). ``held_out`` is never referenced by any
Sampling Config this adapter builds.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType

from whetstone_envs.core import ProbePair, TaskPool

from whetstone.envs.probes import ProbeSurface, probe_surface

#: The five task families this adapter binds, in build/validation order
#: (cheapest/most robust first, matching the validation plan's spend order).
ENV_NAMES: tuple[str, ...] = ("c22", "c11", "c19", "c18", "c23")


@dataclass(frozen=True, slots=True)
class TokenEstimate:
    """The committed per-call token estimate for one env's baseline probes.

    Sourced verbatim from each env's baseline-spec §5 ("Per-instance token
    estimate") total-tokens/call row: the naive probe and the ceiling probe
    are distinguished by every spec, so both are recorded separately. These
    are the committed defaults the pilot's token-sanity check runs against
    when ``--spec-estimate-tokens`` is not passed; the flag still overrides.
    """

    naive: int
    ceiling: int


#: Committed per-env token estimates from each baseline-spec §5 total row.
#: c11: naive ~350 / ceiling ~800 (blended totals); c19: naive ~280 /
#: ceiling ~1,150; c18: naive ~253 / ceiling ~650; c22: naive ~170 /
#: ceiling ~420; c23: naive ~140 (120 in + 20 out) / ceiling ~420 (400 + 20).
_ENV_TOKEN_ESTIMATES: dict[str, TokenEstimate] = {
    "c11": TokenEstimate(naive=350, ceiling=800),
    "c19": TokenEstimate(naive=280, ceiling=1150),
    "c18": TokenEstimate(naive=253, ceiling=650),
    "c22": TokenEstimate(naive=170, ceiling=420),
    "c23": TokenEstimate(naive=140, ceiling=420),
}

#: Spec-default deliberate-observation repeats per task. The envs commit no
#: repeat count (a Repeat Plan is a whetstone execution concern, out of scope
#: for whetstone-envs per its PLAN "Integration handoff"); the validation
#: plan's pilots use 3 temp-0 repeats, so 3 is the whetstone-side default.
DEFAULT_REPEATS = 3

#: Envs whose oracle ``score_gold`` is ``(gold, response)`` rather than the
#: usual ``(prediction, gold)``. Only c22 (its oracle re-runs the constraint
#: checkers against the response, reading the constraint stack from the gold).
_GOLD_FIRST_ENVS: frozenset[str] = frozenset({"c22"})

#: Envs whose generated pool is **blocked** by stratum (all of one stratum's
#: instances contiguous, then the next), so ``TaskPool.split``'s contiguous
#: slicing would skew each split toward the leading strata. c22 is blocked
#: (its generator emits ``n_per_stratum`` instances per (constraint-count x
#: atom-mix) cell in a fixed cell order). The other four envs interleave their
#: strata (verified), so a contiguous split is already balanced for them.
#: Blocked-pool envs are sampled per-stratum instead (see
#: :func:`whetstone.envs.sampling.stratified_split`).
_STRATIFIED_SPLIT_ENVS: frozenset[str] = frozenset({"c22"})

#: c22 commits no ``default_split_sizes`` (unlike the other four). Its spec
#: (Section 1) proposes N=20/stratum; this adapter uses the same
#: small-internal / balanced-official-and-held-out per-stratum shape the
#: other envs commit, kept well inside c22's 20/stratum pool. Recorded as a
#: judgment call in the build report.
_C22_SPLIT_PER_STRATUM = (2, 6, 6)  # (internal_eval, official, held_out)


@dataclass(frozen=True, slots=True)
class EnvSpec:
    """The bound surfaces of one whetstone-envs task family.

    Parameters
    ----------
    name:
        The env identifier (``"c22"`` .. ``"c23"``).
    generate:
        The env's ``generate`` module (``generate_pool`` / ``build_manifest``
        / optional ``default_split_sizes``).
    oracle:
        The env's ``oracle`` module. Its ``score_gold`` returns 0/1 after the
        env's shared normalization; see ``gold_first`` for the argument order.
    probes:
        The env's naive/ceiling :class:`~whetstone_envs.core.ProbePair` as
        committed by whetstone-envs. Retained for reference / byte-fidelity
        equivalence; the adapter renders through :attr:`surface` instead.
    surface:
        The adapter-side :class:`~whetstone.envs.probes.ProbeSurface`: a
        genuinely mutable, serialization-stable ``user_prompt_template`` pair
        plus a content-driven render. For four envs this wraps ``probes``
        verbatim; for c19 it replaces the env's identity-sentinel ``ProbePair``
        (which is non-functional under mutation / JSON round-trip) with real
        ``str.format`` templates, so the Mutation Surface is genuinely mutable
        and Result-Store-stable.
    oracle_qualname:
        The dotted path to the oracle entry point, folded into the Metric
        Extraction Config identity so a change of oracle wiring is visible in
        ``eval_config_hash`` / ``graph_hash``.
    token_estimate:
        The committed per-call :class:`TokenEstimate` (naive + ceiling) from
        this env's baseline-spec §5, the default the pilot's token-sanity
        check runs against absent a ``--spec-estimate-tokens`` override.
    gold_first:
        Whether the env oracle's ``score_gold`` takes ``(gold, response)``
        (``True`` -- c22) rather than the usual ``(prediction, gold)``
        (``False`` -- c11/c19/c18/c23). c22's oracle re-runs its constraint
        checkers against the response and reads the constraint stack from the
        gold, so its signature is ``score_gold(gold, response)``; this flag
        makes :meth:`score_gold` call the oracle with the right order while
        the adapter surface stays uniform ``score_gold(generation, gold)``.
    stratified_split:
        Whether this env's pool is **blocked** by stratum (``True`` only for
        c22), so the adapter samples splits per-stratum
        (:func:`whetstone.envs.sampling.stratified_split`) rather than via
        ``TaskPool.split``'s contiguous slicing (which would skew a blocked
        pool toward the leading strata). The interleaved envs keep the
        contiguous split, which is already balanced for them.
    """

    name: str
    generate: ModuleType
    oracle: ModuleType
    probes: ProbePair
    surface: ProbeSurface
    oracle_qualname: str
    token_estimate: TokenEstimate
    gold_first: bool = False
    stratified_split: bool = False

    def generate_pool(self, *, n_per_stratum: int | None = None) -> TaskPool:
        """Generate the env pool at its spec-default (or given) size."""
        if n_per_stratum is None:
            return self.generate.generate_pool()
        return self.generate.generate_pool(n_per_stratum=n_per_stratum)

    def default_split_sizes(self, pool: TaskPool) -> tuple[int, int, int]:
        """Return ``(internal_eval_n, official_n, held_out_n)`` for ``pool``.

        Delegates to the env's committed ``default_split_sizes`` when present
        (c11/c18/c19/c23). For c22 -- which commits no split call -- the
        whetstone-side per-stratum default (:data:`_C22_SPLIT_PER_STRATUM`) is
        scaled by the pool's stratum count, matching the interleaved-layout
        convention the other envs rely on.
        """
        split_fn = getattr(self.generate, "default_split_sizes", None)
        if split_fn is not None:
            return split_fn(pool)
        n_strata = len(pool.strata)
        internal, official, held_out = _C22_SPLIT_PER_STRATUM
        return (
            internal * n_strata,
            official * n_strata,
            held_out * n_strata,
        )

    def score_gold(self, generation: str, gold: str) -> int:
        """Invoke the env oracle on a generation + the instance gold.

        The single oracle call the whetstone metric-extraction operator
        makes. The env's shared normalization is applied inside ``score_gold``
        (never here), so scoring differences come from the model, not from
        per-adapter string handling.

        The adapter surface is uniform -- ``score_gold(generation, gold)`` --
        but the underlying env oracle's argument order differs: c22's
        ``score_gold(gold, response)`` (``gold_first``) versus the usual
        ``score_gold(prediction, gold)``. This method routes the arguments
        accordingly so a caller never has to know the per-env order.
        """
        if self.gold_first:
            return int(self.oracle.score_gold(gold, generation))
        return int(self.oracle.score_gold(generation, gold))


def _load_env_spec(name: str) -> EnvSpec:
    generate = import_module(f"whetstone_envs.{name}.generate")
    oracle = import_module(f"whetstone_envs.{name}.oracle")
    prompts = import_module(f"whetstone_envs.{name}.prompts")
    return EnvSpec(
        name=name,
        generate=generate,
        oracle=oracle,
        probes=prompts.PROBES,
        surface=probe_surface(name, prompts.PROBES),
        oracle_qualname=f"whetstone_envs.{name}.oracle.score_gold",
        token_estimate=_ENV_TOKEN_ESTIMATES[name],
        gold_first=name in _GOLD_FIRST_ENVS,
        stratified_split=name in _STRATIFIED_SPLIT_ENVS,
    )


class UnknownEnvError(KeyError):
    """A requested env name is not one of the five bound task families."""


def env_spec(name: str) -> EnvSpec:
    """Return the :class:`EnvSpec` for ``name`` (raises on an unknown env)."""
    if name not in ENV_NAMES:
        raise UnknownEnvError(
            f"unknown env {name!r}; expected one of {ENV_NAMES}"
        )
    return _load_env_spec(name)


__all__ = [
    "DEFAULT_REPEATS",
    "ENV_NAMES",
    "EnvSpec",
    "TokenEstimate",
    "UnknownEnvError",
    "env_spec",
]
