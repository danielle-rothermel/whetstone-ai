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
from typing import Protocol

from whetstone_envs.core import ProbePair, TaskPool

from whetstone.envs.probes import ProbeSurface, probe_surface


class PoolPreset(Protocol):
    """The generation-preset interface a variant env's pool is built from.

    Matches ``whetstone_envs.<env>.generate.Preset``: a named config bundling
    the generation axes + a disjoint seed range, with a ``generate`` that
    yields a :class:`~whetstone_envs.core.TaskPool`. Kept structural so the
    registry does not import any single env's concrete ``Preset`` type.
    """

    name: str

    def generate(self, *, n_per_stratum: int | None = None) -> TaskPool: ...

#: The task families this adapter binds, in build/validation order
#: (cheapest/most robust first, matching the validation plan's spend order).
#: ``c22h`` is the c22 hard-mode variant: it maps to the SAME c22 modules
#: (:data:`_ENV_MODULE_NAME`) but generates its pool from that env's
#: ``HARD_PRESET`` (:data:`_ENV_POOL_PRESET`) -- the hardest configuration of
#: the original IFEval suite, with no hidden-information design change. A
#: distinct env id keeps its ledger rows / cells / eval-config identities
#: separate from base c22's.
ENV_NAMES: tuple[str, ...] = ("c22", "c22h", "c11", "c19", "c18", "c23")

#: The underlying ``whetstone_envs`` package a bound env id loads its
#: generate/oracle/prompts surfaces from. Defaults to the env id itself; only
#: ``c22h`` differs (it reuses the c22 modules with a different pool preset).
_ENV_MODULE_NAME: dict[str, str] = {"c22h": "c22"}

#: The named generation preset a bound env id generates its pool from, when it
#: is a *variant* of another env's default pool. The value is the attribute
#: name of a ``Preset`` on the env's generate module. Only ``c22h`` uses one
#: (``HARD_PRESET``); every other env generates its committed default pool.
_ENV_POOL_PRESET: dict[str, str] = {"c22h": "HARD_PRESET"}


#: The two provenance markers a :class:`TokenEstimate` may carry. A
#: ``live-measured`` estimate came from an actual smoke/pilot measurement; a
#: ``scaled-pending-measurement`` estimate is a committed baseline-spec §5
#: value scaled by the reasoning-model correction, to be overwritten once its
#: env's pilot records a measured mean. As of the round-3 update ALL five envs
#: are ``live-measured``; ``scaled-pending-measurement`` is retained as the
#: default provenance for any newly added, not-yet-measured env.
ESTIMATE_LIVE_MEASURED = "live-measured"
ESTIMATE_SCALED_PENDING = "scaled-pending-measurement"
#: A third provenance: an estimate INHERITED verbatim from another env's
#: measured means (c22h seeds from c22's live-measured naive/ceiling), pending
#: its own pilot measurement which will overwrite it.
ESTIMATE_INHERITED_PENDING = "inherited-pending-measurement"


@dataclass(frozen=True, slots=True)
class TokenEstimate:
    """The per-call token estimate for one env's baseline probes.

    The naive probe and the ceiling probe are distinguished by every spec, so
    both are recorded separately. These are the defaults the pilot's token-
    sanity check runs against when ``--spec-estimate-tokens`` is not passed;
    the flag still overrides.

    ``estimate_source`` marks provenance: ``live-measured`` (all five envs, as
    of the round-3 update, from their pilots' measured per-call means) versus
    ``scaled-pending-measurement`` (a committed baseline-spec §5 value scaled
    for the reasoning-model correction, the default for any not-yet-measured
    env). A pilot overwrites its env's estimate with its measured means and
    records those in the pilot JSON so the report can show est-vs-actual.
    """

    naive: int
    ceiling: int
    estimate_source: str = ESTIMATE_SCALED_PENDING


#: Per-env token estimates. ALL FIVE are now LIVE-MEASURED from their pilots'
#: measured per-call means: c22 naive 2526 / ceiling 3046; c11 1735 / 1831;
#: c19 4377 / 5009; c23 5468 / 4953; c18 naive 1306. c18's ceiling keeps its
#: measured 2448 with the caveat that that measurement PREDATES the verdict-
#: extraction scoring fix (the extraction changes only how the reply is
#: scored, not how many tokens the model emits, so the token mean is expected
#: to hold; flagged for re-confirmation by the post-fix c18 pilot).
_ENV_TOKEN_ESTIMATES: dict[str, TokenEstimate] = {
    "c22": TokenEstimate(
        naive=2526, ceiling=3046, estimate_source=ESTIMATE_LIVE_MEASURED
    ),
    "c22h": TokenEstimate(
        # Inherited from c22's live-measured means; overwritten by c22h's
        # own pilot once it records a measured per-call mean.
        naive=2526,
        ceiling=3046,
        estimate_source=ESTIMATE_INHERITED_PENDING,
    ),
    "c11": TokenEstimate(
        naive=1735, ceiling=1831, estimate_source=ESTIMATE_LIVE_MEASURED
    ),
    "c19": TokenEstimate(
        naive=4377, ceiling=5009, estimate_source=ESTIMATE_LIVE_MEASURED
    ),
    "c18": TokenEstimate(
        # naive live-measured; ceiling 2448 measured PRE extraction fix.
        naive=1306, ceiling=2448, estimate_source=ESTIMATE_LIVE_MEASURED
    ),
    "c23": TokenEstimate(
        naive=5468, ceiling=4953, estimate_source=ESTIMATE_LIVE_MEASURED
    ),
}

#: Spec-default deliberate-observation repeats per task. The envs commit no
#: repeat count (a Repeat Plan is a whetstone execution concern, out of scope
#: for whetstone-envs per its PLAN "Integration handoff"); the validation
#: plan's pilots use 3 temp-0 repeats, so 3 is the whetstone-side default.
DEFAULT_REPEATS = 3

#: Envs whose oracle ``score_gold`` is ``(gold, response)`` rather than the
#: usual ``(prediction, gold)``. c22 and its hard variant c22h (same c22
#: oracle: it re-runs the constraint checkers against the response, reading the
#: constraint stack from the gold).
_GOLD_FIRST_ENVS: frozenset[str] = frozenset({"c22", "c22h"})

#: Envs whose generated pool is **blocked** by stratum (all of one stratum's
#: instances contiguous, then the next), so ``TaskPool.split``'s contiguous
#: slicing would skew each split toward the leading strata. c22 is blocked
#: (its generator emits ``n_per_stratum`` instances per (constraint-count x
#: atom-mix) cell in a fixed cell order). The other four envs interleave their
#: strata (verified), so a contiguous split is already balanced for them.
#: Blocked-pool envs are sampled per-stratum instead (see
#: :func:`whetstone.envs.sampling.stratified_split`). c22h shares c22's blocked
#: layout (its HARD_PRESET emits n_per_stratum instances per stratum in cell
#: order), so it is stratified too.
_STRATIFIED_SPLIT_ENVS: frozenset[str] = frozenset({"c22", "c22h"})

#: c22 commits no ``default_split_sizes`` (unlike the other four). Its spec
#: (Section 1) proposes N=20/stratum; this adapter uses the same
#: small-internal / balanced-official-and-held-out per-stratum shape the
#: other envs commit, kept well inside c22's 20/stratum pool. Recorded as a
#: judgment call in the build report.
_C22_SPLIT_PER_STRATUM = (2, 6, 6)  # (internal_eval, official, held_out)

#: Per-env per-stratum split overrides (``(internal, official, held_out)``).
#: c22h keeps base c22's 2:6 internal:official proportion but takes the full
#: remaining stratum depth as held_out, so its 3 x 20 pool splits to totals
#: (internal 6, official 18, held_out 36) with no unused instances. Any env
#: not listed falls back to :data:`_C22_SPLIT_PER_STRATUM`.
_SPLIT_PER_STRATUM_BY_ENV: dict[str, tuple[int, int, int]] = {
    "c22h": (2, 6, 12),
}


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
    #: When set, the named ``Preset`` on :attr:`generate` this env generates
    #: its pool from (e.g. c22h -> the c22 module's ``HARD_PRESET``), instead
    #: of the module's default ``generate_pool``. ``None`` -> default pool.
    pool_preset: PoolPreset | None = None
    #: The dataset revision folded into this env's Task Set identity. Defaults
    #: to the generate module's ``GENERATOR_VERSION``; a preset env overrides
    #: it with the preset's own version so a variant's Task Sets are a DISTINCT
    #: identity from the base env's (even though both load the same module).
    generator_version: str = ""

    def generate_pool(self, *, n_per_stratum: int | None = None) -> TaskPool:
        """Generate the env pool at its spec-default (or given) size.

        A preset env (:attr:`pool_preset` set) generates from that preset --
        its own axes and disjoint seed range -- rather than the module's
        default pool, so the same c22 module can back both the base pool and
        the hard-mode variant with no fork.
        """
        if self.pool_preset is not None:
            return self.pool_preset.generate(n_per_stratum=n_per_stratum)
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
        if split_fn is not None and self.pool_preset is None:
            return split_fn(pool)
        n_strata = len(pool.strata)
        internal, official, held_out = _SPLIT_PER_STRATUM_BY_ENV.get(
            self.name, _C22_SPLIT_PER_STRATUM
        )
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
    # A variant env id (c22h) loads its surfaces from another module (c22).
    module = _ENV_MODULE_NAME.get(name, name)
    generate = import_module(f"whetstone_envs.{module}.generate")
    oracle = import_module(f"whetstone_envs.{module}.oracle")
    prompts = import_module(f"whetstone_envs.{module}.prompts")
    # A preset env generates from a named Preset on its generate module and
    # takes that preset's version as its dataset revision, so its Task Set
    # identity is distinct from the base env's.
    preset_attr = _ENV_POOL_PRESET.get(name)
    preset = getattr(generate, preset_attr) if preset_attr else None
    generator_version = (
        f"{generate.GENERATOR_VERSION}+{preset.name}"
        if preset is not None
        else str(generate.GENERATOR_VERSION)
    )
    return EnvSpec(
        name=name,
        generate=generate,
        oracle=oracle,
        probes=prompts.PROBES,
        surface=probe_surface(name, prompts.PROBES),
        oracle_qualname=f"whetstone_envs.{module}.oracle.score_gold",
        token_estimate=_ENV_TOKEN_ESTIMATES[name],
        gold_first=name in _GOLD_FIRST_ENVS,
        stratified_split=name in _STRATIFIED_SPLIT_ENVS,
        pool_preset=preset,
        generator_version=generator_version,
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
    "ESTIMATE_LIVE_MEASURED",
    "ESTIMATE_SCALED_PENDING",
    "EnvSpec",
    "TokenEstimate",
    "UnknownEnvError",
    "env_spec",
]
