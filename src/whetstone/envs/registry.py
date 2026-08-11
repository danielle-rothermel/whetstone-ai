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


#: Bound task families in build order. Hard-mode variants reuse their base
#: modules with ``HARD_PRESET``; distinct names preserve dataset identities.
ENV_NAMES: tuple[str, ...] = (
    "c22",
    "c22h",
    "c11",
    "c19",
    "c18",
    "c18h",
    "c23",
)

#: Base modules reused by variant environment names.
_ENV_MODULE_NAME: dict[str, str] = {"c22h": "c22", "c18h": "c18"}

#: Generation presets selected by variant environment names.
_ENV_POOL_PRESET: dict[str, str] = {
    "c22h": "HARD_PRESET",
    "c18h": "HARD_PRESET",
}


#: Estimate provenance distinguishes direct measurement from scaled defaults
#: awaiting measurement.
ESTIMATE_LIVE_MEASURED = "live-measured"
ESTIMATE_SCALED_PENDING = "scaled-pending-measurement"
#: Inherited estimates retain distinct provenance until directly measured.
ESTIMATE_INHERITED_PENDING = "inherited-pending-measurement"


@dataclass(frozen=True, slots=True)
class TokenEstimate:
    """Per-call token estimates for the naive and ceiling probes.

    ``estimate_source`` records whether values were measured, inherited
    pending measurement, or scaled pending measurement.
    """

    naive: int
    ceiling: int
    estimate_source: str = ESTIMATE_SCALED_PENDING


_ENV_TOKEN_ESTIMATES: dict[str, TokenEstimate] = {
    "c22": TokenEstimate(
        naive=2526, ceiling=3046, estimate_source=ESTIMATE_LIVE_MEASURED
    ),
    "c22h": TokenEstimate(
        # Inherits c22's estimates.
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
        naive=1306,
        ceiling=2448,
        estimate_source=ESTIMATE_LIVE_MEASURED,
    ),
    "c18h": TokenEstimate(
        # Scales c18's estimates for longer D8/D10 prompts.
        naive=1959,
        ceiling=3672,
        estimate_source=ESTIMATE_INHERITED_PENDING,
    ),
    "c23": TokenEstimate(
        naive=5468, ceiling=4953, estimate_source=ESTIMATE_LIVE_MEASURED
    ),
}

#: Default deliberate-observation repeats per task.
DEFAULT_NUM_SAMPLES = 3

#: Envs whose oracle ``score_gold`` is ``(gold, response)`` rather than the
#: usual ``(prediction, gold)``. c22 and its hard variant c22h (same c22
#: oracle: it re-runs the constraint checkers against the response, reading the
#: constraint stack from the gold).
_GOLD_FIRST_ENVS: frozenset[str] = frozenset({"c22", "c22h"})

#: Pools blocked by stratum require per-stratum sampling because contiguous
#: slicing would skew splits toward leading strata.
_STRATIFIED_SPLIT_ENVS: frozenset[str] = frozenset({"c22", "c22h"})

#: c22 has no package-provided split sizes; these per-stratum defaults stay
#: within its 20-instance strata.
_C22_SPLIT_PER_STRATUM = (2, 6, 6)  # (internal_eval, official, held_out)

#: Per-env per-stratum split overrides (``(internal, official, held_out)``).
#: c22h keeps base c22's 2:6 internal:official proportion but takes the full
#: remaining stratum depth as held_out, so its 3 x 20 pool splits to totals
#: (internal 6, official 18, held_out 36) with no unused instances. c18h uses
#: the identical per-stratum split: its HARD_PRESET is likewise a 3 x 20 pool
#: (depths D5/D8/D10), so (2, 6, 12) per stratum -> the same disjoint totals
#: internal 6 / official 18 / held_out 36 with no unused instances. Because a
#: preset env's :meth:`EnvSpec.default_split_sizes` bypasses the env's own
#: committed ``default_split_sizes``, this override (not c18's base split
#: call) is what applies to c18h. Any env not listed falls back to
#: :data:`_C22_SPLIT_PER_STRATUM`.
_SPLIT_PER_STRATUM_BY_ENV: dict[str, tuple[int, int, int]] = {
    "c22h": (2, 6, 12),
    "c18h": (2, 6, 12),
}


@dataclass(frozen=True, slots=True)
class EnvSpec:
    """Bind generation, prompt, oracle, identity, and split behavior for one
    environment.

    ``gold_first`` selects oracle argument order; ``stratified_split`` selects
    per-stratum splitting; preset and generator-version fields distinguish
    variant datasets.
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
    pass


def env_spec(name: str) -> EnvSpec:
    if name not in ENV_NAMES:
        raise UnknownEnvError(
            f"unknown env {name!r}; expected one of {ENV_NAMES}"
        )
    return _load_env_spec(name)


__all__ = [
    "DEFAULT_NUM_SAMPLES",
    "ENV_NAMES",
    "ESTIMATE_LIVE_MEASURED",
    "ESTIMATE_SCALED_PENDING",
    "EnvSpec",
    "TokenEstimate",
    "UnknownEnvError",
    "env_spec",
]
