"""Session-scoped pool memoization for the env test suite.

Env pools are deterministic in ``(env name, n_per_stratum)`` -- the same
arguments reseed the same generator to the same bytes -- and a
:class:`~whetstone_envs.core.pool.TaskPool` is a frozen, immutable value, so
one generated instance is safe to share across every test that asks for the
same pool.

Most env tests exercise N-INDEPENDENT properties (split disjointness, identity
hashing, procedure partition, oracle agreement) and so build a tiny pool per
env; but the c18 / c18h pools reseed the vendored PrOntoQA generator through a
subprocess per depth (relevant-distractor generation is heavy rejection
sampling), so regenerating the SAME tiny pool in every test dominated the
suite. This autouse, session-scoped fixture wraps ``EnvSpec.generate_pool`` in
a per-session memo keyed on ``(env name, n_per_stratum)``, so each distinct
pool is generated exactly once for the whole session and reused thereafter.

This changes NO behaviour a test can observe: the wrapper returns the exact
pool the real generator would, just cached. The one test that must prove the
generator is deterministic across independent runs -- the committed-manifest
regeneration diff -- lives in whetstone-envs and is unaffected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from whetstone.envs.registry import EnvSpec

if TYPE_CHECKING:
    from collections.abc import Iterator

    from whetstone_envs.core.pool import TaskPool


@pytest.fixture(scope="session", autouse=True)
def _memoize_generate_pool() -> Iterator[None]:
    """Memoize ``EnvSpec.generate_pool`` for the test session.

    Keyed on ``(env name, n_per_stratum)`` -- the only inputs the pool
    depends on. Restored after the session so nothing leaks past the suite.
    """
    original = EnvSpec.generate_pool
    cache: dict[tuple[str, int | None], TaskPool] = {}

    def cached_generate_pool(
        self: EnvSpec, *, n_per_stratum: int | None = None
    ) -> TaskPool:
        key = (self.name, n_per_stratum)
        pool = cache.get(key)
        if pool is None:
            pool = original(self, n_per_stratum=n_per_stratum)
            cache[key] = pool
        return pool

    EnvSpec.generate_pool = cached_generate_pool  # type: ignore[method-assign]
    try:
        yield
    finally:
        EnvSpec.generate_pool = original  # type: ignore[method-assign]
