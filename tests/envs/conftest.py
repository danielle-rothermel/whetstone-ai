from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from whetstone.envs.registry import EnvSpec

if TYPE_CHECKING:
    from collections.abc import Iterator

    from whetstone_envs.core.pool import TaskPool


@pytest.fixture(scope="session", autouse=True)
def _memoize_generate_pool() -> Iterator[None]:
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
