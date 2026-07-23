# Test conventions

## Two-tier suite (fast / slow)

The suite is split into two tiers by the `slow` pytest marker (declared in
`pyproject.toml` under `[tool.pytest.ini_options].markers`).

- **Fast tier (default).** `uv run pytest` (or `tests/`) runs everything
  EXCEPT `@pytest.mark.slow`. `addopts = "-m 'not slow'"` deselects the slow
  tier automatically, so the everyday run stays under ~60s wall time.
- **Slow tier.** `uv run pytest -m slow` runs ONLY the slow tests. These are
  irreducibly slow (>5s): today, the full-N c18h PrOntoQA pool regeneration
  that reseeds the vendored generator through a subprocess per depth (~18s)
  to pin the committed pool shape.

Deselected-by-default is NOT never-run: the slow tier runs as its own CI job
(`slow-tests` in `.github/workflows/whetstone_tests.yml`, via
`scripts/ci/slow.sh`) on every push/PR.

### When to mark a test `slow`

Mark a test `slow` only when it is irreducibly >5s AND its cost cannot be
removed by restructuring. Before reaching for the marker:

- **Prefer tiny-N.** Most env properties (split disjointness, identity
  hashing, procedure partition, oracle agreement) are N-INDEPENDENT. Build a
  tiny pool (`pool_n_per_stratum=2`, `split_sizes=(1, 2, 3)`) instead of the
  full committed pool. Keep exactly ONE test at full N to pin the real
  committed shape, and mark THAT one `slow`.
- **Share generation.** `tests/envs/conftest.py` memoizes
  `EnvSpec.generate_pool` for the session (keyed on `(env name,
  n_per_stratum)`); pools are deterministic and immutable, so the same tiny
  pool is generated once and reused across every env test. Route new pool
  builds through `env.generate_pool(...)` / `build_env_experiment(...)` so
  they hit this cache.

### Running

```sh
uv run pytest              # fast tier (default; slow deselected)
uv run pytest -m slow      # slow tier only
uv run pytest -m ''        # both tiers (override the default deselect)
```
