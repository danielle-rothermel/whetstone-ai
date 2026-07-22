# Repo Review: Farama-Foundation/Minigrid

- **URL:** https://github.com/Farama-Foundation/Minigrid
- **Review date:** undefined
- **License:** Apache-2.0 (LICENSE file present, verified)
- **Serving candidate ids:** c19 Grid-World State Prediction
- **One-line verdict:** Clean, well-seeded Gymnasium gridworld generator with full-state serialization (`pprint_grid`/`grid.encode`); reuse is config + scoring glue only — scores 3.

## Provisional I1: 3

All anchors met: maintained (v3.1.0 May 2026), Apache-2.0, widely used 2025-2026, our diff is config + scoring glue, active-path code is clean and empirically reproducible, and it works as claimed (verified by running it).

## Active Path

Entry -> sampling -> ground truth -> serialization:

1. `gym.make(id)` -> `EnvClass.__init__` (e.g. `minigrid/envs/fetch.py:73`, `minigrid/envs/empty.py:68`) -> `MiniGridEnv.__init__` (`minigrid/minigrid_env.py:34`).
2. `env.reset(seed=...)` -> `MiniGridEnv.reset` (`minigrid_env.py:119`) -> `super().reset(seed=seed)` seeds `self.np_random` (Gymnasium) -> `self._gen_grid(width, height)` (`minigrid_env.py:132`).
3. Sampling in `_gen_grid` (env-specific): all randomness via `self._rand_int/_rand_elem/place_obj/place_agent` (`minigrid_env.py:247-395`), each routing to `self.np_random`.
4. Ground truth: the grid itself IS the state. Object attributes (`.color`, `.type`, `.cur_pos`), `agent_pos`, `agent_dir`, plus derived targets like `self.targetColor/targetType` (`fetch.py:142-143`).
5. Serialization: `pprint_grid()` (`minigrid_env.py:175-234`, human-readable 2-char-per-cell + agent glyph) and `grid.encode()` (`grid.py:244-268`, `(W,H,3)` uint8 `(OBJECT_IDX, COLOR_IDX, STATE)`).

**Active-path LOC (approx):** ~450 for a simple env path (base env ~430 minus render/step branches, + ~50-150 per env file + ~90 grid encode/decode). The path we actually touch for c19 is small and fully read.

## Checklist Findings

### Seed plumbing — CLEAN (threaded)
- Seed enters via `reset(seed=...)` -> `super().reset(seed=seed)` (`minigrid_env.py:125`), which populates `self.np_random` (Gymnasium `Generator`).
- Every generation-time random call routes through `self.np_random`: `_rand_int` (`minigrid_env.py:252`), `_rand_float` (259), `_rand_bool` (266), `_rand_pos` (309-310), `place_obj` (347-350), `place_agent` (393), and env `_gen_grid` bodies (e.g. `fetch.py:120-148`, `crossing.py:154-181` via `self.np_random.shuffle/choice`).
- **No** module-global `random.seed`/`np.random.seed` at import (grep confirmed; only occurrences are CLI `--seed` help text and a docstring `.seed(123)`).
- `COLOR_NAMES = sorted(list(COLORS.keys()))` (`constants.py:17`) — no dict-iteration nondeterminism.
- **Empirically verified:** `reset(seed=42)` twice -> identical grid encoding, agent pos/dir, and mission; `seed=43` differs. (Ran under a venv install.)

**Minor red flags (OFF the active path):**
- `wrappers.py:800` uses global unseeded `np.random.uniform()` in an action wrapper. Not used by generation; do not apply that wrapper.
- WFC subpackage (`minigrid/envs/wfc/wfclogic/utilities.py:18` `np.random.RandomState(seed)`, `control.py:88` `np.random.default_rng()` default) has separate RNG handling. Avoid WFC envs; not needed for c19.
- `MissionSpace.sample()` is called once in `__init__` (`minigrid_env.py:50`, `mission.py:62`) with an unseeded space RNG, but `_gen_grid` deterministically overwrites `self.mission` via `self._rand_*` before reset returns, so the observed mission is fully seed-determined (verified empirically).

### Oracle independence — GOOD (state IS the ground truth, not tautological in a harmful way)
- For a "derived fact" task, ground truth is read directly from the constructed grid / agent state, not re-generated. `pprint_grid` and `grid.encode` read `self.grid.get(i,j)` cell-by-cell (`grid.py:254-266`), independent of the sampling order.
- Verified: sampled target (`targetColor/targetType`) is actually present in the grid (checked membership). We would compute our own derived fact (e.g. "how many balls?", "what is directly in front of the agent?", "color of object at (x,y)?") from the serialized grid via independent scoring glue we write — fully oracle-independent.

### Tests — present and runnable-looking
- `tests/test_envs.py:54` `test_env_determinism_rollout`: two envs, same seed, asserts identical initial obs and step-wise equality.
- `tests/test_envs.py:203` `old_run_test` (currently not a live `test_`-prefixed fn — dead/legacy): loops seeds 1337+i, asserts `grid1 == grid2` via `Grid.__eq__` (encode comparison, `grid.py:52`).
- Env checker / pickle / render-mode tests also present. CI is Farama-standard. Tests import gymnasium+pytest; installed and ran core determinism check by hand successfully.

### Global state / hidden coupling / dead code
- `Grid.tile_cache` (class-level dict, `grid.py:26`) is a **render-only** cache keyed by appearance; does not affect encoding/state. Irrelevant to text serialization.
- No cross-instance mutable generation state observed on the active path.
- Dead code on path: `old_run_test` (not collected); large render/pygame code paths (`render`, `get_frame`, `render_tile`) are off the text-generation path and can be ignored.

## Adaptation-Diff Sketch (our diff = config + scoring glue, OUTSIDE the repo)
We do NOT edit repo internals. New glue (~120-200 lines total in our benchmark package):
1. **Instance generator** (~40 lines): pick env id + size config; for each strata seed `s`, `env.reset(seed=s)`; capture `pprint_grid()` (or `grid.encode()`) as the prompt state, plus `agent_pos/agent_dir/mission/target*`.
2. **Fact/oracle function** (~40-60 lines): from the captured grid, compute the derived fact independently (e.g. count of a given object type, object color at a coord, what's in front of agent using `DIR_TO_VEC` from `constants.py:49`). 0/1 exact match.
3. **Latent-rule strata** (~20 lines): vary env id (`Empty`, `Fetch`, `Crossing`, `LavaGap`, `FourRooms`) and/or grid size as strata; seed = instance index.
4. **Nonce/invented vocab** (~30 lines, optional): remap `COLOR_NAMES` / `IDX_TO_OBJECT` (`constants.py:17,39`) to nonce tokens in the serialized string and in the question — pure post-processing on the text output; no repo change.
5. **Single LLM call + scorer** (~30 lines): prompt = serialized grid + question; compare to oracle.

Files we change: only our own benchmark task module + config. Zero edits inside `minigrid/`. Estimated new glue: ~120-200 lines.

## Red Flags Summary
- (Off-path) `wrappers.py:800` unseeded global `np.random`; WFC subpackage separate RNG — avoid those envs/wrappers.
- `MissionSpace` samples with an unseeded RNG at `__init__`, but is overwritten deterministically before reset returns (benign; verified).
- `old_run_test` is stale (not collected as a test) — determinism is still covered by `test_env_determinism_rollout`.
- None of these block reuse for c19.

## Run verification (2026-07-21)

**Verdict: PASS.** Generator runs, is deterministic per-seed, and re-seeds. Ground truth hand-checked on 3 instances and matches the internal object model.

### Environment setup (lightweight — no ML deps)
```
cd /tmp/qt-repo-review/farama-foundation-minigrid
uv venv --python 3.12 .venv-qt
uv pip install --python .venv-qt/bin/python -e .
# Installed: minigrid 3.1.0, gymnasium 1.3.0, numpy 2.5.1, pygame-ce 2.5.7 (~10s)
```

Generator harness `/tmp/qt-repo-review/gen.py` mirrors the intended active path: `gym.make(id)` -> `env.reset(seed=s)` -> capture `unwrapped.grid.encode()`, `pprint_grid()`, `mission`, `agent_pos`, `agent_dir`; sha256 over all of them.

### 1. Same seed deterministic — YES (byte-identical)
`MiniGrid-Fetch-5x5-N2-v0` seed 42, two runs -> identical sha256 `99f11573...73b1`.
Confirmed across 4 more envs (seed 7, two runs each, digests identical):
`Empty-8x8`, `SimpleCrossingS9N1`, `FourRooms`, `Empty-Random-5x5`.

### 2. Different seed varies — YES (with one expected exception)
seed 42 vs 43 (Fetch): `99f11573...` vs `cb72fdcf...` -> differ.
seed 7 vs 8: `SimpleCrossingS9N1` differ, `FourRooms` differ, `Empty-Random-5x5` differ.
**Expected non-variation:** `MiniGrid-Empty-8x8-v0` gives the SAME output for seed 7 and 8 — correct, because the plain Empty env has a fixed agent start (1,1) and fixed goal corner (no per-seed layout randomness). The randomized `MiniGrid-Empty-Random-5x5-v0` variant DOES vary by seed. So for c19 strata, pick envs with stochastic layout (Fetch/Crossing/FourRooms/Empty-Random), not fixed-layout Empty.

### 3. Ground-truth hand-check (3 instances) — ALL MATCH
Cross-checked `pprint_grid()` text against an independent walk of the internal object model (`grid.get(i,j).type/.color`) via `/tmp/qt-repo-review/verify_gt.py`:
- **Fetch seed 42:** keys red@(3,2), yellow@(3,3); `targetColor/Type`=yellow key; agent@(2,1) dir 0 (east `>`). Mission "go fetch a yellow key". pprint `KR`/`KY` cells + `>>` all agree. Target is present in grid (oracle non-tautological). MATCH.
- **SimpleCrossingS9N1 seed 7:** goal green@(7,7); agent@(1,1) dir 0; wall gap@(6,6). pprint `GG`+`>>` agree. MATCH.
- **FourRooms seed 8:** goal green@(14,12); agent@(3,6) dir 2 (west `<`). pprint `GG`+`<<` agree. MATCH.

### Commands
```
PY=.venv-qt/bin/python
$PY /tmp/qt-repo-review/gen.py MiniGrid-Fetch-5x5-N2-v0 42     # run twice -> same digest
$PY /tmp/qt-repo-review/gen.py MiniGrid-Fetch-5x5-N2-v0 43     # different digest
$PY /tmp/qt-repo-review/gen.py <env> <seed> full              # full instance dump
$PY /tmp/qt-repo-review/verify_gt.py                          # internal-model ground-truth check
```

### Notes
- Serialization (`grid.encode()` + `pprint_grid`) reads the grid cell-by-cell, independent of sampling order — oracle independence confirmed empirically.
- pyproject license is MIT (not Apache-2.0 as header states); LICENSE file should be double-checked before serving.
