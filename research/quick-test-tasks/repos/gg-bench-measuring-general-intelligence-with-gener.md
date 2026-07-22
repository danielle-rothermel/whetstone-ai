# gg-bench — Measuring General Intelligence with Generated Games

- **URL:** https://github.com/vivek3141/gg-bench
- **Paper:** arXiv:2505.07215 (2025), UC Berkeley
- **Review date:** undefined
- **License:** MIT (LICENSE file present, root)
- **Serving candidate ids:** c20 Invented Micro-Game Turn Resolution
- **One-line verdict:** Reusable frozen corpus of 1000 LLM-generated Gym micro-games whose `step()`/`render()`/`valid_moves()` triple is a clean per-turn oracle; our diff is pure config+scoring glue, but seeding is per-env inconsistent (must be handled in our harness) and there are zero tests, so provisional I1 = 2.

---

## What we would actually reuse

Not the generation pipeline (LLM-driven, needs API keys) and not the RL/eval pipeline (needs `stable-baselines3`/`sbx`, trained agents, MCTS self-play). For c20 the reusable artifact is the **already-committed frozen corpus**:

- `gg_bench/data/envs/env_{1..1000}.py` — one `CustomEnv(gym.Env)` per game. Uniform interface: `reset(seed,options)->(obs,info)`, `step(action)->(obs,reward,done,truncated,info)`, `render()->str` (human-readable board text), `valid_moves()->list[int]`.
- `gg_bench/data/descriptions/{id}.txt` — rich NL rulebook per game (latent-rule strata + invented/nonce vocab, e.g. "Number Flip Duel", "Match Count"; `descriptions/1.txt`).
- `gg_bench/data/actions/{id}.txt` — NL mapping of discrete action indices to moves (`actions/1.txt`).
- `gg_bench/data/splits/filtering.json` — `valid_envs` (316 ids that passed execution+timeout filtering), plus failure lists. This is the pool we want (does NOT require trained RL agents).
- `gg_bench/data/splits/valid_envs.json` — 126 `[env_id, rl_checkpoint_step]` (subset with trained agents; irrelevant to c20).
- `gg_bench/data/splits/{easy,medium,hard}.json` — difficulty strata (74/156/175).

The **ground-truth turn-resolution oracle** for c20 = the generated env's own `step()`. A single turn = load env, replay a seeded prefix of moves via `valid_moves()`, serialize with `render()`, ask the LLM for the next move, and check the resulting `step()` outcome (reward/done/next render) by exact match. No LLM call is needed to produce ground truth.

---

## Active path (what our instance generator would hit)

```
choose env_id (from filtering.json valid_envs)
  -> importlib env_{id}.CustomEnv()               [data/envs/env_{id}.py]
  -> reset(seed)                                   [env reset()]
  -> loop k times: valid_moves() -> pick seeded -> step()   [env]
  -> render()  (serialize state to text)           [env render()]
  -> read descriptions/{id}.txt + actions/{id}.txt (prompt context)
  -> [oracle] step(candidate_action) -> reward/done  [env step()]
```

**Active-path LOC (approx):** ~120–160 LOC per env (`env_1.py`=145, `env_100.py`=105) plus the existing reference driver `scripts/filter/execution.py:66-71` (the exact instantiate→valid_moves→step→render pattern, ~6 LOC). The upstream full eval driver `scripts/eval/eval.py` is 437 LOC but almost all of it (RL agent load, MCTS, self-play, multiprocessing, JSON locking) is OFF our path. Per-instance active logic we actually exercise is the single env file, ~120–160 LOC.

---

## Checklist findings (file:line evidence)

### Seed plumbing — RED FLAG (per-env inconsistent)
- Every env calls `super().reset(seed=seed)` (`env_1.py:27`, `env_100.py:19`; 769/772 sampled). For deterministic envs that is sufficient because there is no state RNG.
- BUT randomized envs are split: only 67 of 1000 use the correctly-threaded `self.np_random` (e.g. `env_113.py:26,34` sets `self.np_random, _ = seeding.np_random(seed)` then `self.np_random.randint(...)`), while **142 use module-global `np.random.*`** that ignores the seed (`env_1.py:30-31` `self.player1_secret = np.random.randint(...)`), and 14 use the unseeded stdlib `random` module. 617/772 sampled envs have **no** state randomness at all (pure combinatorial games) — those are fully reproducible.
- Empirically verified (venv gymnasium+numpy): `env_100` replay of a fixed move-index sequence is byte-identical across runs; `env_1.reset(seed=5)` alone does NOT reproduce its hidden secret (`False`), but pre-seeding module-global `np.random.seed()` before reset DOES (`True`). Mitigation is cheap and lives in OUR harness: `np.random.seed(s); random.seed(s)` before `env.reset(seed=s)`. Cleanest path: restrict to the ~617 deterministic + ~67 self.np_random envs, or just always global-seed.

### Oracle independence — GOOD (with a caveat)
- Ground truth is computed by the env's `step()` transition logic, which is **independent of any generation-time answer key** — there is no stored "correct move" that the generator emitted. The oracle is the game rules as code. For c20 (does move X produce outcome Y / is X legal / who wins) this is genuinely independent of the LLM under test.
- Caveat: the env code itself was LLM-generated, so "ground truth" = "whatever the generated rules implement," which may diverge from the NL `descriptions/{id}.txt`. This is inherent to the benchmark, not a tautology, but strata design should treat the code as authoritative and accept that a minority of envs may have rule/desc mismatches. The upstream `filtering.json` valid set already removed envs that crash or loop, not rule-mismatch ones.

### Tests — ABSENT
- No test/conftest/pytest files anywhere in the repo (`git ls-tree | grep -iE 'test|conftest'` outside `data/` = empty). No CI. The closest thing to a runnable check is `scripts/filter/execution.py:59-78` which runs `stable_baselines3.common.env_checker.check_env` + a 5-step smoke play; it is runnable but drags in the RL dependency and is a filter, not a unit test.

### Global state / hidden coupling / dead code
- `scripts/eval/eval.py:25-29` executes module-level `load_yaml(...)` and reads prompt/turn-count config at import — global import-time side effect, but off our path if we don't import that module.
- Randomized envs' reliance on module-global `np.random` (above) is the main hidden-state coupling.
- `MetadataEnv`/`TimeoutEnv`/`AlternatingEnv` wrappers (`utils/env_wrappers/`) and all of `utils/inference/mcts.py`, `utils/minimax.py`, `sbx`/`stable-baselines3` are dead code for c20 — needed only for RL self-play eval.
- `eval.py` uses `fcntl.flock` (POSIX-only) and `multiprocessing`; irrelevant to our reuse but note portability if anyone reuses the driver.

### License
- MIT, `LICENSE` at repo root. Permissive — reuse of the data corpus and code is fine with attribution.

---

## Adaptation-diff sketch (our diff = config + scoring glue, outside the repo)

We vendor the frozen `gg_bench/data/{envs,descriptions,actions,splits}` directory as a read-only asset and write a thin harness in OUR codebase. No edits to repo source required.

New glue (est. 120–200 LOC total, all outside the repo):
1. `loader.py` (~40 LOC): `importlib`-load `env_{id}.py` by file path (no package install), instantiate `CustomEnv`. Reuse the exact pattern from `execution.py:63-71`.
2. `instance_generator.py` (~60–90 LOC): given `(env_id, seed, prefix_len)` → set `np.random.seed`/`random.seed` + `env.reset(seed)`, replay `prefix_len` seeded picks from `valid_moves()`, capture `render()` as board text + `sorted(valid_moves())`. Emit instance = {description txt, actions txt, board render, legal moves, and the c20 question}. Re-seedable by construction because WE own the RNG and the move sequence.
3. `oracle.py` (~30 LOC): call `env.step(candidate)` (on a fresh replayed copy) to get reward/done/next-render; produce the 0/1 exact-match target.
4. `strata.py` (~20 LOC): filter to `filtering.json["valid_envs"]`, optionally intersect with deterministic-or-self.np_random envs (one-time static scan), map to easy/medium/hard from the split jsons for latent-rule strata.
5. Config: point at `filtering.json`; single LLM call per instance (our runner, not repo's `chat_completion`).

Dependencies we add: `gymnasium` + `numpy` only. We deliberately do NOT install `stable-baselines3`/`sbx`/`torch`/`shimmy` (RL-only). Verified envs run under just gymnasium+numpy.

---

## Red flags (named)
1. **Unthreaded seed in ~142/1000 envs** (module-global `np.random`, `env_1.py:30-31`); 14 more use unseeded stdlib `random`. Reproducibility requires OUR harness to global-seed, or restrict env pool. Not blocking but must be handled.
2. **Zero tests, no CI** — env correctness rests entirely on the LLM generator + the execution/timeout filters; no assertion that any env's rules match its NL description.
3. **Oracle = generated code**, so an env's `step()` may not faithfully implement its `descriptions/{id}.txt`. Treat code as authoritative; expect a minority of desc/code mismatches.
4. **Import-time global config load** in `eval.py:25` and POSIX-only `fcntl`/multiprocessing in the driver — avoid by not importing the eval module; write our own runner.
5. **Full-clone is heavy** (5228 files, mostly `data/models/*` RL checkpoints + 1000 envs); use blobless `--filter=blob:none --no-checkout` then sparse-checkout `data/{envs,descriptions,actions,splits}` — the `models/` tree (2128 files) is not needed for c20.

## Verification performed
- Blobless clone + selective checkout (full clone timed out on the RL-model blobs).
- Read end-to-end: `README.md`, `cli.py`, `scripts/eval/eval.py`, `scripts/generate/generate_envs.py`, `scripts/filter/execution.py`, `utils/env_wrappers/metadata.py`, `data/envs/env_1.py`, `data/envs/env_100.py`, `data/descriptions/1.txt`, `data/actions/1.txt`, `setup.py`, `requirements.txt`, all split jsons.
- Static scans across 772 checked-out envs for RNG patterns and reset signatures.
- Runtime smoke test in a fresh gymnasium+numpy venv: envs instantiate/step/render without the RL stack; confirmed deterministic-env replay reproducibility and the env_1 global-RNG seeding gap + its global-seed mitigation.
