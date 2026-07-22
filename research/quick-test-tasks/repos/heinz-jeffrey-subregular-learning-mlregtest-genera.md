# Repo Review: heinz-jeffrey/subregular-learning (MLRegTest generator)

- **Repo:** heinz-jeffrey/subregular-learning
- **URL:** https://github.com/heinz-jeffrey/subregular-learning
- **Review date:** (undefined)
- **License:** CC-BY-4.0 (LICENSE file present, ~18 KB; also shield in README). Permissive/attribution.
- **Serving candidate:** c06 Artificial-Grammar Violation Classification
- **One-line verdict:** Reusable ground-truth oracle (FSA composition) + 1470 committed language automata make this a strong donor for c06, but the sampling generators are unseeded/nondeterministic and we must write our own seeded sampler + scoring glue; oracle is what we actually reuse.

## Active path

Two independent generator entry points exist; both consume the same committed `.fst` automata.

1. `src/data-gen.py` (433 LOC) — `--lang <id>` -> reads `src/fstlib/fst_format/{id}.fst` (data-gen.py:350-351) -> rejection sampling of random strings (`random.choices`, data-gen.py:74,140,254) -> label by FSA acceptance `(A(s) @ fsa).num_states() != 0` (data-gen.py:76,147) -> writes `str\tTRUE/FALSE` to Small/Mid/Large dirs (data-gen.py:101-114).
2. `src/pathogen.py` (434 LOC) — newer, cleaner. `id` positional -> reads same fst (pathogen.py:78) -> builds a uniform-path weighted DAG (`burdenedDAG`, pathogen.py:171-221) and samples via `pynini.randgen` (pathogen.py:265,291,385) -> positives from `fsa`, negatives from `cofsa = difference(sigma*, fsa)` (pathogen.py:80-81) -> same tab-separated output.

Supporting: `src/lang_names.py` (124 LOC) enumerates language ids from filenames. `samples/evalsample.py` demonstrates the standalone oracle: `pynini.acceptor(test_str) @ fsa` non-empty == accepted (evalsample.py:32-34).

**Active-path LOC (what we'd actually read/reuse):** ~120 LOC of oracle + FST-load logic across data-gen.py + evalsample.py; the full generators are ~430 LOC each but we would replace most sampling code.

## Checklist findings

### Seed plumbing — RED FLAG
- `data-gen.py`: uses `random.choices`/`random.choice` at data-gen.py:47,74,140,254,265,274 with **no `random.seed()` anywhere** (grep for "seed" over the file: none). Fully nondeterministic across runs. Not re-seedable.
- `pathogen.py`: `unique_paths` has `seed=0` default (pathogen.py:376) passed to `pynini.randgen` (pathogen.py:385), BUT `create_data_with_duplicate` calls `pynini.randgen` with **no seed arg** (pathogen.py:265, defaults to pynini's internal), and the CLI exposes no `--seed` (pathogen.py:59-66). Seed is neither threaded from entry nor varied per length/call. So even the "seeded" path is not controllably re-seedable and `create_data_with_duplicate` is nondeterministic.
- Net: neither generator satisfies "deterministic RE-SEEDABLE". We must supply our own seeded sampler.

### Oracle independence — STRONG (this is the reusable core)
- Ground truth = FSA membership computed by composition, independent of how a string was sampled: `(A(s, token_type=syms) @ fsa).num_states() != 0` (data-gen.py:76,147) and `pynini.acceptor(test_str) @ fsa` (evalsample.py:32-34). This logic does not depend on the sampler and can label ANY string.
- Caveat: in `pathogen.py` the *training-set* labels are tautological (positives drawn from `fsa`, negatives from `cofsa`, pathogen.py:80-81,90-93) — no separate check. But the underlying membership oracle (composition) is available to re-derive labels independently. For c06 we would call the oracle directly, so the tautology in pathogen's sampler does not affect us.
- 1470 committed `.fst` binaries under `src/fstlib/fst_format/` across 13 subregular families (LT 90, LTT 180, PLT 90, PT 90, Reg 30, SF 30, SL 90, SP 90, TLT 150, TLTT 300, TPLT 150, TSL 150, TSL/Zp 30). These are the latent-rule strata for c06 out of the box.

### Tests — RED FLAG (none)
- No unit tests for generator/oracle. `find -iname "*test*.py"` returns only `src/testaut.py`, which is a DFA-prediction utility (testaut.py:2-10), not a test. `src/classvalidation/validate.py` validates that automata belong to the claimed subregular class (structural check), not the sampler/oracle. No pytest/unittest anywhere. Nothing runnable to confirm correctness beyond reading.

### Global state / coupling / dead code
- `data-gen.py` relies on **module-global** dir vars (`dirLarge`, `dirMid`, `dirSmall`, `dirLog`, `factor`, `num_ss`, `num_ls`, `largedata`) set only under `__main__` (data-gen.py:310-345) but read inside functions (e.g. data-gen.py:58-61,102,248) — functions are not importable standalone without setting these globals. Hidden coupling.
- Dead code: `data-gen.py` `fill_bucket` branches are all guarded by `elif False and ...` (data-gen.py:79,89,150,164) — dead. `create_adversarial_examples` in data-gen.py references undefined `cofsa` at line 155 (only reachable inside the dead branch).
- `dict`/`set` iteration is used for output ordering (`pos_dict[i] = set(...)`, data-gen.py:97-98,175-176); set-iteration order compounds the nondeterminism.
- Hard-coded relative paths (`src/fstlib/fst_format`, `data_gen/...`) assume cwd == repo root (data-gen.py:299-314, pathogen.py:50). Must run from repo root.
- Requires `pynini==2.1.2` (C++/OpenFst binding, conda-only, README:35,47) — a heavy, version-pinned native dependency. This is the main operational risk for standing it up.

### Adaptation-diff sketch (our diff = config + scoring glue, OUTSIDE the repo)
We do NOT reuse the samplers. We reuse: (a) the 1470 committed `.fst` files, (b) the composition oracle.
- **New file `c06_gen.py` (~90 LOC, outside repo):** load a chosen `.fst` via `pynini.Fst.read`; a *seeded* `random.Random(seed)` sampler that draws fixed-length strings over the fst's input alphabet (mirror data-gen.py:64-75 but with threaded `rng`); for MC "which rule does this string violate", pick a target language L and 3 distractor languages from a different/related family (use the 13-family tag in the filename `NN.NN.CLASS.k.i.j`); emit a string that violates L but respects distractors (label via oracle). Latent-rule strata = the CLASS tag.
- **New file `c06_oracle.py` (~20 LOC):** wrap `pynini.acceptor(s) @ fsa` -> bool, exactly evalsample.py:32-34. This is the 0/1 exact-match ground truth.
- **New scoring glue (~30 LOC):** single LLM call -> parse MC letter -> compare to oracle-derived correct index -> 0/1.
- **Repo files changed: 0** (we vendor the `.fst` directory + copy the 3-line oracle pattern). No edits to repo source needed; we treat it as a data+oracle donor.
- Est. new glue outside repo: ~140 LOC. Requires pynini install (conda) OR a one-time export of the automata to a pure-python FSA format to drop the native dep.

## Red flags (summary)
- No seeding in `data-gen.py`; `pathogen.py` seed not plumbed from CLI and one sampler path unseeded — not re-seedable as shipped.
- Zero generator/oracle tests; correctness rests entirely on reading.
- Module-global state + set/dict-iteration ordering => generators not importable/deterministic without rework.
- Dead code branches (`elif False`) referencing undefined names on the adversarial path in data-gen.py.
- Heavy pinned native dependency (pynini==2.1.2, conda-only); repo last code push 2024-07 (no recent code maintenance, only metadata touched 2025-12).
- Nonce vocab already satisfied: alphabet is abstract symbols (a,b,c,d,... up to 64), no natural-language leakage.

## Provisional score: 2
Genuinely reusable as an oracle + latent-rule-strata donor (independent composition oracle, 1470 committed automata, permissive CC-BY license, actively cited 2025-2026). Falls short of 3 because our diff is NOT merely config+scoring glue: the shipped samplers are unseeded/nondeterministic with dead code and global-state coupling, so we must write our own seeded sampler, and there are no tests to trust. The reliable part (oracle) is 3 lines; the generation path is not directly reusable.
