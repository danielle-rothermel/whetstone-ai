# InductionBench (Wenyueh/inductive_reasoning_benchmark) — Skeptical Repo Review

- **Repo:** Wenyueh/inductive_reasoning_benchmark (InductionBench)
- **URL:** https://github.com/Wenyueh/inductive_reasoning_benchmark
- **Review date:** undefined
- **License:** Apache-2.0 (LICENSE file present, full text; permissive)
- **Serving candidate ids:** c23 (Hidden-Rule String-Transform Induction Ladder — primary fit); c07 (Case-Marked Micro-Conlang — weak/no fit, no case morphology here)
- **Last commit:** 2025-06-26 (e0b8392); ACL 2025 long paper; actively cited 2025-2026
- **One-line verdict:** Conceptually ideal seeded subregular string-transform generator with an independent oracle, but the active path does NOT import/run as shipped (three import-time crashes) and is seed-nondeterministic unless `PYTHONHASHSEED` is pinned — reusable only as an algorithm to re-implement, not as drop-in code.

## Active Path

Entry (`standard_benchmark/inference.py`) -> `generate_rules` -> `generate_data` -> `generate_characteristic_sample` + `generate_fixed_size_dataset` -> `apply_*_rule` (oracle) -> `translate_input_output_pairs` (serialization/prompt). For c23 the LLM-call + precision/recall/compatibility scoring in `inference.py` would be replaced by our own single-call, 0/1-exact-match glue.

- **Active-path LOC (approx):** ~430 LOC. `standard_benchmark/synthetic_data_generation.py` ~330 lines on-path (generation + oracle), `utils.py` ~100 lines on-path (`generate_all_k_strings`, `translate_input_output_pairs`, `extract_answer`). `inference.py` scoring is ~60 lines but we would replace it. `model.py` is off-path for our purposes.

## Checklist Findings (file:line evidence)

### Seed plumbing — RED
- Seed is a **module-global** `random.seed(0)` set once in `inference.py:159` and `standard_benchmark/standard_run.py:116`. It is a hard-coded constant, not a threaded parameter; there is no way to re-seed for distinct strata/instances without editing source. To make N re-seedable instances we must add a seed param end to end.
- Vocab is another global: `config.vocab = list(...)` mutated at `inference.py:162`, set on the `config` module object; every generator function reads `config.vocab` (e.g. `synthetic_data_generation.py:26,58,61,195,312`). Global mutable state, not passed as an argument.
- **Set-iteration nondeterminism (named red flag):** `generate_ISL_rules` does `all_k_strings = list(set(all_k_strings))` (`synthetic_data_generation.py:53`); `is_minimality` and `possible_output = list(set(config.vocab).difference(...))` (lines 61,107,132) also iterate sets of strings. Because Python randomizes string hashing per process, `random.seed(0)` alone is **not reproducible**: I ran the generator twice with identical args and seed and got different instance hashes (`6e24c8b1...` vs `312481ca...`). Pinning `PYTHONHASHSEED=0` made both runs identical (`c91ecc15...`). The repo never sets `PYTHONHASHSEED`, so "seeded" is misleading as shipped. `translate_input_output_pairs` also `random.shuffle`s pair order (`utils.py:116`), fine once the RNG is truly seeded.

### Oracle independence — GREEN (this is the repo's strength)
- Ground truth is computed by **independent application logic**, not tautologically. Rules are sampled by `generate_ISL_rules`/`generate_OSL_rules`; the output for every input is computed by separate deterministic transducers `apply_ISL_rule` / `apply_L_OSL_rule` / `apply_R_OSL_rule` (`synthetic_data_generation.py:333-387`) that walk the string and match k-suffixes against the rule dict. I verified self-consistency: for every generated `(input, output)` pair, re-running `apply_rule(args, rules[i], input)` reproduced the stored output across multiple configs. So the oracle = "apply the latent rule set," genuinely separate from the sampler. Good fit for a hidden-rule induction ladder with exact-match scoring.

### Tests — RED (absent)
- No test files anywhere (`find -iname "*test*"` empty; no `pytest`/`unittest`/`def test_` hits). Only `__main__` demo blocks in `synthetic_data_generation.py:464` and `utils.py:216`. Nothing runnable to trust the generator/oracle beyond my own ad-hoc checks.

### Global state / hidden coupling / dead code on active path
- **Missing `config` module:** `import config` (`synthetic_data_generation.py:4`, `inference.py:3`, both `standard_run.py`) but there is **no `config.py` in the repo** and it is not in `.gitignore`'s tracked ignores as a project file. `import config` raises `ModuleNotFoundError` as cloned; the code relies on `config` existing purely as a namespace to hang `config.vocab` on. User must create a stub `config.py` (e.g. `vocab=[]`) before anything imports. Hidden required file.
- **`inference.py:4` `sys.path.add('..')` is a crash:** `sys.path` is a list; `.add` does not exist -> `AttributeError` at import time (verified). Should be `.append`. This runs before `from model import call_model` on line 5, so `inference.py` cannot even be imported unedited.
- **`synthetic_data_generation.py:8` imports a nonexistent function:** `from utils import ... translate_fewshot_input_output_pairs` — that function is **not defined in `utils.py`** (only `translate_input_output_pairs`). Top-level import -> `ImportError`, so the entire generator module fails to import as shipped (verified). Only `generate_few_shot_data` (dead for our path) would use it.
- **`exploration_benchmark/synthetic_data_generation.py:1-2`** uses `sys` before importing it (`sys.path.append` on line 2, `import sys` never at top) — that variant is also broken, but it is off our (ISL/OSL) path anyway.
- `model.py` has its own bugs (`steam=` typo instead of `stream=`, `system_prompt = system_prompt` self-reference) but is off-path since we write our own single-call glue.
- Dead/off-path on our route: `generate_few_shot_data`, `generate_example`, `reevaluate`, `--repeat` branch (reads prior result JSONs), the whole `exploration_benchmark/` (IOSL, no provable oracle).

## Adaptation-Diff Sketch

The generation+oracle core is small and, once the three import bugs are patched, usable. Estimated changes:

1. **Vendor 2 files** into our repo (Apache-2.0, keep NOTICE): `synthetic_data_generation.py` (~330 lines) and the 3 functions we need from `utils.py` (~40 lines). Do NOT vendor `inference.py`/`model.py`/`standard_run.py`.
2. **Fix 3 blockers** (~5 line edits): add a real `config.py` OR refactor `config.vocab` into an explicit arg; delete the `translate_fewshot_input_output_pairs` import (line 8); drop the whole few-shot path.
3. **Thread a seed (~15-25 new lines):** add `seed` param to `generate_rules`/`generate_data`, replace the module-global `random.seed(0)`, and set `PYTHONHASHSEED=0` in our launcher OR replace every `list(set(...))` over strings with `sorted(...)` to kill hash-order nondeterminism (preferred, ~4 call sites: lines 53,61,107,132,208,244). Without this the benchmark is not re-seedable — mandatory.
4. **New glue (outside repo, ~60-100 lines):** our own instance loop calling `generate_rules`+`generate_data` per stratum (vary `--type`/`--k`/`--vocab_size`/`--number_of_rules`), a single-LLM-call driver, a 0/1 exact-match scorer comparing the model's applied output (or its stated rule set fed back through `apply_rule`) against the oracle, latent-rule strata config, and optional nonce-vocab remapping of `config.vocab`.
5. For **c07** (case-marked conlang) this repo provides essentially nothing reusable — no morphology/case system, no glossing; would be a from-scratch build. c23 is the only real fit.

## Red Flags (summary)
- Does not import/run as shipped: missing `config.py`, `sys.path.add` crash (`inference.py:4`), import of nonexistent `translate_fewshot_input_output_pairs` (`synthetic_data_generation.py:8`). All three verified by execution.
- Not reproducible under `random.seed(0)` alone: `list(set(...))` over strings makes output depend on `PYTHONHASHSEED` (verified: differing hashes across runs). Never pinned in-repo.
- Zero tests; global mutable `config.vocab`; hard-coded seed constant, no seed parameter anywhere.
- Broken sibling code (`exploration_benchmark`, `model.py`) signals low overall code hygiene, though off our path.
- Strength: oracle is genuinely independent of the sampler and verifiably self-consistent — the core algorithm is sound and worth re-implementing.
