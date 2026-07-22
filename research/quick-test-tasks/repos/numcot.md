# Repo Review: CAS-SIAT-XinHai/NUMCoT

- **URL:** https://github.com/CAS-SIAT-XinHai/NUMCoT
- **Review date:** 2026-07-21
- **License:** CC0-1.0 (root `LICENSE:1-3`, "Creative Commons Legal Code / CC0 1.0 Universal"). Public-domain dedication; fully reusable. Not missing.
- **Serving candidate:** c02 Fictional Mixed-Radix Unit Conversion
- **One-line verdict:** The unit-conversion generator/oracle exists and is real Python, but it is uncalled dead code with NO seeding at all (`random.*` never seeded; the variable literally named `random_seed` is a 1-4 category selector, not a seed), the shipped experiment consumes frozen CSVs, the oracles are hard-coded to real metric/Chinese units (not re-parameterizable to invented radices), and the "hard" oracle is provably buggy — so serving c02 means rewriting the generator and oracle, not config + glue.

## Active path

Two candidate paths exist for the unit-conversion task; only one is actually wired, and it does NOT run the generator.

**Shipped/active path (`src/code/experience_second/`):**
- Entry: `run_unit_measurement.py:164-170` — module-level `process_data('chatglm26b', 20, 'zh', 'easy')` etc. (runs on import).
- Data source: `process_data` `:115-161` -> `generate_exist_data(0, language, level)` `:117`, which **reads a pre-generated CSV** `f'C:/unit_measurement_{level}_{language}.csv'` `:32-34`. Questions/answers are frozen on disk; the generator is never invoked here.
- Fan-out: 20 threads `:118-129`, each calls `call_api` (`commmon.py:3-12`) 5 times/prompt-style, `score(answer, response, level, language)` `:97-109`.
- Scoring: `score` `:9-28` — regex substring match of the frozen `answer` against the model response (`:12-13` for easy/medium; `eval(correct_answer)` then two-number match for hard `:19-24`). Serialization: CSV writer `:138-149`.

**Generator path (`generate_data.py`) — DEFINED BUT UNCALLED:**
- `generate_easy_question`/`solve_easy_question` `:17-67`, `generate_medium_question`/`solve_medium_question` `:69-125`, `generate_hard_question`/`solve_hard_question` `:128-195`, `replace_in_csv` `:201-249`.
- `grep` for call sites: none. No `__main__` block; the only importer, `test.py:1` (`from generate_data import *`), is a CSV-QA-counting script (`process_csv_files` `:5-59`) that never calls any `generate_*`/`solve_*`. So the generator/oracle is dead relative to any runnable entry point.

**Active-path LOC:** the runnable experiment is `run_unit_measurement.py` (170) + `commmon.py` (75) + `chat_api.py` (94, network) and consumes CSVs — the generator is not on it. The *reusable-for-c02* code is the ~130 lines of `generate_data.py:17-195` (generate+solve), which is off the live path.

## Checklist findings

### Seed plumbing — ABSENT (red flag: no seeding, unthreaded)
- There is **no `random.seed(...)` anywhere** in the repo (`grep -rn "seed(" src/ preprocess/` returns only the misnamed local below). Every generator uses module-global `random.randint`/`random.choice` with no seed and no injected RNG: `generate_data.py:19,30,32,34,76,78-81,85,130,141,143-147,152`.
- **Misleading name (red flag):** `random_seed = random.randint(1, 4)` at `:19` and `:130` is NOT a seed — it is a category index (`random_seed % 4` picks a unit family at `:21-28`, `:132-139`). A skeptic scanning for seeding would be misled; determinism is nonexistent.
- Not re-seedable: no seed parameter reaches any call site; two runs produce different problems. This directly fails the "deterministic re-seedable generator" requirement — we would add all seed plumbing ourselves.

### Oracle independence — INDEPENDENT LOGIC, but hard-coded and one oracle is BUGGY
- Easy/medium oracles are genuinely independent of generation: `solve_easy_question` `:38-67` re-parses the question string and recomputes via a `conversions` dict (`:41-47`); `solve_medium_question` `:89-125` uses a separate `conversion_factors` adjacency dict `:91-98`. These are not tautological — they recompute from the rendered string, not from generator internals. Good structurally.
- **But the conversion tables are hard-coded to real metric/time/currency/Chinese units** (`:41-47`, `:71-74`, `:91-98`, `:168-177`), keyed on Chinese unit glyphs. There is no table abstraction to swap in invented `grol/blen/mir` radices — the radices are literals embedded in every function. c02's "seeded conversion tables per run" has no seam here.
- **`solve_hard_question` `:161-195` is buggy** (red flag). Line 166 matches units with a character-class regex `r'(\d+)([吨千克克毫克]+)'` — a *set* of characters, so it does not cleanly tokenize multi-glyph units and can mis-group. Line 188 `total_milligrams -= 2*second_value` subtracts the third term *twice* after it was already added in the `:180-181` loop — an ad-hoc correction that assumes term ordering and is wrong for general inputs. The "answer" column in the shipped hard CSVs is whatever this produced. Not a trustworthy oracle.
- Scoring is substring regex, not exact match: `score` `:12` uses `(?<!\d){answer}(?!\d)(?!\.\d)` — lenient, and for hard it `eval()`s the answer string `:19`. We would replace this entirely with 0/1 exact match.

### Tests — NONE
- No test framework, no assertions. Files named `test.py` (`experience_second/test.py`, etc.) are ad-hoc run/QA scripts, not unit tests. No generator↔oracle agreement check, no determinism check. We bring all tests.

### Global state / coupling / dead code
- **Module-level side effects on import (red flag):** `run_unit_measurement.py:164-170` executes 6 `process_data(...)` calls (each spawning 20 network threads) at import time; `test.py:62` calls `process_csv_files()` at import; both hard-code Windows path `C:/` (`:32`, `test.py:5`). Not importable as a library without firing.
- Dead code: the entire `generate_data.py` generate/solve suite (see above) plus `replace_in_csv` `:201-249` (a zh->en CSV rewriter) are unreferenced by any entry point.
- `utils_of_num_to_word.py` (206 lines) is duplicated verbatim across `experience_first/second/third` — copy-paste divergence risk; not on the unit path.
- Nondeterministic dict use: not a correctness issue here (dicts are lookups, not iterated for ordering), so not flagged as set/dict red flag — the determinism failure is the missing seed, not iteration order.

## Adaptation-diff sketch

The generate/solve functions are a usable *reference* for the shape of a mixed-radix unit task, but almost nothing is reused as-is for c02. Estimated:
- **New generator (~80-120 LOC, NEW):** re-seedable `Random(seed)` per instance; sample an invented unit ladder with mixed radices (`1 grol = 7 blens`, `1 blen = 12 mirs`, ...) into a per-run table; sample a quantity; render the question. Cannot reuse `generate_data.py` bodies because units/radices are hard-coded literals there — only the *structure* (`:69-88` medium, `:128-159` hard) is a template. Seed plumbing is entirely new (repo has none).
- **New oracle (~40-60 LOC, NEW):** exact conversion over the *sampled* table (base-unit reduction), returning a canonical mixed-radix answer. `solve_easy/medium` `:38-125` are a loose model but must be rewritten around a passed-in table; `solve_hard` `:161-195` must be discarded (buggy).
- **Config (~15 LOC):** ladder depth / radix ranges / quantity ranges / strata (latent-rule: ladder length, radix magnitude) as the re-parameterization knobs.
- **Scoring glue (~20 LOC, NEW):** single LLM call + 0/1 exact-match on canonical form; drop the repo's substring-regex `score` `:9-28`, the 5-prompt-style fan-out, threads, and `C:/` CSV I/O.
- **Tests (~40 LOC, NEW):** determinism-by-seed; oracle round-trip (convert then invert); agreement on a hand-checked example.

Net: this is a **rewrite of generator + oracle + scoring**, using the repo only as a design reference. Not config + glue.

## Red flags (named)
1. **No seeding at all** — `random.*` never seeded (`generate_data.py`, all call sites); non-deterministic, not re-seedable. Fails the core c02 requirement.
2. **Misnamed `random_seed`** (`:19,:130`) is a 1-4 category selector, not a seed — actively misleading.
3. **Generator/oracle is dead code** — `generate_*`/`solve_*` are never called by any entry point; the shipped experiment reads frozen `C:/unit_measurement_*.csv` (`run_unit_measurement.py:32`). The "generator" ships as a frozen dataset in practice.
4. **`solve_hard_question` is buggy** — double-subtraction `:188` and char-class unit regex `:166`; untrustworthy oracle.
5. **Oracle tables hard-coded to real metric/Chinese units** (`:41-47,:91-98,:168-177`) — no seam for invented radices; radices are literals, not a swappable table.
6. **Module-level side effects + hard-coded `C:/` paths** (`run_unit_measurement.py:164-170`, `:32`; `test.py:5,62`) — not importable as a library; Windows-only.
7. **Scoring is lenient substring regex with `eval()`** (`score:12,19`), not 0/1 exact match.
8. **No tests** of any kind for the generator or oracle.
