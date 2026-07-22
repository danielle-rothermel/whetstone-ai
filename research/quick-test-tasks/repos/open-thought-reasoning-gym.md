# Repo Review: open-thought/reasoning-gym

- URL: https://github.com/open-thought/reasoning-gym
- Review date: undefined
- License: Apache-2.0 (`LICENSE`, standard Apache 2.0 header) — permissive, no obstacle
- Serving candidate ids: c04 Term-Rewriting to Normal Form
- One-line verdict: A clean, per-instance re-seeded generator framework with a genuine term-rewriting-to-normal-form task (`ab.py`: 4 productions, rewrite neighbors until normal form, 0/1 exact match) and an independent oracle — everything c04 wants EXCEPT that the rewrite rules and symbols are FIXED (the A::B system), not seeded/invented per instance, so "latent-rule strata over invented symbols" needs a modest generator edit rather than pure config glue.

---

## Active path we would hit

For c04 the native task is `ab` = the A::B system (Victor Taelin's puzzle): an abstract rewrite system over 4 tokens with 4 productions, applied to a neighbor pair until no rule fires (normal form), scored by exact string match. Path:

`create_dataset("ab", seed=..., size=..., length=...)` (`factory.py:51`) — builds `ABConfig`, calls `config.validate()`, instantiates `ABDataset`
-> `ABDataset.__init__` (`ab.py:71`) -> `ProceduralDataset.__init__` (`dataset.py:13`) sets `self.seed`
-> `ABDataset.__getitem__(idx)` (`ab.py:74`) — the generator entry:
  - `rng = Random(self.seed + idx)` (`ab.py:83`) — per-instance RNG, the ONLY random source
  - `generate_program(length, rng)` (`ab.py:11`) samples the initial term via `rng.choice`
  - `compute_steps(initial_program)` (`ab.py:17`) — the ORACLE: applies the 4 productions leftmost-first until fixpoint (`ab.py:40-42`) or loop/step-cap (`ab.py:44-51`); rejects non-halting programs and resamples (`ab.py:85-89`)
  - serializes `{"question": prompt, "answer": " ".join(steps[-1]), "metadata": {...}}` (`ab.py:117-127`)
-> `ABDataset.score_answer(answer, entry)` (`ab.py:129`) — strict `answer == entry["answer"]` -> 1.0 else 0.0.

**Active-path LOC:** `ab.py` is 165 lines end to end (generator + oracle + config + curriculum + scorer). Framework scaffolding on the path: `dataset.py` `ProceduralDataset` (~75 relevant lines, 13-88) and `factory.py` (119 lines, register/create). Roughly **250-300 LOC** total on the path we'd exercise, all of it read.

## Checklist findings

### Seed plumbing — CLEAN (green)
- Single random call site, fully threaded: `rng = Random(self.seed + idx)` (`ab.py:83`), and `generate_program` takes that `rng` explicitly (`ab.py:11,14`). No bare module-global `random.*` in the content path.
- `self.seed` is set once in `dataset.py:20`: `seed if seed is not None else Random().randint(0, 2**32)` — if you pass a seed it is deterministic; only the seed==None fallback draws entropy, and we always pass a seed.
- Determinism verified empirically (I inlined the pure functions since full import needs `yaml`): seed=42, idx 0-3 produced identical normal forms across two runs. Repo's own `test_ab.py:19` asserts `dataset1[i] == dataset2[i]`.
- Re-seedability: `Random(seed + idx)` means instance N is a pure function of (seed, idx) — exactly the re-seedable oracle model. `ReseedingDataset` (`dataset.py:93`) derives chunk seeds as `(base_seed + chunk_num) % 2**32` (`dataset.py:118`) for an infinite stream. No `random.seed(None)`, no framework assertion against seeding.
- Minor note (NOT a red flag for us): `compute_steps` uses a `set` `seen_states` of tuples (`ab.py:23,44,49`) only for loop detection — membership test, never iterated for output ordering, so no set-iteration nondeterminism reaches the answer.

### Oracle independence — GOOD (green)
- `compute_steps` (`ab.py:17-51`) is an independent term-rewriting interpreter: scans left to right, applies the first matching production (`ab.py:28-35`), stops at a fixpoint. The answer is `steps[-1]` — the computed normal form, NOT an echo of a stored value. Not tautological: the generator samples a random term and the oracle re-derives its normal form.
- The 4 productions are hard-coded as elif branches (`ab.py:28-35`), mirroring the prompt's stated rules (`ab.py:100-108`) — prompt and oracle share the same rule set (correct by construction), but the normal form itself is genuinely computed.
- `test_ab.py:49-63` pins scoring against the oracle's own answers plus injected wrong answers and `None`.

### Scoring — MATCHES our 0/1 requirement (green)
- `ABDataset.score_answer` (`ab.py:129-144`) is already strict `answer == entry["answer"]` -> {1.0, 0.0}. This overrides the base class's lenient substring reward (`dataset.py:63-72`), so we need NO scoring glue for the exact-match variant.
- Optional `score_answer_cascade` / `cascade_score` (`dataset.py:74`, `scoring.py`) exists for lenient matching but is opt-in; we simply do not call it.

### Tests — runnable, focused (green)
- `tests/test_ab.py` (137 LOC): config validation, determinism (`:19`), program-generation invariants (`:29`), scoring 1.0/0.0/None (`:49`), iteration stability (`:66`), item structure (`:83`), curriculum level up/down + bounds (`:103`). Standard pytest, no network. `tests/test_string_synthesis.py` also present for the sibling task.

### Global state / coupling / dead code
- Only global state is the dataset registry `DATASETS`/`CURRICULA` dicts (`factory.py:14-15`), populated by `register_dataset(...)` at import (`ab.py:165`). Read-only after import; fine.
- Import cost: `reasoning_gym.coaching.__init__` pulls `yaml` (`curriculum_config.py:4`), so a bare `import reasoning_gym.algorithmic.ab` fails without PyYAML installed. Deps are declared in `pyproject.toml`; install-deps issue, not a code defect.
- No dead code on the path. `ab.py` is self-contained.

### Sibling task worth noting
- `reasoning_gym/algorithmic/string_synthesis.py` (165 LOC) is a SECOND rewrite-to-fixpoint task (9 block types, 6 combination rules, iterate until no rule applies or a state repeats; `_apply_rule` `:61`, `_get_answer` `:100`). Same clean `Random(self.seed + idx)` seeding (`:117`), independent oracle, exact-answer string. A multiset-rewriting variant rather than string rewriting; viable second stratum or fallback for c04.

### c04 fit gap — rules/symbols are FIXED, not seeded/invented
- c04 asks for "seeded abstract rewrite systems (productions over invented symbols)" with "latent-rule strata." `ab.py` uses ONE fixed rule set (A::B, tokens `A# #A B# #B`) hard-coded in both prompt (`ab.py:100-108`) and oracle (`ab.py:28-35`). What is seeded is the initial TERM, not the RULE SET or the SYMBOLS.
- The curriculum stratifies only on `length` (`ab.py:154-160`), i.e. term length — not on latent rule families.
- Consequence: to get invented-symbol, per-seed rule systems and latent-rule strata, we must generalize the generator+oracle to sample a rule table from the seed. A real (if modest) code edit, not config.

## Adaptation-diff sketch

Two options depending on how literally c04's "invented symbols / latent rules" must be honored.

**Option A — use A::B as-is (fixed rules), config + trivial glue only:**
1. Config: `create_dataset("ab", seed=S, size=N, length=L)` per stratum, strata = `length in {10,25,50,100}` (existing curriculum). ~10 lines.
2. Scoring: reuse `ABDataset.score_answer` (already 0/1 exact match). 0 lines.
3. Harness glue: single LLM call, feed `item["question"]`, compare model output to `item["answer"]` via `score_answer`. ~20-30 lines of our runner.
Total: **~30-40 lines of glue, no repo edits.** But this does NOT deliver invented symbols or latent-rule strata — one fixed rewrite system only.

**Option B — true seeded abstract rewrite systems (matches c04 fully):**
1. New file `reasoning_gym/algorithmic/term_rewriting.py` (~120-160 LOC), modeled directly on `ab.py`: from `Random(seed+idx)` sample (a) an alphabet of invented nonce symbols, (b) a set of neighbor productions, (c) an initial term; generalize `compute_steps` to a rule-table lookup instead of the 4 hard-coded elifs; render prompt from the sampled rule table; keep the exact-match `score_answer`. Register via `register_dataset`. Net-new authoring but a near-clone of an existing, correct 165-line file.
2. Curriculum with latent-rule strata (rule-count / alphabet-size levels) — ~20 lines, copy the `ScalarAttributeDefinition` pattern (`ab.py:154`).
3. Harness glue as in Option A. ~30 lines.
Total: **~150-180 lines net-new, structurally a copy-and-generalize of `ab.py`**, reusing the framework's seeding, registry, curriculum, and scoring wholesale.

Because a clean, independent term-rewriting generator+oracle already exists and is trivially re-seedable, the diff is either pure config (Option A) or a bounded generalize-an-existing-file edit (Option B) — not framework surgery. The sketch is writable, so the repo clears the bar to score.

## Red flags (summary)
- **c04's invented-symbol / latent-rule requirement is not native:** `ab.py` uses a single fixed rule set and fixed tokens; only the initial term is seeded. Latent-rule strata require the Option-B generator edit (~150 LOC clone-and-generalize). The one thing keeping it from a pure-config zero-edit 3.
- Import-time dep: `coaching/__init__` imports `yaml`, so importing the task needs declared deps installed (not a code bug).
- (Non-issues, for the skeptic:) the `seen_states` set is loop-detection only, never affects output order; `self.seed` entropy fallback only triggers when seed is None, which we never do.

## Provisional I1: 3
Maintained (last push 2026-04-17), Apache-2.0, actively used (NeurIPS 2025 Spotlight), well-written, and demonstrably works from the read (determinism reproduced empirically; oracle is an independent rewriter; scoring is already 0/1 exact match). A genuine term-rewriting-to-normal-form task exists on a clean per-instance re-seeded path. The only shortfall vs c04 is that rules/symbols are fixed rather than seeded/invented — but the fix is a bounded clone-and-generalize of an existing 165-line file plus config, i.e. still "near-clone + glue" territory, not major surgery. Scored 3 with the Option-B edit noted; if the benchmark demands strictly zero generator edits, treat as a strong 2.
