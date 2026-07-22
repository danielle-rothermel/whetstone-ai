# Repo Review: sileod/reasoning_core

- **URL:** https://github.com/sileod/reasoning_core
- **Review date:** undefined
- **License:** MIT (`LICENSE`, Copyright (c) 2025 sileod) — permissive, no obstacle
- **Serving candidate ids:** c01 Invented-Glyph Positional Base Arithmetic
- **One-line verdict:** Well-engineered seeded-generator framework and a solid base-10 arithmetic oracle, but (a) NO invented-glyph / positional-base task exists and (b) the architecture is deliberately NON-reseedable — re-seedable determinism would require framework surgery, so this is major-surgery territory for c01.

---

## Active path we would hit

For c01 the closest native task is `Arithmetics` (`reasoning_core/tasks/arithmetics.py`). Path:

`get_task("arithmetics")` (`reasoning_core/__init__.py` lazy loader)
-> `Task.generate_example()` (`template.py:489`) — the harness entry, wraps generation with timeout/token filtering/metadata
-> `Arithmetics.generate_entry()` (`arithmetics.py:293`) — samples an expression tree via `gramforge.generate` then `fill_num`
-> `fill_num()` (`arithmetics.py:178`) — samples numeric literals, calls `_evaluate` to get the value
-> `_evaluate()` (`arithmetics.py:121`) — the ORACLE: AST-walks the expression with exact `Fraction` arithmetic
-> `Entry(metadata=..., answer=...)` (`arithmetics.py:308`) serialized by `template.py:582-590` / `generation_worker.serialize_example` (`generation_worker.py:28`).

**Active-path LOC (approx):** ~330 in `arithmetics.py` (`Arithmetics` class + `_evaluate` + `fill_num` + formatting helpers, lines 1-333) + ~250 relevant lines of `template.py` (`Task`, `Config`, `generate_example`) + `score_scalar` in `utils/__init__.py:8`. Roughly **550-600 LOC** on the path we'd actually exercise.

## Checklist findings

### Seed plumbing — RED FLAG (disqualifying for our re-seedable requirement)
- Generation uses the **module-global `random`**: bare `random.random()`, `random.randint`, `random.choice`, `random.sample`, `random.shuffle` throughout `arithmetics.py` (e.g. `fill_num` 187-194, `generate_entry` 297,303, word-problem gens 406-521). No per-instance/threaded `random.Random(seed)` for content sampling.
- `Config.seed` exists but is threaded ONLY into stochastic difficulty rounding: `stochastic_rounding(..., seed)` (`template.py:137-142`, used via `__getattribute__` 652 and `_ROUNDING_SEED` ContextVar 681). It never seeds the content RNG.
- `Task.generate_example` (`template.py:489`) takes NO seed argument and never seeds the RNG.
- The framework is **actively designed to be non-reproducible**: `Task.validate` asserts `r1 != r2` with comment "Example generation should not set a seed" (`template.py:430-433`); `generation_worker.run_task` calls `random.seed(None)` / `np.random.seed(None)` (`generation_worker.py:41-42`) to reseed from OS entropy for diversity.
- Net: there is no way to say "give me instance N under seed S deterministically." This is the opposite of a RE-SEEDABLE generator. Named red flag: module-global RNG + intentional entropy reseed.

### Oracle independence — GOOD (for the arithmetic task)
- The ground truth is computed by `_evaluate` (`arithmetics.py:121-156`), an independent AST interpreter over the expression STRING using exact `Fraction` math, not by echoing a generation-time value. `generate_entry` re-derives the answer from the final expression (`arithmetics.py:298-299`).
- Tests pin the oracle to hardcoded expected outputs independent of the generator: `_evaluate("0.3 // 0.1","exact")==3` vs `"python"==2.0` (`test_arithmetics.py:62-63`), gcd/lcm/prime ops (18-30). Not tautological.

### Scoring — MISMATCH with our 0/1 requirement
- Default `Arithmetics.score_answer` uses `score_scalar` (`utils/__init__.py:8-35`), a CONTINUOUS reward: `exp(-k*normalized_error) * format_reward`, NOT exact match. For non-"normal" digit modes it does fall back to exact string compare (`arithmetics.py:325-327`). We would override `score_answer` for a strict 0/1.

### Tests — runnable, focused
- `tests/test_arithmetics.py` (90 LOC): 7 tests covering oracle values, semantics cues, digit perturbation, canonicalization, rounding, number-theory ops. Standard pytest, no network; look runnable given deps installed. Framework-wide smoke via `Task.validate` (`template.py:383-448`).

### Global state / coupling / dead code
- Module-global grammar `g = _grammar()` built at import (`arithmetics.py:88`) — shared, fine for read-only use.
- Heavy import-time deps: `gramforge` (grammar sampler, hard dep in pyproject), `sympy`, `tiktoken`, `wrapt`, `easydict`, `nfsdict`, `psutil`, `udocker` (imported in `utils/__init__.py:6` at module top — pulls udocker even for `score_scalar`). `reasoning_gym` is optional.
- `template.py` carries subprocess/timeout/udocker-killing machinery (`timeout_retry` 151-199) irrelevant to arithmetic but on the wrapper path.
- The second half of `arithmetics.py` (338-587, `MathWordProblem`) is a whole separate task sharing the file — not on our path.

### There is NO invented-glyph / positional-base task
- Grep across `reasoning_core/tasks/*.py` for glyph/legend/radix/numeral/base-conversion finds nothing native. `Arithmetics` is strictly base-10 decimal (values formatted as decimals, `_output_decimal` 159, `_canonical_decimal` 241).
- `base_conversion` appears only as a string in `_reasoning_gym.py:13` — a passthrough NAME to the external `reasoning_gym` library (`Reasoning_Gym.generate_entry`, `_reasoning_gym.py:51-70`), not code in this repo. Its generator/oracle/seed behavior live in a different package and would need separate review; and it still inherits this repo's non-reseedable wrapper.

## Adaptation-diff sketch

Because no glyph/base task exists, "config + scoring glue" is NOT achievable. To serve c01 we would have to AUTHOR a new task from scratch plus fight the framework on determinism:

1. **New file** `reasoning_core/tasks/glyph_base_arith.py` (~150-250 LOC we write): invented-glyph legend sampler, nonce-vocab pool, positional-base encode, ground-truth decode+arithmetic oracle, prompt renderer, subclass `Task`. This is net-new authoring, not adaptation of existing logic.
2. **Framework surgery for re-seedability** (outside a clean diff): either thread a `random.Random(seed)` through the new task AND avoid the global-`random` idiom the base class/harness assume, or wrap `generate_example` to `random.seed(S)` before each call and NOT go through `generation_worker` (which reseeds to `None`). Must also neutralize the `validate()` assertion (`template.py:433`) that forbids seeding. ~30-60 lines of glue/monkeypatching in our harness.
3. **Scoring glue:** override `score_answer` for strict 0/1 exact match (trivial, ~5 lines) — do NOT use `score_scalar`.
4. **Latent-rule strata:** implement ourselves inside the new task's config (`apply_difficulty`).

Estimated: **200-300 lines of net-new task code + ~50 lines of reseeding glue**, i.e. we would essentially only be reusing the `Task`/`Config`/`Entry` scaffolding and the pytest `validate()` smoke pattern — not any generator or oracle. That is major surgery, not config glue.

## Red flags (summary)
- No invented-glyph / positional-base task exists; c01 requires authoring a new generator+oracle.
- Module-global `random` + intentional `random.seed(None)` entropy reseed (`generation_worker.py:41`); framework asserts generation must NOT be seeded (`template.py:433`). Directly hostile to our RE-SEEDABLE requirement.
- Default scoring is continuous reward, not 0/1 (`utils/__init__.py:8`).
- `base_conversion` is an external `reasoning_gym` passthrough, not in-repo code.
- Import-time side effects / heavy deps (gramforge, udocker at `utils` top-level).
- Set/dict iteration and `random.shuffle` used for sampling ordering (e.g. `arithmetics.py:499,521`) compounding non-determinism.

## provisionalI1: 1
Maintained, MIT, actively used 2025-2026, and the arithmetic oracle is genuinely well-written — but for c01 specifically the target task does not exist, our "diff" is net-new authoring, and the architecture actively resists the re-seedable determinism the benchmark demands. That is "major surgery," anchor 1.
