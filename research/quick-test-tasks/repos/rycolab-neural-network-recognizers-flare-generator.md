# rycolab/neural-network-recognizers (FLaRe generator) — Skeptical Code Review

- **Repo:** rycolab/neural-network-recognizers (FLaRe generator)
- **URL:** https://github.com/rycolab/neural-network-recognizers
- **Review date:** 2026-07-21
- **License:** MIT (`LICENSE`, "Copyright (c) 2025 Rycolab")
- **Serving candidate ids:** c06 (Artificial-Grammar Violation Classification)
- **One-line verdict:** Well-engineered, deterministic, seed-threaded binary in/out-of-language string generator with a genuinely independent oracle — but it produces a BINARY accept/reject label, has NO rule-violation taxonomy or multiclass strata, so serving c06 (classify WHICH rule is violated, 4-way) requires writing substantial new labeling logic, not config glue.

## Active path

Entry point: `src/recognizers/string_sampling/sample_dataset.py::main` (line 397).
Flow: `main` → builds `random.Random(seed)` (L438) → constructs a language (hand-coded via `get_hand_coded_language` L360, or automaton via `get_automaton_language` L387) → `generate_split` (L452) → `generate_files` (L41) → per-example loop `generate_example` (L151) → `generate_positive_example` (L200) / `generate_negative_example` (L220) → `language.sample()` / `language.is_negative()` → serialization to `main.tok` / `labels.txt` / `num-edits.txt` (L69-124).

Two language tracks on the path:
- Hand-coded (e.g. `hand_picked_languages/majority.py`): `sample` and `is_negative` are hand-written (majority L35, L44-52).
- Automaton (`string_sampling/finite_automaton_language.py` + `finite_automaton_weight_pushing.py`): FSA defined in rayuela (e.g. `parity.py`, `dyck_k_m.py`), compiled OFFLINE to a normalized counting DFA by `prepare_sampler.py` and loaded from a `.pt` file. Sampling = weight-pushed random walk (`NormalizedCountingFiniteAutomaton.sample`, weight_pushing.py L159); membership = `accepts()` (L150-157).

**Active-path LOC (approx):** ~550 (sample_dataset.py 537) + ~170 (finite_automaton_language.py) + ~300 (finite_automaton_weight_pushing.py) + per-language files (majority ~70). Core generation/oracle logic to read end-to-end: ~900-1100 LOC excluding the automata algebra library (`automata/`, `rayuela/`) which the automaton track pulls in transitively.

## Checklist findings

### Seed plumbing — CLEAN (threaded)
- Single source: `generator = random.Random(args.random_seed)` (`sample_dataset.py:438`); `--random-seed` is a **required** CLI arg (L402).
- Threaded explicitly as `generator=` into `generate_files` (L96) → `generate_example` (L161) → all sampling helpers and into `language.sample(generator=...)` / random-string / perturbation / edit functions (L208, L277-357).
- numpy usage is derived deterministically: `python_to_numpy_generator` seeds `numpy.random.default_rng(generator.getrandbits(32))` (L296-297) — no global numpy seed.
- No module-global `random.seed`/`np.random.seed`/`torch.manual_seed` on the active path. The only module-level `random.seed()` calls are in `src/rayuela/cfg/random.py` (L26/76/153) — that is the CFG generator, **NOT on the FA or hand-coded path** we would use.
- Set/dict nondeterminism: `accept_states` is a `set` but only used via `in` membership (`accepts()` L150-157) — order-independent. `next_symbols` sets built by comprehension over `.tolist()`/`enumerate` (weight_pushing.py L293-299) — deterministic order. No red flag.

### Oracle independence — INDEPENDENT (not tautological)
- Hand-coded: `Majority.is_negative` calls `_parse_string` which recomputes membership by counting 1s vs 0s (`majority.py:44-52`) — logic separate from `_sample_string` (L54-60).
- Automaton: `is_negative` → `uncached_label` → `automaton.accepts(s)` (`finite_automaton_language.py:103-104`), a plain DFA walk over `transitions` dict, independent of the weight-pushed sampler.
- Cross-validated by a test: `tests/test_edit_distance.py` checks the generator's edit-distance against an independent rayuela tropical-semiring `EditDistanceExamples.edit_distance` (L60-78). Strong signal the oracle is not just echoing the generator.

### Tests — present, plausibly runnable, narrow
- `tests/`: test_edit_distance, test_counting_semiring, test_log_counting_semiring, test_finite_automaton_allsum, test_padded_batch.
- `test_edit_distance.py` exercises the real generator active path (`generate_negative_example`, `generate_positive_example`, `push_finite_automaton_weights`, `lift_finite_automaton`) for `repeat-01` and `dyck-2-3` with a fixed `random.Random(123)` (L34-78). Runnable in principle but needs the automaton `.pt` fixtures via `get_automaton` and the heavy deps below.

### Global state / coupling / dead code
- No global mutable state on the path. `FiniteAutomatonLanguage.cache` (L50) is a per-instance memo (label + edit distance) — benign.
- Heavy hidden coupling to torch: automaton track uses torch tensors, softmax, cumsum for sampling even on CPU (weight_pushing.py L106-118). Hand-coded track (majority etc.) is pure-Python and torch-free at runtime, but `sample_dataset.py` imports torch unconditionally (L12) and the automaton loader uses `torch.load` (L388).
- **Dependency friction:** `pyproject.toml` pins `rau = {git = "git@github.com:bdusell/rau.git", rev=...}` (SSH URL) plus torch/scipy/matplotlib/frozendict. `parse_device` is imported from `rau` (sample_dataset.py:14). Non-trivial environment setup.
- Two-stage pipeline for automaton languages: you must first run `save_automaton.py` then `prepare_sampler.py` to produce the `.pt` sampler before `sample_dataset.py` can run. Not a single invocation.

## Adaptation diff sketch (for c06)

**Blocking gap:** the repo emits a BINARY label (`labels.txt`: 1=in-language, 0=out; `sample_dataset.py:100`). c06 requires MULTICLASS "which of N rules did this string violate" with a 25% guess floor (=> 4 rule strata). Confirmed by grep: no `violat*`/`rule`/`strata`/`category`/multiclass concept exists anywhere in `string_sampling/` or `hand_picked_languages/`. `is_negative` returns `(bool, edit_distance)` only.

To serve c06 we would have to author, OUTSIDE the repo:
1. A grammar with ≥4 named rules and, per rule, a **rule-specific corruption operator** that violates exactly that rule while satisfying the others — none of this exists; the repo's negatives are undifferentiated random/perturbed strings (`generate_negative_example` L220), so the violated rule is not identifiable.
2. A per-rule **independent checker** to assign/verify the violated-rule label (new oracle logic; the repo's `accepts()` only gives a boolean).
3. Config + single-LLM-call + 0/1 exact-match scoring glue.

**Files we would touch/reuse:** we could reuse the seed-threading pattern, the `random.Random` sampling primitives, and possibly a hand-coded grammar as scaffolding — but the rule-stratification and per-rule violation labeling (~150-300 new LOC) is core benchmark logic we write from scratch, not a config edit. That exceeds "config + scoring glue."

## Red flags

- **Task-shape mismatch (major):** binary recognizer output, no rule-violation multiclass — the exact thing c06 needs is absent. This alone caps reuse well below "config-only."
- Negative examples are un-attributed (random string vs. perturbed positive), so a violated-rule label cannot be recovered from repo output.
- SSH git dependency `rau` + torch requirement = real install friction; automaton track needs a two-step offline `.pt` build before generation.
- Automaton track only works for languages expressible as a **deterministic** FA (`from_parts` raises on nondeterminism, weight_pushing.py L94-98) — limits grammar choices if reusing that track.
- Strengths worth keeping: clean seed threading, genuinely independent + test-cross-validated oracle, MIT license, ICLR 2025 / actively referenced.
