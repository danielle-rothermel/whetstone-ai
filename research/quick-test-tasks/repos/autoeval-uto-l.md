# AutoEval (∀uto∃∨∧L) — Skeptical Code Review

- **Repo:** AutoEval — https://github.com/AAIR-lab/autoeval
- **Review date:** 2026-07-21
- **License:** GPL-3.0 (copyleft; `LICENSE` line 1-2 "GNU GENERAL PUBLIC LICENSE Version 3"). Datasets separately CC-BY-4.0 per README.
- **Serving candidate:** c17 Invented-Gate Circuit Evaluation
- **One-line verdict:** Reusable seeded CFG-expression generator with clean seed plumbing, but it generates propositional/FOL/regex/3-SAT *formulas* — not invented-gate circuits — and its "oracle" is a Prover9 equivalence check, not a truth-table evaluator; serving c17 requires writing our own circuit generator + truth-table oracle, so it is a structural template at best.

## Active path

Entry `dataset_generator.py:114` (`__main__`) → `get_logic_dataset`/`get_regex_dataset` → `LogicFilteredDataset.generate` (`nlfs/dataset/propositional_dataset.py:276`) → `sentence_generator.generate` (`nlfs/grammar/sentence_generator.py:46`) → `populate_dataset` (`propositional_dataset.py:328`) → `substitute_bindings` (`:243`) → serialize `logic.expr_to_str` (`nlfs/verifier/logic.py:10`) → `json.dump` (`dataset_generator.py:180`).

**Active-path LOC ≈ 1050** (of 1318 across the 9 files read: `dataset_generator.py` 184, `dataset.py` 14, `propositional_dataset.py` 462, `regex_dataset.py` 268, `sentence_generator.py` 132, `constant_vocabulary.py` 64, `predicate_vocabulary.py` 90, `dummy_vocabulary.py` 22, `logic.py` 82). Regex path is an either/or alternative to the logic path.

## Checklist findings

### Seed plumbing — mostly clean, single global reseed
- CLI `--seed` (`dataset_generator.py:152`) → `dataset.generate(seed=args.seed, ...)` (`:172-177`) → `random.seed(seed)` (`propositional_dataset.py:284`; regex `regex_dataset.py:153`). One reseed of the module-global `random` per run. If `seed is None`, falls back to `int(time.time())` (`:281-282`) — non-reproducible unless caller passes a seed (we always would).
- All sampling uses the stdlib `random` module reseeded at `generate()`: `sentence_generator.py:55,93,121` (`random.sample`), `propositional_dataset.py:101-102,348` (`random.random/choice`), `predicate_vocabulary.py:53,63,67`, `constant_vocabulary.py:51`. Because everything routes through the single seeded global, a fixed `--seed` is deterministic **given fixed PYTHONHASHSEED** (enforced at `dataset_generator.py:117-121`).
- **RED FLAG (determinism dependency):** reproducibility depends on `PYTHONHASHSEED` being set (README exports `PYTHONHASHSEED=0`). `substitute_bindings` de-dups via a Python `set` of nltk expr objects (`propositional_dataset.py:257-260`); `expr.variables()` returns a set, and dict/set iteration order feeds `generate_bindings` (`constant_vocabulary.py:53-61`, `predicate_vocabulary.py:75-84`). Hash-seed sensitivity is real; the repo mitigates by forcing PYTHONHASHSEED but this is an env-coupling, not code-level determinism.
- **RED FLAG (unthreaded/global RNG):** no `np.random`/`random.Random` instance is threaded; the module-global RNG is reseeded in-place, so concurrent generation in one process is unsafe. Regex path spins a `ProcessPoolExecutor` (`regex_dataset.py:214`) whose subprocesses run `process_sentence` (DFA sizing only, no RNG) so that is safe, but it makes wall-clock ordering of results nondeterministic — however results are keyed by `idx` assigned in the main process after `as_completed`, so output *content* can reorder run-to-run for regex. Logic path is single-threaded.
- `random.seed(1234)`/`random.seed(9869765)` appear in `name_vocabulary.py:47` and `verb_vocabulary.py:126,130` but only inside `__main__` demo blocks — **not** on the active path (fol_human vocab excluded from our use).

### Oracle independence — NOT an independent oracle for our purposes
- Ground truth stored per instance is just the generated formula string `fs` (`propositional_dataset.py:265` via `logic.expr_to_str`). The generator does **not** compute or store a truth value / truth table. There is no per-instance answer key beyond the formula itself — **tautological**: the "answer" is the instance.
- The scoring oracle lives in `nlfs/verifier/logic.py:48` `verify()`: it takes (ground-truth formula, LLM formula), converts both, and calls `Prover9.equiv` (`:71`). Equivalence is checked by an **external theorem prover**, logically independent of the generator's sampling — good for the paper's FS↔NL round-trip task, but it is a formula-equivalence check, **not** a truth-table evaluation of a circuit under an assignment.
- **For c17** (evaluate invented gates under randomized truth tables, single input assignment → 0/1): this repo has *no* component that (a) invents gates with random truth tables, (b) composes them into a circuit, or (c) evaluates a circuit on an input row. The propositional grammar (`propositional_dataset.py:125-131`) uses only fixed and/or/not — no invented gates, no per-gate truth tables. We would write the invented-gate generator and the truth-table oracle from scratch.

### Tests — none for the generator/oracle
- No pytest/unittest suite for `nlfs`. Only `code_x_glue_eval/test.py` (unrelated CodeXGLUE eval) and C/JS tests under `prover9/` and `dependencies/reg2dfa/`. **No runnable test covers sampling, seed determinism, or the oracle.** Cannot verify claims by running the authors' tests.

### Global state / coupling / dead code
- Module-global `random` reseeded in `generate()` (see above) is the main global-state coupling.
- Vocab objects carry mutable `free_count` reset via `reset_free()`/`reset_arities()` between sentences (`propositional_dataset.py:197-198,244`) — stateful but reset deterministically per sentence.
- Logic oracle requires a compiled Prover9: `_PROVER_9_ROOT = .../prover9/bin` (`logic.py:8`); `prover9/bin` is **absent** in the shallow clone (must run `scripts/install.sh` to build). Not needed if we supply our own oracle.
- Dead/broken on the alt path: `propositional_dataset.py:406` references `VerbVocabularyVocabulary` (typo, undefined) in `get_fol_human_args` — a `__main__` helper, not reached by `dataset_generator.py`. Confirms the `__main__` blocks are stale demo code.
- `alg.py` (447 LOC, imports numpy) is off our active path (batch verification driver).

## Adaptation-diff sketch (for c17)

The repo does not generate circuits, so reuse is **structural/template**, not config-only:

- **Keep/borrow (unchanged files):** `nlfs/grammar/sentence_generator.py` (CFG depth-bucketed sampler) and the `generate()`→`populate_dataset()`→JSON-serialization skeleton in `propositional_dataset.py` as a pattern. Seed plumbing pattern (`random.seed(seed)` at entry) is reusable as-is.
- **New generator (write ~150-250 LOC, outside repo):** invent N gates each with a random k-input truth table (2^k random bits, seeded), sample a random DAG/circuit over those gates + input vars, choose latent-rule strata (e.g., gate arity, circuit depth), emit nonce gate names. Cannot reuse the logic grammar — invented gates ≠ and/or/not.
- **New oracle (write ~40-80 LOC):** evaluate the circuit on a given input assignment by table lookup + topological eval → single 0/1. Independent of generation logic. Replaces `nlfs/verifier/logic.py` entirely (drop Prover9).
- **Scoring glue (write ~30 LOC):** single LLM call, 0/1 exact match against oracle output.
- **Files changed in-repo:** effectively none reused verbatim on the semantic path; we'd fork the *structure* of `propositional_dataset.py`/`sentence_generator.py`. Estimated total new code ≈ 250-400 LOC, ~all new glue outside the repo. This is more than "config + scoring glue."

## Red flags (named)
1. Reproducibility depends on external `PYTHONHASHSEED` env var (set-iteration/hash order feeds sampling), not on code-level determinism.
2. Module-global `random` reseeded in-place — unthreaded RNG; unsafe for in-process parallel generation.
3. Regex path result ordering is `as_completed`-dependent (ProcessPool); content ordering can vary run-to-run.
4. No tests for generator or oracle — claims unverifiable by running author code.
5. Logic oracle needs a compiled Prover9 binary absent from the repo (build step); irrelevant only because we'd replace it.
6. Stale/broken `__main__` demo code (`VerbVocabularyVocabulary` typo) signals uneven maintenance.
7. GPL-3.0 copyleft — forking/adapting code obligates GPL for our derivative; note for licensing.
8. Core mismatch: generates formulas, not invented-gate circuits; "ground truth" is the instance itself (equivalence-checked), not a truth-table evaluation — the exact primitive c17 needs is absent.

## provisional I1 = 1
Maintained/published 2025 (ICLR) and cleanly-written seeded generator, BUT: (a) GPL copyleft (not permissive), (b) our diff is a new generator + new oracle (~250-400 LOC), not config + scoring glue, (c) no invented-gate/truth-table primitive — the semantic core must be built, (d) no generator/oracle tests. This is a structural template requiring major surgery for c17, not a drop-in.
