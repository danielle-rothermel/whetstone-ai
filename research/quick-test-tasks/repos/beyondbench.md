# BeyondBench — Skeptical Code Review

- **Repo:** BeyondBench — https://github.com/ctrl-gaurav/BeyondBench
- **Review date:** undefined
- **License:** Apache-2.0 (`LICENSE`, verified header)
- **Serving candidate ids:** c17 Invented-Gate Circuit Evaluation
- **One-line verdict:** Reusable seeded generator + independent oracle for boolean-logic tasks, but the flagship `logical_operations` task has a real correctness bug (XOR silently dropped) and relies entirely on module-global RNG state — usable with edits, not turnkey.

---

## Active path (for c17, the relevant families)

Two candidate tasks map to c17 "Invented-Gate Circuit Evaluation":

1. **`beyondbench/tasks/medium/logical_operations_task.py`** — AND/OR/NOT/XOR over 3-5 vars. Closest to c17 (evaluate a boolean expression under an assignment). ~117 LOC.
2. **`beyondbench/tasks/hard/boolean_sat_task.py`** — CNF SAT, find satisfying assignment. ~1060 LOC (heavy prompt/parse machinery). Different task shape (search, not evaluate); relevant only if c17 wanted SAT.

Shared infra on the path:
- `beyondbench/core/base_task.py` (~1080 LOC; only `__init__` seed hookup + `generate_data`/`evaluate_response` contract are load-bearing for generation).
- `beyondbench/core/seed_manager.py` (~53 LOC).
- `beyondbench/parsers/logical_operations_parsing.py` (~72 LOC) — response parser (scoring glue, not generation).

**Active-path LOC we would actually read/reuse for logical_operations:** ~240 LOC
(logical_operations_task 117 + seed_manager 53 + relevant base_task `__init__`/contract ~40 + parser 72). The 1080-line base_task is mostly model-inference/retry/reporting we would NOT hit in a single-call harness.

### logical_operations generator flow
- Entry: `generate_data(self, sequence_length=1)` — `logical_operations_task.py:47`.
- Seeding: `if self.seed is not None: random.seed(self.seed)` — line 48-49 (module-global stdlib `random`).
- Sampling: per sample, `n_vars = random.randint(3,5)` (line 53), `values = {v: random.choice([True,False])...}` (line 55), expression via recursive `_make_expr` (lines 23-33) using `random.random()`/`random.choice`.
- Ground truth: `_eval_expr(expr, values)` (lines 35-45) — translates the op-string to Python and `eval`s it in a locked-down namespace (`{"__builtins__": {}}`).
- Serialization: returns list of dicts `{'expression', 'variables', 'answer'}` (lines 65-69). Plain JSON-able dict.
- Prompt: `create_prompt` (lines 72-84) asks for `\boxed{True|False}`.
- Scoring: `evaluate_response` (lines 86-107) parses via `parse_logical_answer`, exact bool match → `accuracy` 0/1.

---

## Checklist findings

### Seed plumbing — RED FLAG (module-global)
- Seed is **not threaded through an RNG instance**. Two global seed sites fire:
  - `BaseTask.__init__` → `set_all_seeds(self.seed)` (`base_task.py:135`) which does `random.seed`, `np.random.seed`, `torch.manual_seed` at the process level (`seed_manager.py:28,34,42`).
  - `generate_data` re-seeds `random.seed(self.seed)` again (`logical_operations_task.py:49`).
- All generation reads the **module-global `random`** (`import random` at top; every call site is bare `random.*`). Deterministic **per process for a fixed call order**, but any interleaving with other `random` consumers, or reordering of tasks, perturbs the stream. No `np.random`/`torch` used in logical_operations generation, so those seeds are dead weight here.
- `boolean_sat_task.py` `main()` additionally re-seeds `random.seed`/`np.random.seed` at line 987-988 — third redundant global seed site.
- **Verdict:** re-seedable (deterministic if we call `generate_data(seed=k)` fresh each strata) but via global state, not a plumbed `rng`. For our harness we would pass a distinct seed per instance and call generation in isolation, which is sufficient.

### Oracle independence — GOOD (with a caveat)
- Ground truth for logical_operations is computed by `_eval_expr` (`logical_operations_task.py:35-45`), a **separate code path** from `_make_expr` generation. It re-parses the emitted string and evaluates it. This is genuinely independent of how the expression was built — not tautological. Good for c17.
- For boolean_sat, the oracle is the **generator-constructed** `reference_solution` (assignment built alongside the clauses; `boolean_sat_task.py:460,469`). But scoring does NOT require matching that assignment — `parse_response` validates the model's assignment against the clauses via `SATSolver.evaluate_formula` (`boolean_sat_task.py:282-294,620`), which IS independent logic. So SAT oracle is also acceptable (any satisfying assignment counts).

### Correctness bug — RED FLAG (XOR silently dropped)
- `_eval_expr` (`logical_operations_task.py:38-39`) does chained replaces:
  `.replace('AND','and').replace('OR','or').replace('NOT','not').replace('XOR','^')`.
  Because `'OR'` is a substring of `'XOR'`, the second replace turns `XOR` → `Xor` **before** the `XOR`→`^` replace can fire. `eval('(A Xor B)', {"__builtins__":{}}, ...)` raises `NameError` → `_eval_expr` returns `None`.
- Generation loop (lines 57-64) retries up to 20 times and, since XOR expressions always fail, discards every XOR expression; on total failure falls back to trivial `expr='A'`.
- **Empirically verified:** with seed 42, 200 samples → **0/200 final expressions contain XOR**, 0 fallbacks. The advertised "AND/OR/NOT/XOR" task effectively only ever emits AND/OR/NOT. Also `_eval_expr` treats `^` on Python bools as int XOR — would work numerically IF it were reached, but it never is.
- Impact for c17: if we reuse this generator as-is we silently lose the XOR gate stratum. Trivial one-line fix (reorder replaces or use word-boundary regex), but it must be fixed and re-tested.

### Tests — present, runnable, but thin / do not catch the bug
- `tests/unit/test_new_medium_tasks.py:457-510` (`TestLogicalOperations`): asserts `generate_data` shape, `answer` is bool, prompt has `\boxed`, and two hand-written eval cases (`(A AND B)` only). **No XOR case in generation tests**, so the bug is uncaught. Parser round-trip tested well (`test_parser_formats`).
- `tests/test_reproducibility.py` covers `set_all_seeds` determinism and `BaseTask` seed propagation (lines 23-78, 245-297) — confirms same-seed → same `random`/`np` stream and that `__init__` calls `set_all_seeds`. Runnable with `pytest` + `unittest.mock` (uses `MagicMock` model handler; no GPU/model needed).
- boolean_sat covered only by import/instantiation smoke tests (`tests/unit/test_tasks.py:612`, `tests/integration/...` need a real model). No unit test of `SATSolver.evaluate_formula` correctness.
- Tests look runnable offline for the generator/oracle bits (mock handler pattern). I did not execute the suite.

### Global state / hidden coupling / dead code
- `base_task.py` module-level `_GPU_SAMPLER` globals (lines 21-24) — irrelevant to generation, safe to ignore.
- `BaseTask.__init__` does real work beyond seeding: creates `task_dir` on disk (`base_task.py:149-150`), calls `self.model_handler.get_model_info()` and initializes a token counter (lines 153, 146). **To instantiate the task you need a (mock) model_handler.** For a config+scoring-glue reuse we would bypass `BaseTask` entirely and call `_make_expr`/`_eval_expr` directly, or subclass with a stub handler (tests use `MagicMock`).
- `boolean_sat_task.main()` imports `from reporting import reconstruct_individual_metrics` (line 999) — a top-level module not in the package; that path is broken/dead unless run from a specific cwd. Not on our reuse path.
- `torch` deterministic flags set globally in `seed_manager` (lines 48-49) — side effect on the whole process.

---

## Adaptation-diff sketch (c17)

We would NOT run BeyondBench's engine. We lift the generator + oracle only.

- **New file (ours, outside repo):** `c17_gates.py`, ~80-120 lines.
  - Copy `_make_expr` + `_eval_expr` logic (~25 lines) from `logical_operations_task.py`, **fixing the XOR replace bug** (reorder to `NOT/XOR/AND/OR` or regex with word boundaries; ~3 changed lines).
  - Add invented/nonce gate names + randomized truth tables per c17 ("invented gates"): replace the fixed AND/OR/XOR set with N generated gate symbols each backed by a random truth table; evaluate via table lookup instead of Python `eval` (the `eval` shortcut only works for standard ops, so **~30-40 new lines** for table-driven eval — this is the real adaptation, not trivial glue).
  - Instance emitter: `generate(seed, strata) -> {"prompt", "answer", "meta"}` threading an explicit `random.Random(seed)` instead of module-global (~15 lines). This removes the global-RNG red flag on our side.
  - Reuse `create_prompt` shape (`:72-84`) and `parse_logical_answer` (`logical_operations_parsing.py`) verbatim for `\boxed{}` scoring + 0/1 exact match (~10 lines glue).
- **Files changed in repo:** none required if we vendor the ~25 relevant lines; alternatively 1 file (`logical_operations_task.py`, ~3 lines) if we fix in place and import.
- **Estimated total new glue:** ~100 lines, of which ~40 is genuine new logic (invented-gate truth tables), the rest copy/adapt.

Because c17 specifically wants *invented* gates with *randomized truth tables* and this repo hardcodes standard boolean ops evaluated via `eval`, the diff is **more than config+scoring glue** — we rewrite the gate-evaluation core. That, plus the XOR bug and global-RNG design, caps this below a 3.

---

## Red flags (summary)
1. **XOR silently dropped** (`logical_operations_task.py:38-39`) — string-replace ordering bug; verified 0/200 XOR instances generated. Advertised gate set is not actually produced.
2. **Module-global RNG** — no threaded/instance RNG; three redundant global `random.seed` sites (`base_task.py:135`, `logical_operations_task.py:49`, `boolean_sat_task.py:987`). Deterministic only under fixed call order.
3. **Instantiation coupled to a model_handler + disk I/O** (`base_task.py:149-153`) — cannot call the generator without a (mock) handler unless we vendor the functions.
4. **Thin tests** — no XOR generation test; boolean_sat oracle untested. Bug slipped through.
5. **Broken `main()` import** in boolean_sat (`from reporting import ...`, line 999) — dead/fragile, not on reuse path but signals uneven code quality.
6. **c17 gate model mismatch** — repo uses fixed ops via `eval`; invented gates with random truth tables require new eval logic.

---

## Run verification (2026-07-21)

Ran the **real** `LogicalOperationsTask.generate_data` (not a vendored copy). Instantiated the actual task class with a `MagicMock` model_handler + temp `output_dir`, so `BaseTask.__init__` → `set_all_seeds(seed)` and the task's own `random.seed(self.seed)` both fire exactly as in production.

### Environment (minimal, no torch)
```
cd /tmp/qt-repo-review/beyondbench
uv venv --python 3.14 .qtvenv
uv pip install --python .qtvenv/bin/python numpy tqdm click
```
`numpy` + `tqdm` are imported by `base_task.py`; `click` is pulled in by the package `__init__` → `cli/main.py`. `torch`/`tiktoken`/`transformers` are all optional (try/except) and NOT needed for generation — no heavyweight ML deps required. Total install ~5 s.

### Harness
`/tmp/qt-repo-review/beyondbench/qt_run_gen.py` — builds the real `LogicalOperationsTask(model_handler=MagicMock, ...)`, calls `task.generate_data()`, prints `json.dumps(data, sort_keys=True, indent=2)`. Usage: `python qt_run_gen.py <seed> <num_samples>`.

### 2. Same seed twice → byte-identical (PASS)
```
.qtvenv/bin/python qt_run_gen.py 42 50 > out_seedA_run1.json
.qtvenv/bin/python qt_run_gen.py 42 50 > out_seedA_run2.json
diff -q out_seedA_run1.json out_seedA_run2.json   # -> IDENTICAL
md5 -q out_seedA_run1.json out_seedA_run2.json     # both ef960cf373767eaa4b7b527727cccae8
```
Byte-identical, matching md5. **Deterministic re-run confirmed.**

### 3. Different seed → outputs differ (PASS)
```
.qtvenv/bin/python qt_run_gen.py 7 50 > out_seedB.json
diff out_seedA_run1.json out_seedB.json   # -> DIFFER, 385 changed lines (of 50 instances)
```
Seed 7 diverges substantially from seed 42. **Re-seeding confirmed** (the global-RNG design is deterministic here because each run is a fresh isolated process with a single fixed call order — the caveat in the seed-plumbing section holds but does not break single-process reuse).

### 4. Ground-truth hand-check (PASS, oracle independent)
Wrote a from-scratch recursive-descent boolean evaluator (regex tokenize → parse `( L OP R )` / `( NOT E )`), sharing NO code with the repo's `_eval_expr` string-replace path. Checked all 50 seed-42 instances; first 3:

| # | expression | repo answer | independent | match |
|---|---|---|---|---|
| 0 | `(NOT ((NOT (D AND C)) AND C))` (D=T,C=F) | True | True | yes |
| 1 | `(C OR C)` (C=F) | False | False | yes |
| 2 | `(E OR C)` (E=F,C=T) | True | True | yes |

**0/50 mismatches.** Ground truth is correct for the expressions actually emitted.

### XOR bug — REPRODUCED at runtime (confirms review red flag #1)
`0/50` generated instances contain `XOR`, despite `_make_expr` sampling ops from `['AND','OR','XOR']`. Direct repro of the mechanism:
```
'(A XOR B)'.replace('AND','and').replace('OR','or').replace('NOT','not').replace('XOR','^')
  -> '(A Xor B)'      # 'OR' inside 'XOR' is replaced first
eval('(A Xor B)', ...) -> SyntaxError
```
So `_eval_expr` returns `None` for every XOR expression; the 20-retry loop in `generate_data` silently discards them. The advertised AND/OR/NOT/**XOR** task emits only AND/OR/NOT in practice. Verified live, not just from code reading.

### Verdict
Generator **runs, is same-seed deterministic (byte-identical), and re-seeds** on distinct seeds; oracle is independent and correct for emitted instances. The one substantive defect — XOR silently dropped — is confirmed at runtime and remains a required fix before reuse.
