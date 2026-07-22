# Repo Review: google/BIG-bench — multistep_arithmetic

- **URL:** https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/multistep_arithmetic
- **Review date:** undefined
- **License:** Apache-2.0 (root `LICENSE`; per-file header `task.py:1-12`)
- **Serving candidate:** c03 Keyed Custom-Operator Expression Evaluation
- **One-line verdict:** Tiny, clean, deterministic seeded generator over parenthesized arithmetic strings — but the oracle is `eval()` of the generated string (tautological w.r.t. standard operators) and the "keyed custom-operator" family needs us to replace both the operator vocab AND the oracle, so it is config + real scoring/oracle glue, not just config.

## Active path

Entire active path is one self-contained file: `bigbench/benchmark_tasks/multistep_arithmetic/task.py` (172 lines).

Flow (all in `MultistepArithmeticTask`):
- Entry / config: `__init__` `task.py:28-61` — sets `self.rng = np.random.RandomState(seed)` (`:54`).
- Eval driver: `evaluate_model` `task.py:114-171`.
- Sampling: `generate_string` `:111-112` -> `generate_subparenthesis` `:92-109` (recursive) -> `generate_content` `:78-90`. Random calls: `rng.choice(self.numbers)` / `rng.choice(self.operations)` at `:84-89`, `:103`, `:107`.
- Ground-truth: `targets.append(str(eval(code)))` `:143`, where `code = input` (`:142`) is the exact generated string.
- Serialization: `problem = input + " = "` `:140`; inputs/targets collected `:141,:143`.
- Scoring: single model call `model.generate_text(inputs, output_regex=r"[-+]?\d+")` `:145`; exact string compare `if result == target` `:148`; `score/trials` into `ScoreData` `:158-170`.

**Active-path LOC:** ~145 (of 172; remainder is docstring/metadata). Base classes `TaskMetadata`/`ScoreData`/`Task` in `bigbench/api/task.py:25-171` are trivial dataclasses/ABC — no logic on our path.

## Checklist findings

### Seed plumbing — MOSTLY CLEAN, one threading wart
- Seed enters via `__init__(seed=42)` and is bound once to `self.rng = np.random.RandomState(seed)` `task.py:54`. No module-global `np.random.seed`/`random.seed` at import; no reliance on set/dict iteration order (iteration is over `itertools.product(depth_level_list, lengths)` `:131`, deterministic).
- Re-seedable: `evaluate_model(..., random_seed=None)` `:114-116` rebinds `self.rng = np.random.RandomState(random_seed)` when truthy. Good for our re-seed requirement.
- **Wart (red flag, minor):** `generate_content(self, length, rng)` takes an explicit `rng` param `:78`, but `generate_subparenthesis` calls it as `self.generate_content(length, self.rng)` `:94` — passing the instance rng. The recursive branch uses `self.rng` directly (`:103,:107`). So the `rng` parameter is a vestigial pass-through, not an independent stream. All randomness ultimately flows from `self.rng`; determinism holds, but the signature is misleading.
- **Wart (minor):** `random_seed=0` would be treated as "no seed" because of `if random_seed:` `:115` (falsy). Same for `max_examples=0` `:117`. Cosmetic for our use.

### Oracle independence — TAUTOLOGICAL (as-is)
- Ground truth is `str(eval(code))` `:143` where `code` is the generator's own output string. The oracle is Python's expression evaluator applied to the generated text. There is **no** independent reference computation — correctness of the target is defined by Python operator semantics/precedence, which is exactly what the string was built to be parsed as.
- For standard `+ - *` this is fine (Python is a trusted oracle). **For c03 (invented infix operators, hidden precedence/wrap conventions) `eval()` is NOT a valid oracle** — Python does not know your operator semantics. We must write our own evaluator. So oracle independence is effectively absent for the target family.
- Note `README` claims division is included, but default `operations=["+","-","*"]` `:33` omits `/` (and `eval` would do true float division anyway). Minor doc/code mismatch.

### Tests — NONE for this generator/oracle
- No test file in the task dir (`find ... -name "*test*"` empty).
- No reference to `multistep_arithmetic`/`MultistepArithmeticTask` in `bigbench/api/` tests. `bigbench/api/test_tasks.py` is a generic harness and does not target this generator or the `eval` oracle. Effectively **untested** for our purposes; we bring our own tests.

### Global state / coupling / dead code
- No module-global mutable state; randomness confined to `self.rng`.
- Coupling: imports `bigbench.api.task` for `Task`/`TaskMetadata`/`ScoreData` and `model.generate_text` interface — thin, replaceable. We would not use the BIG-bench model interface.
- Dead/vestigial code on path: the `rng` param of `generate_content` (see seed wart); `correct` variable set but only used for verbose print `:138,:150-151`.
- **`eval()` on generated-but-untrusted-looking string** `:143` — safe here since we generate it, but any adaptation that lets external strings reach this line is an injection risk. Flag when adapting.

## Adaptation-diff sketch

We would NOT edit repo files in place; we lift the ~40 lines of generation logic and rewrite the oracle. Estimated new/changed:
- **Reuse (copy, ~40 LOC):** `generate_content`/`generate_subparenthesis`/`generate_string` recursion + `RandomState(seed)` plumbing from `task.py:78-112`. Keeps deterministic re-seedable structure.
- **Replace vocab (config, ~10 LOC):** swap `operations`/`numbers` for per-instance nonce operator symbols + a sampled symbol table (latent-rule strata: precedence/wrap conventions). New glue outside repo.
- **Write new oracle (~40-70 LOC, NEW):** a real expression evaluator honoring the per-instance operator semantics and wrap/precedence conventions — replaces `eval()` at `:143`. This is the load-bearing new code; cannot be config-only.
- **Scoring glue (~20 LOC):** single LLM call + `output_regex`/parse + 0/1 exact-match; drop `ScoreData`/BIG-bench `model` interface. Reuse the `output_regex=r"[-+]?\d+"` idea `:145`.
- **Tests (~40 LOC, NEW):** determinism-by-seed + oracle-vs-generator agreement on standard ops as a sanity anchor.

Net: generation skeleton is genuinely reusable; oracle + vocab keying are new code. So our diff is **config + a nontrivial custom oracle**, not config + scoring glue alone.

## Red flags (named)
1. Oracle is `eval()` of the generator's own string (`:143`) — tautological; invalid for c03's invented operators. Must reimplement oracle.
2. Vestigial `rng` parameter in `generate_content` (`:78`) not actually threaded independently — misleading but deterministic.
3. `if random_seed:`/`if max_examples:` treat `0` as unset (`:115,:117`).
4. `eval()` on a string (`:143`) — injection surface if adaptation ever admits external input.
5. No task-specific tests for generator or oracle.
6. README/code mismatch on division (`README` says division; default ops omit `/`).
