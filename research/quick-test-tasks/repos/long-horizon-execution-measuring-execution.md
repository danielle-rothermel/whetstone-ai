# Repo Review: long-horizon-execution/measuring-execution

- **URL:** https://github.com/long-horizon-execution/measuring-execution
- **Review date:** undefined
- **License:** NONE (no LICENSE file; `pyproject.toml` has no license field). Official code for arXiv:2509.09677 / ICLR 2026 "The Illusion of Diminishing Returns." Paper-attached => assumable/reimplementable, not a disqualifier.
- **Serving candidate ids:** c05 (Deterministic Execution & State Tracking)
- **One-line verdict:** Reusable deterministic seeded generator + inline exact oracle for a running-sum-over-dictionary task; clean small active path but real-English (not nonce) keys capped at 101 words, hardcoded standalone generator config, and fragile module-global `random` state coupling mean our diff is config + a nonce-vocab swap + scoring glue. Provisional 2.

## Active Path

Two entry points reach the same generator:

1. **Standalone dataset dump (most relevant to us):**
   `generate_dataset_json.py:75-121` -> `generate_dict_sum_data()` -> `DictSumUtil(...)` (`dict_sum_util.py:85-107`) -> writes `dict_sum_{dict_size}.json`.
2. **In-experiment dynamic gen:**
   `main.py` -> `experiment_runner.ExperimentRunner` -> `exp.py:setup()` (`exp.py:291-308`) -> `DictSumUtil(seed=self.eval_seed)`.

Core generator/oracle chain (read end to end):
- `DictSumUtil.__init__` `dict_sum_util.py:85-107`: builds `DictCreator`, calls `create_dict`, then `generate_rollout_instances`.
- `DictCreator.create_dict` `dict_sum_util.py:46-82`: reseeds `random.seed(self.seed)` (L68), draws keys via `have_a_word(5, n)` (L73), values via `random.randint(a,b)` (L79).
- `have_a_word` `dict_sum_util.py:13-16`: opens `words_alpha.txt`, `random.sample` of length-5 words.
- `generate_rollout_instance` `dict_sum_util.py:109-131`: `random.choices(keys, k=horizon_length)` (L116), computes prefix sums inline (L119-125). **Ground truth = `output` (prefix sums), `values` = looked-up values.**
- Serialization: `generate_dataset_json.py:89-121` (JSON with `dictionary`, `ground_truth_data`, `config`).
- Load-back path: `exp.py:_load_from_local_json` `exp.py:224-289`.

**Active-path LOC (generator + oracle + serialization only):** ~230 lines
(`dict_sum_util.py:13-145` ~130; `generate_dataset_json.py:75-126` ~50; `exp.py:224-308` load/setup ~50). The rest of `dict_sum_util.py` (evaluators L148-1506, ~1360 lines) is LLM-output scoring, NOT on the generation path — we would replace it with our own 0/1 exact-match glue.

## Checklist Findings

### Seed plumbing (NAMED RED FLAGS present)
- Standalone path: `seed=42` hardcoded `generate_dataset_json.py:80` -> `DictSumUtil(seed=...)` -> threaded to `DictCreator(seed)` -> `random.seed(self.seed)` at `dict_sum_util.py:67-68`. Threaded correctly for the dict + values.
- **Module-global `random` reseed at a random call site**, not import time: `dict_sum_util.py:68`. The rollout keys (`random.choices`, L116) are NOT reseeded; they consume the module-global `random` stream left over from `create_dict`. Determinism holds only because `create_dict` and `generate_rollout_instances` run sequentially in `__init__` with no intervening `random.*` calls (verified empirically: reproducible=True). Any future code touching `random` between them breaks reproducibility.
- **Multi-instance coupling:** all `num_instances` rollouts share one continuing `random` stream (`generate_rollout_instances` L133-145 loops without reseeding). Instance N depends on N-1; changing `num_instances` changes every instance. Not per-instance seedable.
- Dynamic path: `exp.py:303` passes `self.eval_seed` (default 42, `config.py:10`). `base_experiment.py:43-44` seeds **only `np.random`**, with the `random.seed` line commented out (L45-46). Harmless here because `create_dict` reseeds `random` itself, but it is a latent trap: any generator relying on the base-class seeding would be nondeterministic.
- `np.random.seed(42)` hardcoded in bootstrap CI `dict_sum_util.py:881` (scoring only, off our path).

### Oracle independence
- **Semi-independent / borderline tautological.** Prefix sums are computed inline in the same loop that emits the input keys (`dict_sum_util.py:119-125`), so ground truth is produced BY the generator. But the computation is a trivial `cumsum` over looked-up values, and the input (`values`) is serialized alongside, so an independent re-derivation is trivial (verified: `accumulate(values) == output`). For our use we can recompute the oracle ourselves from `dictionary` + `input` and not trust the stored `output`. Low risk.

### Tests
- **No test suite** (no `tests/`, no `*test*` files besides in-file demos). `dict_sum_util.py:1272-1506` contains `test_evaluator_comparison()` and `example_usage_with_json_logging()` run under `__main__` — these are print-based demos of the *evaluators*, not the generator, with no assertions. Not runnable as CI. Generator itself has zero tests.

### Global state / hidden coupling / dead code
- Module-global `random` state is the coupling vector (see seed plumbing).
- `generate_dataset_json.py:19,52` reference `StringCrudGenerator` / `PrefixSumGenerator` that are **never imported** — those two functions are dead/broken; only `generate_dict_sum_data` (L75) works.
- `have_a_word` uses a hardcoded relative path `"words_alpha.txt"` (`dict_sum_util.py:14`); must run from repo root.
- `words_alpha.txt` contains only **101 words, all length 5**. `random.sample(all_words, k)` (L16) requires `k <= 101`; `dict_size > 101` raises `ValueError`. Hard cap on key vocabulary.
- Package import coupling: `src/experiments/__init__.py` transitively imports `dotenv`/openrouter/vllm config, so importing the generator via the package pulls heavy deps. Must import `dict_sum_util.py` by file path to isolate (verified working via importlib).
- Keys are **real English words**, not invented/nonce tokens. The task family wants nonce vocab "where applicable" — we must swap the word list.

### Reliability from read
- Generator + oracle verified running end-to-end via direct file import: deterministic, reproducible, oracle == cumsum. The ~130 active generation lines are simple and correct. Confidence high for the generation core; the surrounding repo (evaluators, runner) is large, verbose, and untested but off our path.

## Adaptation Diff Sketch

We do NOT reuse the experiment runner, LLM clients, or evaluators. We lift only the generator core.

1. **Copy** `dict_sum_util.py:13-145` (functions `have_a_word`, `DictCreator`, `DictSumUtil`) into our repo as a ~130-line module. Drop everything from L148 onward.
2. **Replace the vocabulary** (`have_a_word`, `dict_sum_util.py:13-16`): swap `words_alpha.txt` real words for our invented/nonce token generator (repo already lists `gibberish`/`random-word` deps we can mimic). ~10 lines. This also removes the 101-key cap.
3. **Fix seed plumbing** for our re-seedable strata: add an explicit `random.Random(seed)` instance threaded through `create_dict` and `generate_rollout_instance` instead of the module-global `random.seed`, and reseed per-instance if we want independent instances. ~15 lines.
4. **Latent-rule strata:** parameterize value range / dict_size / horizon per stratum via a config dict (our glue, outside repo). ~20 lines config.
5. **Scoring glue (new, outside repo):** single LLM call, parse final `<answer>`, recompute oracle as `sum` of looked-up values, 0/1 exact match. We write ~30-40 lines; do NOT reuse `NewDictSumEvaluator`.
6. **Serialization:** keep the `generate_dict_sum_data` JSON shape (`generate_dataset_json.py:75-102`) or emit our own instance schema. ~15 lines.

**Estimated:** ~130 lines lifted + ~90 lines new/modified glue. All changes are config + vocab swap + scoring — matches "config + scoring glue" but with a required vocab substitution and a seed-plumbing fix, so not a pure config diff.

## Red Flags
- Module-global `random.seed` inside a random call site (`dict_sum_util.py:68`); rollout relies on leftover global stream (`dict_sum_util.py:116`).
- Multi-instance rollouts share one un-reset stream; not per-instance seedable (`dict_sum_util.py:133-145`).
- Base-class seeds only `np.random`, `random.seed` commented out (`base_experiment.py:44-46`) — latent nondeterminism trap.
- No generator/oracle tests; only assertion-free print demos.
- Real-English keys, 101-word hard cap (`words_alpha.txt`), hardcoded relative path.
- Dead/broken sibling generators referenced but unimported (`generate_dataset_json.py:19,52`).
- Oracle computed by the generator itself (must recompute independently for trust).
- No LICENSE; `pyproject.toml` name is `scaling-laws` (stale/mismatched metadata).
