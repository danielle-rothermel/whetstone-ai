# IFEval (google-research/instruction_following_eval)

- **URL:** https://github.com/google-research/google-research/tree/master/instruction_following_eval
- **Review date:** undefined
- **License:** Apache-2.0 (repo-root `LICENSE`; per-file headers `Copyright 2026 The Google Research Authors`)
- **Serving candidate id(s):** c22 Stacked Verifiable-Constraint Micro-Generation
- **One-line verdict:** Reliable, well-tested paired generator+checker library whose constraint classes each self-sample params and check independently; strong fit for c22, but seed is module-global `random` (not threaded) so re-seeding needs a one-line wrapper, and it ships a frozen 541-row dataset rather than a batch generator driver.

## Active path

For c22 we do NOT use the shipped eval binary flow (`evaluation_main.py` -> `read_prompt_list` over the frozen `input_data.jsonl`). We use the **latent generator + oracle inside each instruction class**. The reusable active path is:

1. Entry: `instructions_registry.INSTRUCTION_DICT[instruction_id]` (`instructions_registry.py:39-77`) -> class.
2. Generation/sampling: `instruction.build_description(**kwargs)` with kwargs OMITTED -> samples params via module-global `random` and returns the NL prompt string (`instructions.py`, per class).
3. Ground-truth serialization: `instruction.get_instruction_args()` returns the sampled params dict (e.g. `instructions.py:206-209`).
4. Oracle: `instruction.check_following(response)` recomputes pass/fail from the response text via independent logic (regex/count/langdetect).
5. All-pass 0/1 and strata reporting already exist: `evaluation_lib.test_instruction_following_strict` -> `follow_all_instructions = all(is_following_list)` (`evaluation_lib.py:75-104`); `print_report` tiers by `instruction_id` prefix (`evaluation_lib.py:170-219`).

**Active-path LOC (approx):** ~450 read end-to-end. `instructions.py` = 1566 total but each constraint is a self-contained ~40-line `build_description`/`get_instruction_args`/`check_following` triple; a c22 stack uses a chosen subset. Plus `instructions_util.py` (147), `instructions_registry.py` (176), `evaluation_lib.py` (220). `instructions.py:16-91` (imports + module constants) all read.

## Checklist findings

### Seed plumbing — RED FLAG (fixable)
- All sampling uses the **module-global `random`** module: `import random` (`instructions.py:19`, `instructions_util.py:19`). Call sites e.g. `random.randint(1,_MAX_NUM_SENTENCES)` (`instructions.py:190`), `random.choice(_COMPARISON_RELATION)` (`instructions.py:193`), `random.choice(list(_LANGUAGES.keys()))` (`instructions.py:131`), `random.sample(WORD_LIST,k=num_keywords)` (`instructions_util.py:147`).
- **No `random.seed` / `np.random.seed` anywhere** (grep clean). No seed parameter is threaded into `build_description`. Determinism therefore depends entirely on the CALLER seeding `random` before each `build_description`. This is a global-state red flag but trivially controlled: our glue calls `random.seed(instance_seed)` (or holds a `random.Random(seed)` and monkeypatches) before generating each instance. No hidden secondary RNG (numpy etc.) to worry about.
- `KeywordChecker._keywords = sorted(self._keywords)` (`instructions.py:723`) — sorting keeps serialization order stable; good for exact-match.

### Oracle independence — STRONG
- Ground truth is NOT tautological. `check_following` recomputes from the response with logic disjoint from generation: `PlaceholderChecker` counts `re.findall(r"\[.*?\]", value)` (`instructions.py:275-277`); `BulletListChecker` regex-counts markdown bullets (`instructions.py:323-326`); `NumberOfSentences` uses `instructions_util.count_sentences` via nltk (`instructions.py:228-232`); `ResponseLanguageChecker` uses `langdetect.detect` (`instructions.py:158`). Generation only fixes the target parameter; the checker independently measures the actual response. This is exactly the independent-oracle property c22 wants.

### Tests — present, thorough, look runnable
- `instructions_test.py` (1290 lines) + `instructions_util_test.py` (123). `absltest` + `parameterized.named_parameters`; each test builds an instruction and asserts `check_following` on positive AND negative fixtures (e.g. `instructions_test.py:38-43, 65-96, 120-125, 182-187`). Standard `absltest.main()` structure; runnable once `pip install -r requirements.txt` (absl, langdetect, nltk, immutabledict). NOTE: not executed in this review env (deps absent), and nltk needs the `punkt` tokenizer (`instructions_util.py:135`) which requires a one-time `nltk.download('punkt')` — sentence/word counters will error otherwise.

### Global state / hidden coupling / dead code
- Global RNG (above) is the main global state.
- `INSTRUCTION_CONFLICTS` + `conflict_make` (`instructions_registry.py:79-176`) mutate dict sets to enforce symmetric conflicts — relevant for c22 stack construction (which constraints can co-occur without contradiction). Useful, not a bug.
- Dead code on/near path: registry comments out `KeySentenceChecker`, `RephraseParagraph`, `RephraseChecker`, `ConstrainedStartChecker` (`instructions_registry.py:42-65`); classes still exist in `instructions.py` but are unreachable via the registry — ignore them.
- `write_outputs` serializes via `dir(o)` reflection (`evaluation_lib.py:64-71`) — brittle but only in the eval binary we won't use.

## Adaptation-diff sketch

Our diff is config + scoring glue OUTSIDE the repo; we import the library unmodified.

- **NEW `c22_generator.py` (~80-120 lines, outside repo):** for each instance: `random.seed(seed)`; pick a constraint stack (list of `instruction_id`s, honoring `INSTRUCTION_CONFLICTS`); for each id, `cls(id).build_description()` (no kwargs -> sampled) collecting NL fragments + `get_instruction_args()`; concatenate fragments into one prompt; serialize `{seed, instruction_id_list, kwargs, prompt}` (mirrors `input_data.jsonl` schema, `evaluation_lib.InputExample`).
- **NEW `c22_score.py` (~40-60 lines):** single LLM call per instance; then reuse `evaluation_lib.test_instruction_following_strict(InputExample, {prompt:response})` to get `follow_all_instructions` as the 0/1 exact-match. Strata = `instruction_id` prefixes (reuse `print_report` logic or reimplement).
- **Config:** a small table selecting which `instruction_id`s form each latent-rule stratum; optional nonce-vocab injection by passing invented `keywords=[...]`/`forbidden_words=[...]` into `build_description` (both accept explicit lists: `instructions.py:706-722`, `1073-`), bypassing the built-in `WORD_LIST`.
- **Repo edits:** effectively ZERO to `instructions.py`/`instructions_util.py`. Only optional: a 1-line seed wrapper if we prefer `random.Random` isolation over `random.seed`.
- **Est. new glue:** ~150-200 lines outside the repo; 0 lines inside.

## Red flags (summary)
1. Module-global `random`, no seed param threaded — re-seeding requires caller discipline (`random.seed` before each instance). Fixable in glue, but it IS unthreaded seed plumbing.
2. Ships a frozen 541-row dataset + eval binary; the batch generator DRIVER for c22 does not exist and must be written (the per-constraint generator does exist and is the reusable core).
3. nltk `punkt` runtime download dependency for sentence/word counters (`instructions_util.py:135`); langdetect nondeterminism edge (`instructions.py:159-164` returns True on detection failure) — affects only language/count constraints, choose stack accordingly.
4. Some registry entries commented out (dead classes) — do not wire them.
