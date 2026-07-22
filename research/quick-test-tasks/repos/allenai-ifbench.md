# allenai/IFBench — Skeptical Reuse Review

- **Repo:** allenai/IFBench — https://github.com/allenai/IFBench
- **Review date:** 2026-07-21
- **License:** Apache-2.0 (code, `LICENSE`); data ODC-BY-1.0 (README "Licensing")
- **Serving candidate id:** c22 Stacked Verifiable-Constraint Micro-Generation
- **One-line verdict:** Reliable, well-tested deterministic *checker/oracle* library (58 constraints, 59 passing unit tests) — but there is **no seeded instance generator**; the benchmark ships as a frozen 299-row dataset, so we must write the stacking/sampling generator ourselves. Reusable as an oracle, not as a re-seedable generator. **provisionalI1 = 2.**

---

## Active path we would hit

For c22 we need: generator entry -> sampling -> ground-truth -> serialization. **Only the ground-truth half exists as code.** What the repo actually ships:

1. `run_eval.py` (85 LOC) — CLI entry. Reads a **frozen** `--input_data` jsonl + a model `--input_response_data` jsonl, runs strict/loose checkers, prints accuracy. **Evaluator, not generator.** (`run_eval.py:47-59`)
2. `evaluation_lib.py` (228 LOC) — `InputExample{key,instruction_id_list,prompt,kwargs}` dataclass; `read_prompt_list` just `json.loads` the frozen file (`evaluation_lib.py:43-54`); `test_instruction_following_strict/loose` instantiate each checker via the registry, call `build_description(**kwargs)` then `check_following(response)` (`evaluation_lib.py:84-104`).
3. `instructions_registry.py` (80 LOC) — `INSTRUCTION_DICT`: 59 `id -> Checker class` entries (`instructions_registry.py:20-79`).
4. `instructions.py` (2306 LOC) — the checker library. Each class has `build_description` (emits constraint text, fills missing kwargs with `random`) and `check_following` (**the oracle**). This is the reusable core.
5. `instructions_util.py` (1610 LOC) — tokenizers, `count_words`, `count_stopwords`, `WORD_LIST`, `generate_keywords`.
6. `generate_responses.py` (218 LOC) / `config.py` (61 LOC) — thin OpenAI-compatible chat client that POSTs each frozen prompt and writes responses. Not on any generation path.

**Active-path LOC (oracle we'd reuse):** ~2300 in `instructions.py` + ~1600 in `instructions_util.py` (only a subset per constraint actually executes) + ~80 registry + ~230 eval glue. Per single constraint the executed slice is small (one `check_following` + a few util fns). **There is no code that composes constraint stacks, selects prompts, or emits `IFBench_test.jsonl` — that pipeline is absent from the repo.**

---

## Checklist findings

### Seed plumbing — RED FLAG (unseeded module-global `random`)
- `build_description` fills unspecified kwargs with bare `random.randint/choice/sample` from the module-global `random` module. 19 call sites in `instructions.py` (`instructions.py:121,127,164,205,299,340,424,463,805,953,1035,1082,1120,1259,1389,1976,1980,2171,2175`) plus `instructions_util.generate_keywords` -> `random.sample(WORD_LIST)` (`instructions_util.py:1608-1610`).
- **No seed is threaded anywhere.** No `random.seed`/`np.random.seed` is ever called (grep `seed` across `*.py`: only `config.py:39` and `generate_responses.py:33,46,101,170`).
- The `seed` in `config.py:39-42` / `generate_responses.py:46-47` is put into the **LLM API payload only** (`payload["seed"]`), never into the instruction RNG. It does not make instance generation reproducible.
- Net: if we drive `build_description` to *generate* stacks, output is nondeterministic and non-re-seedable out of the box. We must add seed plumbing ourselves (pass explicit kwargs, or wrap with a seeded `random.Random`).

### Oracle independence — STRONG (genuinely independent)
- `check_following` computes ground truth from response text with logic independent of how the constraint string was built: e.g. prime-length set membership (`instructions.py:781-785`), stopword ratio via nltk (`instructions.py:219-226`, `instructions_util.py:1599-1606`), emoji-at-sentence-end (`instructions.py:883-899`), options exact-match (`instructions.py:830-838`).
- Ground truth reads only `self._<param>` (the target value) + the response — never re-derives the answer from generation randomness. **Not tautological.** This is the part worth reusing.

### Tests — PRESENT and RUNNABLE (verified)
- `instructions_test.py` (1189 LOC), absl `parameterized`, one+ test per checker exercising `build_description(explicit kwargs)` + `check_following` against hardcoded pass/fail messages (`instructions_test.py:36-56` etc.).
- **I ran them:** venv + `absl-py langdetect nltk immutabledict emoji syllapy unicodedata2`, nltk `punkt/punkt_tab/stopwords/tagger`, `setuptools<81` (syllapy needs `pkg_resources`; py3.14 needs the pin). Result: **`Ran 59 tests ... OK`**. Tests cover the oracle, which is exactly our reuse target.

### Global state / coupling / dead code
- Module import side effects: `instructions.py:26-32` creates `.nltk_data` dir and mutates `os.environ`/`nltk.data.path` at import. Benign but a global side effect.
- `spacy` and `langdetect` are in `requirements.txt` but **not used** in `instructions.py` (grep empty) — tests passed without spacy installed. Dead deps, not dead active-path code.
- No external CSV/JSON data files loaded by checkers (grep clean) — checkers are self-contained.
- `run_eval.py` still carries "Copyright 2025 The Google Research Authors" headers (IFEval lineage) — cosmetic.

### License
Apache-2.0, real `LICENSE` file present; README confirms code Apache-2.0, data ODC-BY-1.0. Permissive; no obstacle.

---

## Adaptation-diff sketch (what WE build, outside the repo)

The oracle is reusable almost verbatim; the generator is **net-new** (this is why it is a 2, not a 3).

**Reuse unchanged (vendor as a library):**
- `instructions.py`, `instructions_util.py`, `instructions_registry.py` — import `INSTRUCTION_DICT`, use each class's `check_following` as our 0/1 oracle. ~0 lines changed (optionally strip unused `spacy`/`langdetect` imports and the `.nltk_data` env mutation).

**New glue we write (~120-200 LOC, our repo):**
1. **Seeded generator** (~60-100 LOC): given a seed + a chosen `instruction_id` subset (latent-rule strata), instantiate checkers, call `build_description` with a `random.Random(seed)`-derived explicit kwargs dict (pass values in rather than relying on the unseeded internal `random`), and assemble the composite prompt string (concatenate per-constraint descriptions onto a base/nonce prompt). This is the sampling + composition layer the repo lacks.
2. **Instance serializer** (~20 LOC): emit our own record shape (or the `{key,prompt,instruction_id_list,kwargs}` shape it already consumes).
3. **Scoring glue** (~30 LOC): single LLM call, then `all(check_following(resp) for each constraint)` -> strict 0/1 exact match, reusing `evaluation_lib.test_instruction_following_strict` logic (`evaluation_lib.py:75-104`) nearly verbatim.
4. **Nonce/invented-vocab strata:** feed our own nonce keyword lists as explicit `keyword*` kwargs (constraints like `count:keywords_multiple` already accept keyword kwargs — see `data/IFBench_test.jsonl` rows) instead of `WORD_LIST` sampling.

**Files changed in-repo: ~0 (vendor read-only). New code outside repo: ~120-200 LOC.** The diff is *more than* "config + scoring glue" because we author the seeded generator/sampler the repo never provided — hence not a 3.

---

## Red flags (named)
1. **Unseeded module-global `random`** in every `build_description` (19 sites + `generate_keywords`); no `random.seed` anywhere. Generation is nondeterministic/non-re-seedable as shipped. Mitigated by passing explicit kwargs from our own seeded RNG.
2. **No generator/sampler in the repo** — benchmark is a frozen 299-row `data/IFBench_test.jsonl`; kwargs and prompt text are pre-baked. The "seeded-generator" discovery tag overstates what ships; only the oracle + a per-field kwarg randomizer exist.
3. **`config.seed` is a decoy** — wired only to the LLM API payload, not to instance generation. Easy to mistake for reproducibility plumbing.
4. Minor: unused `spacy`/`langdetect` deps; import-time env/dir side effects; syllapy needs `pkg_resources` (`setuptools<81` on py3.12+); IFEval Google copyright headers linger.

**provisionalI1 = 2** — maintained (PR #25 2026-05, tests green), permissive license, actively used 2025-2026, oracle well-written and verified runnable; BUT our diff exceeds config+scoring glue (we must author the seeded stacking generator) and seed plumbing is absent, so it fails the "diff is config+scoring glue" and "works as a re-seedable generator as claimed" anchors for a 3.
