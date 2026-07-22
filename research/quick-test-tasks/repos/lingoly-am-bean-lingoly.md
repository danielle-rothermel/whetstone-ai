# LINGOLY (am-bean/lingOly) — Skeptical Code Review

- **Repo:** am-bean/lingOly
- **URL:** https://github.com/am-bean/lingOly
- **Review date:** undefined
- **License:** Dataset = CC-BY-NC-ND-4.0 (LLM-benchmark use only, no-derivatives). **Code = CC-BY-SA-4.0** (permissive with ShareAlike; LICENSE lines 5-15). Discovery note said "code license unstated" — that is WRONG; the LICENSE file explicitly dual-licenses.
- **Serving candidate:** c08 Rosetta-to-Match-Up MC Linguistic Puzzles
- **One-line verdict:** A frozen-dataset benchmark *scoring harness* over ~scraped, human-authored UKLO puzzles — there is **no instance generator, no sampling, no seed, no distractor logic, and no MC/match-up construction**; it cannot serve as a re-seedable generator without being effectively rebuilt from scratch.

## What this repo actually is

It runs a fixed set of UK Linguistics Olympiad puzzles (stored password-protected in `testing/data/benchmark.zip`, password `lingoly`) against LLMs and scores free-text answers. The "puzzles" are transcribed PDFs of human-written exams; the ground-truth answers are hand-transcribed strings embedded in the JSON. Nothing is synthesized at runtime.

## Active path (for OUR intended use = generate instance + oracle)

There is **no generator active path**. The closest analog — "produce a puzzle instance + its ground truth" — is just deserialization of a static file:

1. `benchmark_model.py:93-100` — `pyminizip.uncompress("../data/benchmark.zip", ...)` then read `test.jsonl` line-by-line into `question_sheets`. This is the entire "instance source": read a frozen file.
2. `load_questions.py:load_all_questions` (94-109) → `load_question` (41-91) → `load_questionsheet` (4-26). These format a puzzle sheet into a prompt string and collect the answer via `sp["answer"]` (line 58). The answer is a pre-stored field, not computed.
3. Oracle: `scoring.py:115-120` → `scoring_methods.compute_scores` (133-138) → `parse_str_list_score` (78-130) → `safe_exact` (33-41), which is `float(references[0] == predictions[0])` after `clean_answer` normalization (8-30).

**Active-path LOC (approx):** ~200 LOC total across `load_questions.py` (110), `scoring_methods.py` (139 — but only `clean_answer`/`safe_exact`/`parse_str_list_score`/`compute_scores` are on the exact-match path), plus ~40 LOC of file-loading glue in the benchmark/scoring scripts. All of it is I/O + string formatting + string comparison.

## Checklist findings (file:line evidence)

### Seed plumbing
**ABSENT — there is no randomness at all.** `grep -rn "random|seed|shuffle|.sample|distractor|multiple.choice"` over `testing/code` and `creation/code` (both `.py` and `.ipynb`) returns **zero hits**. No `np.random`, no `random.seed`, no shuffling of options. Instances are read in file order from `test.jsonl`. Selection is only positional slicing: `benchmark_model.py:106-110` (`questions_restart:questions_limit`). Not a red flag per se — but it means the "re-seedable generator" requirement is entirely unmet; there is nothing to seed.

### Oracle independence
**Tautological / pre-stored, not independently computed.** The correct answer is simply `sp["answer"]` read from the puzzle JSON (`load_questions.py:58`). There is no solver that derives the answer from the puzzle's latent rules — a human transcribed both puzzle and answer. `safe_exact` (`scoring_methods.py:33-41`) is a normalized string equality check. For our c08 need (latent-rule-differentiated MC distractors + oracle), this provides neither the distractors nor a rule-based oracle.

### Tests
**None.** `find . -name "*test*"` matches only the `testing/` directory name; there is no test suite, no `pytest`, no asserts on generator/oracle. `demo.ipynb`, `scoring.ipynb`, and the `creation/*.ipynb` notebooks are analysis/scraping scaffolding, not tests. Runnability of scoring depends on `evaluate`, `pyminizip`, `nltk`, and downloading HF metrics (bleu/rouge/chrf) at `scoring.py:109-111`.

### Global state / hidden coupling / dead code
- **`os.chdir` coupling:** both `benchmark_model.py:94` and `scoring.py:77` hardcode relative paths and `os.chdir("../code")`; scripts only run from `testing/code`. Fragile working-directory coupling.
- **Model registry side-loading:** `load_questions.py:84-89` opens `../data/model_list.json` on every `load_question` call to fetch chat header/footer. On the exact-match scoring path this is called with a hardcoded `model="Gemini_1.5_Pro"` (`scoring.py:85-87`) whose header/footer are then irrelevant — wasted coupling, and it forces the file to exist even when scoring.
- **Heavy dead weight for our purpose:** bleu/rouge/chrf scoring (`scoring_methods.py:44-75`, `scoring.py:122-134`) and the entire `prompt_models.py` / transformers/guidance stack (`benchmark_model.py:14-19,127-206`) are irrelevant to a single-call 0/1-exact-match task.
- **`requirements.txt`** pins ~200 packages incl. torch, bitsandbytes, guidance, geopandas — massive footprint for what is fundamentally file-read + string-compare.

## Adaptation-diff sketch (for c08)

I **cannot** write a "config + scoring glue" diff, because the core deliverable we need — a deterministic re-seedable Rosetta/match-up **instance generator with latent-rule strata and nonce vocab, plus a rule-based oracle** — **does not exist anywhere in this repo**. To use LINGOLY we would either:

- **(a) Reuse only the frozen dataset as-is:** blocked by CC-BY-NC-ND-4.0 no-derivatives on the puzzles; and it is a fixed, non-re-seedable, contamination-exposed (canary string published) set of human puzzles with free-text (not MC) answers. This yields no strata control and no nonce vocab.
- **(b) Write a generator from scratch:** we would author (i) a rule engine that samples a latent grammar/lexicon with nonce vocab, (ii) a renderer producing Rosetta-style parallel data, (iii) a match-up MC builder where each distractor perturbs exactly one latent rule, (iv) a rule-based oracle, and (v) seed plumbing throughout. That is essentially a new codebase; the only salvageable pieces are trivial: `clean_answer` normalization (`scoring_methods.py:8-30`, ~20 LOC) and the `safe_exact` idea (~5 LOC). Estimated new glue: **300-600+ LOC written outside the repo**, borrowing <30 LOC from it.

Because the adaptation cannot be reduced to config + scoring glue, this repo cannot score 3 (per protocol).

## Red flags (named)

- **No generator / no sampling / no seed** — the single most disqualifying fact for a "re-seedable instance generator" requirement.
- **Tautological oracle** — ground truth is a pre-stored transcribed string, not independently computed; no rule-based solver exists.
- **Free-text, not MC** — answers are open strings scored by string equality; there is no multiple-choice or match-up structure, and no distractor-generation logic. Our "each distractor differs by one latent rule" requirement has zero support here.
- **Dataset license is no-derivatives (CC-BY-NC-ND-4.0)** — even reusing the puzzles as strata seeds is legally constrained; non-commercial + no-derivatives.
- **Contamination-exposed frozen set** — published canary string (`README.md:53-59`) and a well-known public benchmark; not suitable as a fresh generative benchmark.
- **No tests** on any path.
- **Fragile `os.chdir` + hardcoded relative paths**; huge (~200-dep) requirements footprint for trivial logic.
- **Discovery note error:** code license is stated (CC-BY-SA-4.0), not "unstated."
