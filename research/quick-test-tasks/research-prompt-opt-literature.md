# Prompt-Optimizer Literature: Tasks Used, Gains, and Rubric Fit

**Date:** 2026-07-21 · whetstone-ai
**Scope:** Which evaluation tasks the prompt-optimizer papers/frameworks our harness implements
(DSPy COPRO/MIPROv2, GEPA, OPRO, APE, ProTeGi, PromptBreeder, EvoPrompt) actually use, plus
prominent 2024–2026 successors. For each candidate: which optimizers used it, reported
baseline→optimized gains (headroom / incrementality evidence), cost profile, and the adaptation
our quick-test rubric (`design/quick-test-rubric.html`) would require.

Rubric anchors used to score fit: (1) single LLM-call node + exact deterministic scoring, no
sandbox; (2) outputs bounded to MC or a few words; (3) default prompt well below a KNOWN ceiling
with an INCREMENTAL, decomposable path; (4) synthetic/generatable, hundreds of instances,
contamination-resistant; (5) diagnostic failures for reflection; (6) few-shot demos measurably
help; (7) prompt-quality differences resolvable on 10–20 evals at temperature 0.

---

## Cross-cutting observations

- **Most-used shared substrates:** BIG-Bench Hard (BBH), GSM8K, the Instruction-Induction (II)
  and BIG-Bench Instruction Induction (BBII) suites, and a small handful of text-classification
  sets (Ethos, Liar, SST-2, Subj, TREC). These recur across OPRO, APE, ProTeGi, PromptBreeder,
  EvoPrompt, and PromptWizard.
- **DSPy-lineage optimizers (COPRO, MIPROv2, GEPA)** use *multi-hop pipeline* tasks (HotpotQA,
  HoVer, PUPA, IFBench) — these are exactly the tasks our harness's optimizers demonstrably move,
  but they violate rubric criteria 1–2 (retrieval, multi-node graphs, non-exact metrics). They are
  the best *evidence of an incremental optimization path* but the *worst direct fit*; the value is
  in adapting their headroom structure, not reusing the task.
- **Headroom pattern:** the cleanest incremental gaps appear on (a) BBH symbolic sub-tasks and
  (b) II/BBII latent-rule tasks — both closest to our rubric because they are algorithmic,
  exact-match, few-word-output, and decomposable into latent rules. OPRO's "up to 50% on BBH"
  and Promptbreeder/EvoPrompt BBH gains are the strongest published headroom signals on
  exact-match, single-call tasks.
- **Contamination caveat:** every *published* benchmark above is contaminated to some degree,
  which is why the rubric wants *synthetic generation*. The literature already points the way:
  II tasks are algorithmically generatable, and BBH's symbolic sub-tasks (multi-step arithmetic,
  tracking shuffled objects, logical deduction, word sorting) are parameterized generators with
  exact ground truth — the natural template for our synthetic family.

---

## Candidate tasks

### 1. BIG-Bench Hard (BBH) symbolic sub-tasks — *adaptation → synthetic family*
- **Optimizers that used it:** OPRO, EvoPrompt, PromptBreeder, TextGrad, PromptWizard, APE (via BBII).
- **Gains / headroom:** OPRO reports up to **+50%** over human-designed prompts on BBH tasks;
  EvoPrompt reports up to **+25%** on BBH (avg ~2.5–3.5%); TextGrad ~64.9 avg with Llama3-8B.
  Large, clearly incremental gaps on exact-match tasks.
- **Cost:** single LLM call, short output (a letter/number/short string), case-insensitive
  exact match. ~250 examples/sub-task shipped, but the *generators* are parameterizable to
  hundreds+.
- **Rubric fit:** Excellent. Multi-step arithmetic, tracking shuffled objects, logical deduction,
  word sorting each decompose into independent latent rules (operator precedence, swap order,
  ordering constraints, lexicographic key) → incremental score, diagnostic failures, few-shot
  demos help.
- **Adaptation:** Reimplement 1–2 sub-tasks (e.g. tracking-shuffled-objects, multi-step-arithmetic)
  as a *seeded synthetic generator* with a length/complexity knob (number of swaps / operations)
  as the difficulty and vestigial second-metric (output-length) axis. This directly satisfies
  criteria 3, 4, 7, 8, 10, 11, 12.

### 2. Instruction Induction (II) suite — *adaptation → synthetic family*
- **Optimizers that used it:** APE (24/24 tasks), PromptBreeder, EvoPrompt, PromptWizard (via BBII), ProTeGi lineage.
- **Gains / headroom:** APE reaches human-level on 24/24 (IQM 0.765–0.810 vs human 0.749);
  BBII: APE improves/matches zero-shot on 17/21. The *point* of these tasks is that a naive
  instruction scores far below a good induced instruction — canonical incremental headroom.
- **Cost:** single call, few-word outputs, exact match. Tasks like larger-animal, second-letter,
  antonym, negation, pluralization, rhymes, cause-selection, sum, number-to-word.
- **Rubric fit:** Very strong. Each task is one latent rule; several are trivially generatable
  with exact labels (second-letter, larger-animal from a size table, sum, pluralization). Few-shot
  demos are the whole premise (rule is inferable from I/O pairs) → satisfies criterion 12 directly.
- **Adaptation:** Compose a *multi-rule* synthetic variant (e.g. "apply rule A then rule B to the
  input") so difficulty decomposes into independent latent rules (criterion 10) and partial
  prompts earn partial score (criterion 4). Generate hundreds per rule combo with RNG seed.

### 3. GSM8K (grade-school math) — *existing-benchmark (reference, not direct fit)*
- **Optimizers that used it:** OPRO, PromptBreeder, EvoPrompt, MIPROv2 (DSPy), PromptWizard, TextGrad.
- **Gains / headroom:** OPRO 71.8%→80.2% (PaLM-2-L; "+8% over human prompts"); PromptBreeder
  83.9% zero-shot (evolved prompt "SOLUTION:") vs CoT 63.8% / Plan-and-Solve 65.4%; PromptWizard 90%.
- **Cost:** single call but **long CoT output** and numeric answer; answer extraction + numeric
  compare (deterministic).
- **Rubric fit:** Partial. Scoring is deterministic (number match) which fits criterion 2, but
  outputs are NOT bounded to a few words (CoT), it is heavily contaminated, and it is a single
  latent skill (no clean rule decomposition). Good *evidence of headroom for our optimizers*, poor
  direct fit.
- **Adaptation:** Would require forcing short-form answers (answer-only) and a synthetic arithmetic
  generator — at which point it collapses into candidate #1 (multi-step arithmetic). Keep as a
  reference-gain data point, not the quick-test task.

### 4. HotpotQA (multi-hop QA) — *existing-benchmark (DSPy-lineage evidence)*
- **Optimizers that used it:** COPRO, MIPROv2, GEPA (the flagship DSPy-lineage task).
- **Gains / headroom:** GEPA paper (Qwen3-8B): baseline 42.3 → MIPROv2 55.3 → **GEPA 62.3**;
  (GPT-4.1-mini): 38.0 → 58.0 → **69.0**. Clear, monotone, optimizer-separating gaps. Note a
  separate teleprompter comparison found HotpotQA gains small vs cross-seed std — signal can be noisy.
- **Cost:** **Multi-node retrieval pipeline**, multiple LLM calls, exact-match answer scoring.
- **Rubric fit:** Poor direct fit (violates criteria 1–2: retrieval + multi-node graph), but it is
  the single strongest published proof that *our exact optimizers* (COPRO/MIPROv2/GEPA) produce an
  incremental, monotone gap on the same task. Use as headroom/incrementality evidence only.
- **Adaptation:** Not adaptable to the rubric without discarding what makes it HotpotQA. Excluded
  as a task; retained as the canonical "these optimizers close gaps incrementally" citation.

### 5. HoVer (multi-hop claim verification) — *existing-benchmark (DSPy-lineage evidence)*
- **Optimizers that used it:** MIPROv2, GEPA.
- **Gains / headroom:** GEPA (Qwen3-8B): 35.3 → MIPROv2 47.3 → **52.3**; (GPT-4.1-mini):
  46.3 → 48.3 → 51.7. Incremental, optimizer-separating.
- **Cost:** Multi-hop retrieval + query writing; scored on gold-document recall. Multi-call.
- **Rubric fit:** Poor direct fit (retrieval, multi-node). Value is as incrementality evidence for
  MIPROv2/GEPA specifically.
- **Adaptation:** None to rubric. Reference only.

### 6. PUPA (privacy-conscious delegation) — *existing-benchmark (GEPA evidence)*
- **Optimizers that used it:** MIPROv2, GEPA.
- **Gains / headroom:** GEPA (Qwen3-8B): 80.8 → 81.6 → **91.9**; (GPT-4.1-mini): 78.6 → 83.4 → **94.5**.
  Notably MIPROv2 barely moves while GEPA jumps — good optimizer-discrimination example.
- **Cost:** Composite metric (response quality + PII-leakage), model ensemble. Not exact-match.
- **Rubric fit:** Poor (composite/soft metric violates criterion 2 exactness). Reference only, but
  a nice illustration that a two-objective aggregate (quality vs leakage) resembles our vestigial
  compression axis (quality vs output-length).
- **Adaptation:** None directly; conceptual template for the two-objective / length-budget knob.

### 7. IFBench (instruction-following constraints) — *adaptation candidate*
- **Optimizers that used it:** GEPA (and MIPROv2 baseline).
- **Gains / headroom:** GEPA (Qwen3-8B): 36.9 → 36.2 → **38.6**; (GPT-4.1-mini): 47.8 → 49.2 → **52.7**.
  Small gaps — a caution that instruction-following alone may be too flat for clean ranking.
- **Cost:** Single-ish call; **programmatic constraint checking** (e.g. "answer only yes/no",
  output-format constraints) is exactly rubric-style deterministic scoring.
- **Rubric fit:** Partial-to-good. The *scoring mechanism* (verifiable output constraints, exact
  programmatic check, few-word outputs) is an excellent rubric match; the published gaps are thin.
- **Adaptation:** Build a *synthetic constraint-composition* task: each instance imposes N
  independent, programmatically-checkable output constraints (casing, length, forbidden words,
  required prefix, MC letter). Each constraint = one latent rule → incremental score, diagnostic
  "violated constraint X" failures, few-shot-teachable. Strong candidate; closely aligned with #1/#2.

### 8. Ethos (hate-speech classification) — *existing-benchmark → synthetic-family seed*
- **Optimizers that used it:** ProTeGi, PromptBreeder.
- **Gains / headroom:** ProTeGi reaches 0.95; PromptBreeder 80% (hand prompt) → **89%** (evolved).
  Clear headroom from a naive prompt.
- **Cost:** Single call, **binary label output**, exact match. Cheapest possible scoring.
- **Rubric fit:** Format fits perfectly (single call, MC/binary, exact match, temp-0 resolvable).
  Weaknesses: single latent construct (limited rule decomposition), contaminated, subjective labels.
- **Adaptation:** Use as the *shape template* for a synthetic binary/MC classifier where the label
  is set by a hidden Boolean rule over generated features (see #10). Keeps the cheap format, adds
  decomposable latent rules and contamination-resistance.

### 9. Liar (fact-checking, 6-way) — *existing-benchmark (ProTeGi evidence)*
- **Optimizers that used it:** ProTeGi.
- **Gains / headroom:** ProTeGi ~0.64 (hard task; large residual headroom, which is the point —
  low ceiling attainable makes gaps visible).
- **Cost:** Single call, short categorical output, exact match.
- **Rubric fit:** Format good (MC, exact match, single call); but real-world content = contaminated
  and no clean latent-rule decomposition or known 100% ceiling.
- **Adaptation:** Same as Ethos — borrow the multi-way categorical exact-match format for a
  synthetic rule-based classifier with a *known* ceiling.

### 10. Text-classification suite (SST-2, Subj, TREC, AG-News) — *existing-benchmark (EvoPrompt/APE evidence)*
- **Optimizers that used it:** EvoPrompt, APE, ProTeGi, PromptWizard, SAMMO.
- **Gains / headroom:** EvoPrompt reports 8–22% accuracy gains over baselines across SST-2 /
  classification / BBH; SAMMO reports >100% relative gains on some older-model classification cases.
- **Cost:** Single call, MC/short-label output, exact match — ideal cheap format.
- **Rubric fit:** Format ideal; but sentiment/topic are single latent constructs (poor rule
  decomposition) and contaminated.
- **Adaptation:** Convert to a **synthetic rule-based classifier**: generate feature-bearing
  strings and assign the label by a composition of hidden Boolean rules (e.g. "positive iff
  contains token-class A AND NOT token-class B"). Each rule adds accuracy when the prompt captures
  it → incremental, decomposable, diagnostic, few-shot-teachable, contamination-proof, known
  ceiling. This is one of the top two synthetic-family designs.

### 11. Multistep-arithmetic / word-sorting generators — *synthetic-family (from BBH)*
- **Optimizers that used it (as BBH members):** OPRO, EvoPrompt, PromptBreeder, PromptWizard.
- **Gains / headroom:** Inherits BBH's large exact-match gaps (see #1).
- **Cost:** Single call; the model can be forced to emit **only the final answer** (number / sorted
  list) → truly bounded output, exact match, no sandbox.
- **Rubric fit:** Excellent and possibly the single best fit. Difficulty knob = number of
  operations / list length. Latent rules = operator precedence, associativity, lexicographic key,
  tie-handling. Fully generatable with exact ground truth, hundreds+ of instances, contamination-
  resistant, temp-0 deterministic, few-shot demos plainly help, failures diagnostic
  ("expected 42, got 48" → precedence rule missed).
- **Adaptation:** Implement the generator + answer-only scoring + a length-budget knob as the
  vestigial second metric. Minimal engineering; maximal rubric coverage.

### 12. Symbolic "hidden-rule mapping" task (II-style, fully synthetic) — *synthetic-family*
- **Lineage:** Generalizes Instruction-Induction (#2) and MIR-Bench-style function I/O induction
  (Yan et al. 2025) — active 2025–2026 research interest in in-context rule induction.
- **Gains / headroom:** Analogous to APE/II results (naive prompt near-floor, induced-rule prompt
  near-ceiling); designed so partial-rule prompts land in between.
- **Cost:** Single call, few-word output (the transformed string / class letter), exact match.
- **Rubric fit:** Excellent. Define K independent latent transformation rules (e.g. character
  substitution, positional swap, arithmetic on indices); each instance applies a seeded subset;
  the ground-truth output is computed exactly. Prompt quality = how many rules it states → smooth
  incremental score; failures reveal exactly which rule was violated; few-shot demos let the model
  infer rules directly (criterion 12); known 100% ceiling (state all rules) and known floor
  (naive prompt).
- **Adaptation:** This is essentially the rubric's ideal task written from scratch, drawing the
  *incrementality/decomposition* structure from the II literature and the *exact-match/few-word*
  format from BBH. Top candidate alongside #11.

---

## Recommendation summary (for the quick-test designer)

- **Best direct-fit synthetic families:** #11 (multistep-arithmetic/word-sort generator), #12
  (hidden-rule mapping), and #10-as-synthetic (rule-based classifier). All three give a *known
  ceiling/floor*, *decomposable latent rules*, *exact few-word scoring*, *contamination resistance*,
  and *demo-learnability* — the full rubric.
- **Best headroom/incrementality evidence for OUR optimizers:** HotpotQA/HoVer/PUPA (COPRO,
  MIPROv2, GEPA show monotone, optimizer-separating gaps) and BBH/GSM8K (OPRO/EvoPrompt/
  PromptBreeder show large exact-match gaps). These justify the *expectation* that our optimizers
  can close an incremental gap, even though they are not themselves the quick-test task.
- **Format templates:** Ethos/Liar/SST-2 (cheap single-call MC/exact-match) and IFBench
  (programmatic constraint checking) for the scoring harness.

## Sources

- GEPA: Reflective Prompt Evolution (ICLR 2026 Oral), arXiv:2507.19457 — https://arxiv.org/abs/2507.19457 , https://arxiv.org/html/2507.19457v1
- MIPROv2 / Optimizing Instructions and Demonstrations (EMNLP 2024), arXiv:2406.11695 — https://arxiv.org/abs/2406.11695
- OPRO / Large Language Models as Optimizers (ICLR 2024), arXiv:2309.03409 — https://arxiv.org/abs/2309.03409
- APE / Large Language Models Are Human-Level Prompt Engineers (ICLR 2023), arXiv:2211.01910 — https://arxiv.org/pdf/2211.01910
- ProTeGi / Automatic Prompt Optimization with "Gradient Descent" and Beam Search (EMNLP 2023), arXiv:2305.03495 — https://aclanthology.org/2023.emnlp-main.494/
- PromptBreeder (arXiv 2023), arXiv:2309.16797 — https://arxiv.org/pdf/2309.16797
- EvoPrompt / Connecting LLMs with Evolutionary Algorithms (ICLR 2024), arXiv:2309.08532 — https://arxiv.org/abs/2309.08532
- Instruction Induction (ACL 2023), arXiv:2205.10782 — https://arxiv.org/abs/2205.10782
- BIG-Bench Hard / Challenging BIG-Bench Tasks (ACL Findings 2023) — https://aclanthology.org/2023.findings-acl.824.pdf
- PromptWizard (Microsoft) — https://microsoft.github.io/PromptWizard/
- SAMMO (Microsoft Research) — https://www.microsoft.com/en-us/research/blog/sammo-a-general-purpose-framework-for-prompt-optimization/
- A Systematic Survey of Automatic Prompt Optimization Techniques, arXiv:2502.16923 — https://arxiv.org/html/2502.16923v1
- LangProBe: a Language Programs Benchmark, arXiv:2502.20315 — https://arxiv.org/html/2502.20315
