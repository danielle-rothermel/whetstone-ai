# Constructed-Notation Arithmetic & Nonstandard Number Systems — Quick-Test Candidate Survey

**Domain:** Constructed-notation arithmetic and nonstandard number systems (Theme A: Glyphscript, mixed-radix units, nonstandard positional bases, custom operators, keyed-DSL arithmetic).
**Prepared for:** whetstone-ai quick-test task selection (prompt-optimization harness: COPRO / MIPROv2 / GEPA / Codex agent).
**Date:** 2026-07-21
**Author role:** breadth-focused literature researcher (12 candidates, unranked).

---

## How to read this against the rubric

The [quick-test rubric](../design/quick-test-rubric.html) wants: single LLM call + exact deterministic scoring (no sandbox); output bounded to MC or a few words (here: a single integer / short token); default prompt scores **well below a known ceiling**; the gap closes **incrementally** through prompt improvement (not one-shot guessable); difficulty **decomposes into independent latent rules**; a **synthetic/generatable** pool with exact ground truth, hundreds of instances, contamination-resistant; **diagnostic failures** (expected X, got Y reveals which rule was missed); **few-shot demos measurably help**; prompt differences resolve on **10-20 evals at temperature 0**.

Constructed-notation arithmetic is an unusually good structural fit because ground truth is computed by a trivial reference function (base convert / apply operator table), output is a single integer, and per-instance randomization of the glyph/base/operator map gives both contamination resistance and a **known ceiling** (a prompt that fully states the map should hit ~100%). The literature below tells us *where the default-prompt floor sits* and *how the gap behaves*.

Recurring caveat across all candidates: modern reasoning-capable models with chain-of-thought can partly solve base-conversion by "translate to base-10, compute, translate back." To keep a **low floor + incremental gap** you generally must (a) use an *invented* glyph alphabet and/or *invented* operator semantics not reducible to a known base, and/or (b) bound output to a bare integer with little room for scratch work at temperature 0, and/or (c) compose multiple latent rules so single-rule prompts leave points on the table.

---

## Candidates

### 1. Base Arithmetic counterfactual task (Wu et al., "Reasoning or Reciting?")
- **What it is:** Two-digit addition performed in non-decimal bases (base-8, 9, 11, 16) as a *counterfactual* probe. Default (base-10) accuracy is near-perfect; counterfactual bases degrade substantially and consistently. Includes "comprehension checks" (successor relation in the base) to separate world-understanding from arithmetic.
- **Citation:** Wu, Qiu, Ross, Akyürek, Chen, Wang, Kim, Andreas, Kim. *Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks.* arXiv:2307.02477, 2023 (NAACL 2024). https://arxiv.org/abs/2307.02477
- **Adaptation required:** Regenerate synthetically (trivial: sample two operands, pick base, compute sum with a reference function) to dodge any contamination. Bound output to the single result integer/glyph-string. Extend digit count as a difficulty knob. To lift the floor and defeat "convert to base-10", swap standard digits for an invented glyph alphabet (per-instance randomized) so the model cannot lean on memorized base-16 tables. Add a second latent rule (e.g., a carry variant or digit permutation) for decomposition.
- **Rubric-fit notes:** Very strong core fit — exact scoring, single call, single-integer output, documented default-vs-ceiling gap, few-shot measurably narrows the gap (their Fig. 6, though residual gap persists for bases 9/11/16 — a plus, it means not one-shot solvable). Weaknesses: bases 8/16 are "easier" (pretraining frequency), so choose 9/11 or invented glyphs; strong CoT models may close much of the gap if allowed free-form reasoning, so constrain output; only ~two operands as published, so you must add the latent-rule decomposition yourself.

### 2. GSM-Symbolic / GSM-Plus symbolic-template word-problem generators
- **What it is:** Symbolic templates that parameterize GSM8K-style word problems, generating many exact-ground-truth variants; used to show fragility under surface perturbation and distractor clauses (up to ~65% drops). GSM-Plus adds targeted perturbations (numeric, arithmetic, distractor).
- **Citations:** Mirzadeh et al. *GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in LLMs.* arXiv:2410.05229, ICLR 2025. https://arxiv.org/pdf/2410.05229 · Li et al. *GSM-Plus.* ACL 2024. https://aclanthology.org/2024.acl-long.163.pdf · (related re-eval: arXiv:2605.28700).
- **Adaptation required:** Repurpose only the **template-generation methodology**, not the English narrative. Reskin templates so quantities are expressed in a nonstandard number system / mixed-radix units and the final numeric answer must be returned in that system — turning it into constructed-notation arithmetic with a single-integer answer. Strip multi-step narrative to keep cost bounded and output short.
- **Rubric-fit notes:** Best-in-class as a *generation-methodology* donor (parameterized, exact GT, hundreds+ of instances, latent difficulty knobs = clause count / distractors, published fragility curve = known floor). Weaknesses: original problems are natural-language-heavy (longer inputs, higher token cost), CoT-friendly, and the base GSM8K text is contaminated — so you must regenerate the surface entirely and constrain output. Not itself a nonstandard-notation task; you supply that layer.

### 3. Teaching Algorithmic Reasoning via In-Context Learning (multi-digit addition / parity)
- **What it is:** Algorithmic-prompting study on parity and multi-digit addition; a fine-grained "algorithmic prompt" lets models generalize addition to ~19-digit answers from ≤5-digit demos. Directly characterizes how *prompt phrasing of the algorithm* moves accuracy — the exact "incremental gap via prompt improvement" behavior the rubric wants.
- **Citation:** Zhou, Nova, Larochelle, Courville, Neyshabur, Sedghi. *Teaching Algorithmic Reasoning via In-Context Learning.* arXiv:2211.09066, 2022 (NeurIPS MATH-AI workshop). https://arxiv.org/abs/2211.09066
- **Adaptation required:** Move addition into a nonstandard base / invented digit set so the floor is low (base-10 addition is near-ceiling). Use their staged prompt ideas as the *reference "ceiling" prompt* and naive "add these" as the floor. Bound output to the final number. Latent rules = carry handling, digit-value map, base modulus — each independently promptable.
- **Rubric-fit notes:** Strong on decomposition, incremental gap, and demonstrated few-shot/prompt sensitivity. Deterministic exact scoring. Weaknesses: base-10 version is at ceiling (must swap notation); long scratchpad prompts raise token cost, so cap generation length; contamination risk on the original decimal task (mitigated by nonstandard notation + synthetic regen).

### 4. Length-generalization arithmetic suites (Position Coupling; operand length & count)
- **What it is:** Controlled synthetic addition/multiplication datasets designed to test extrapolation to longer operands/more operands than seen in-context. Exact-match scoring; accuracy is a smooth function of length — a natural difficulty dial.
- **Citations:** Cho et al. *Position Coupling: Improving Length Generalization of Arithmetic Transformers.* NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/27aa3a0e6d63db269977bb2df5607cb8-Paper-Conference.pdf · *Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count.* arXiv:2410.15787, ICLR 2025. https://arxiv.org/abs/2410.15787 · *The Lookahead Limitation: Why Multi-Operand Addition is Hard for LLMs.* arXiv:2502.19981, 2025. https://arxiv.org/pdf/2502.19981
- **Adaptation required:** These target *trained* transformers, not prompt optimization; reuse the **instance generators** and length-as-difficulty framing. Cast operands in a nonstandard/invented base to lift the floor for a large pretrained model, then use operand length + operand count as orthogonal latent-difficulty axes. Single-integer output.
- **Rubric-fit notes:** Excellent generator/difficulty-dial donor with a well-studied accuracy-vs-length curve. Weaknesses: literature is mostly about model training, not prompting — the "prompt closes the gap incrementally" claim is untested here and you must construct it; pure length generalization for frontier models in decimal is near-solved, so notation swap is mandatory to keep a floor.

### 5. InductionBench (subregular string-transformation function induction)
- **What it is:** LLMs must infer a hidden string-transformation function (ISL / L-OSL / R-OSL subregular classes) from input-output pairs and apply it — pure inductive rule-learning with dynamically generated synthetic data and exact-match scoring. Finding: even top models fail the *simplest* complexity class.
- **Citation:** Hua et al. *InductionBench: LLMs Fail in the Simplest Complexity Class.* arXiv:2502.15823, ACL 2025. https://arxiv.org/abs/2502.15823 · https://aclanthology.org/2025.acl-long.1287/
- **Adaptation required:** Reframe the transformation as an **invented arithmetic/notation operator** (e.g., a keyed digit-remapping or positional rewrite) so outputs are short and numeric-ish. Provide the rule *in the prompt* (ceiling) vs. withheld (floor), and vary complexity class as the latent-difficulty axis. Bound output to the transformed string/integer.
- **Rubric-fit notes:** Directly matches "difficulty decomposes into independent latent rules," "few-shot demos help," and "very low default floor with real headroom." Contamination-resistant by construction. Diagnostic failures (which sub-rule violated). Weaknesses: string-edit outputs can be multi-token and partial-credit-tempting — you must define a strict exact-match; some function classes may be *too* hard (floor near zero with no incremental path) so you must select complexity to get a gradient resolvable on 10-20 evals.

### 6. Cryptarithmetic / verbal-arithmetic puzzles (letters→digits under constraints)
- **What it is:** Puzzles like SEND+MORE=MONEY where each letter is a distinct digit and the arithmetic must hold; a constraint-satisfaction task with a unique-solution generator and a single set of digit assignments as the answer.
- **Citations:** (methodology) *Enumerating Cryptarithms Using Deterministic Finite Automata.* arXiv:1807.11580. https://arxiv.org/pdf/1807.11580 · classic CSP formulation widely documented.
- **Adaptation required:** Generate instances programmatically with a solver that guarantees uniqueness; ask for a specific queried digit (e.g., "what digit is M?") to bound output to a single integer and enable exact scoring. Vary number of distinct letters / base as latent difficulty. Randomize letter set per instance for contamination resistance.
- **Rubric-fit notes:** Naturally low floor (search-heavy), exact ground truth, hundreds+ generatable, contamination-resistant if randomized. Latent rules = column carries, all-different constraint, leading-digit ≠ 0. Weaknesses: solving is *combinatorial search*, which favors CoT/scratchpad and can be noisy at small n; some instances are near-unsolvable without long reasoning (bad for temp-0 short-output constraint). Better as a "hard tail" than the primary gradient unless you restrict to few-letter instances and query a single digit.

### 7. NUMCoT — numerals and units-of-measurement conversion in CoT
- **What it is:** Decomposes math word problems into numeral conversions (language↔number) and measurement conversions (unit ratios); shows LLMs consistently fail to memorize/apply conversion ratios. Mixed-radix unit systems (time, imperial, ancient units) are its bread and butter.
- **Citation:** Wang et al. *NUMCoT: Numerals and Units of Measurement in Chain-of-Thought Reasoning using LLMs.* arXiv:2406.02864, 2024. https://arxiv.org/html/2406.02864
- **Adaptation required:** Invent a **fictional mixed-radix unit system** (per-instance randomized ratios, e.g., "1 grum = 7 flits, 1 flit = 4 sprigs") and ask for a total in the smallest unit — a single integer. Ground truth = trivial mixed-radix reduction. Ratios given in prompt = ceiling; withheld/embedded in few-shot = incremental.
- **Rubric-fit notes:** Excellent thematic fit for the "mixed-radix units" cluster; conversion-ratio brittleness gives a genuine floor; latent rules = each conversion boundary + carry across mixed radices. Contamination-resistant when ratios are invented per instance. Weaknesses: published task is natural-language and CoT-heavy; you must strip narrative and bound output; real-world units (imperial/metric) are contaminated so invention is mandatory.

### 8. NumericBench — fundamental numeracy (recognition, arithmetic, comparison, logic)
- **What it is:** Broad synthetic+crawled benchmark over six numerical capabilities including arithmetic on number lists, with controllable synthetic generation and noise/long-context stressors.
- **Citation:** *Exposing Numeracy Gaps: A Benchmark to Evaluate Fundamental Numerical Abilities in LLMs (NumericBench).* arXiv:2502.11075, 2025. https://arxiv.org/pdf/2502.11075
- **Adaptation required:** Borrow the **synthetic number-list arithmetic generator**, re-express operands/operators in constructed notation, and select a single sub-capability (arithmetic) to keep output a bare integer. Add latent rules via operator remapping.
- **Rubric-fit notes:** Good generator donor with exact GT and difficulty controls (length, noise). Weaknesses: partly crawled/real data (contaminated — use only the synthetic path); breadth means most sub-tasks are off-scope; standard-notation arithmetic sub-tasks may be near-ceiling for good models without a notation swap.

### 9. Non-decimal / novel-base arithmetic sub-tasks in ICL arithmetic studies (Yuan et al.)
- **What it is:** Systematic evaluation of LLM arithmetic including non-decimal base summation over 2-4 digits; documents the "convert-to-decimal-then-back" failure mode and its error amplification.
- **Citation:** Yuan et al. *How well do Large Language Models perform in Arithmetic tasks?* arXiv:2304.02015, 2023. https://arxiv.org/pdf/2304.02015 (and the survey *Benchmarking LLMs for Math Reasoning Tasks*, arXiv:2408.10839).
- **Adaptation required:** Regenerate the non-decimal summation set synthetically; scale digits 2→4→n as a difficulty dial; substitute invented glyphs to prevent decimal-conversion shortcut and contamination. Single-integer/glyph output.
- **Rubric-fit notes:** Confirms floor behavior and a *named diagnostic failure mode* (decimal round-trip) that a reflection LM could learn to correct — strong for GEPA's mechanism. Weaknesses: as published it's near-identical in spirit to candidate 1 (overlapping evidence, not independent); CoT models can execute the round-trip correctly, so constrain output and prefer invented glyphs.

### 10. Balanced ternary / nonstandard positional systems (signed-digit arithmetic)
- **What it is:** Balanced ternary uses digits {−1, 0, +1}; arithmetic rules differ structurally from unsigned bases (symmetric negation, different carry). A clean, well-defined nonstandard positional system with trivial reference implementation.
- **Citations:** (background) *Balanced ternary* (Wikipedia, canonical rules) https://en.wikipedia.org/wiki/Balanced_ternary · recent arithmetic formalization *Tekum: Balanced Ternary Tapered Precision Real Arithmetic.* arXiv:2512.10964, 2025. https://www.arxiv.org/pdf/2512.10964
- **Adaptation required:** Build the generator yourself (no LLM benchmark exists — this is a *synthetic-family* candidate). Sample operands, compute sum/product in balanced ternary via reference code, ask for the result in balanced-ternary glyphs. Randomize the three glyph symbols per instance for contamination resistance. Latent rules = signed-digit values, carry/borrow into next place, normalization of {−1,0,+1}.
- **Rubric-fit notes:** Excellent latent-rule decomposition and a genuinely low floor (balanced ternary is rare in pretraining, so the decimal shortcut is unreliable). Exact scoring, single output. Weaknesses: no off-the-shelf dataset (must implement + validate the reference); risk that with the full rule stated in the prompt a strong model jumps near ceiling in one shot (mitigate by composing with a second rule, e.g., a glyph permutation, so the gap is multi-step).

### 11. Novel-operator / keyed-DSL in-context arithmetic (algorithmic-generalization family)
- **What it is:** Define a *made-up* binary operator via a small in-context table or rule (e.g., "a ⊕ b = (a·k + b) mod m with per-instance k, m") and require the model to apply it. Covered conceptually by algorithmic-generalization and in-context-algebra work; no single canonical benchmark, so this is a **synthetic-family** design.
- **Citations:** *In-Context Algebra.* arXiv:2512.16902, 2025. https://arxiv.org/pdf/2512.16902 · *Quantifying artificial intelligence through algorithmic generalization.* arXiv:2411.05943, 2024. https://arxiv.org/pdf/2411.05943 · (grounding on ICL algorithm teaching: arXiv:2211.09066).
- **Adaptation required:** Author the generator: per-instance random operator parameters, express operands in glyphs, ask for a single-integer result. Ceiling prompt states the operator's closed form; floor prompt gives only I/O examples; few-shot demos are the natural incremental lever. Latent rules = the operator's component transforms (scale, offset, modulus, glyph map).
- **Rubric-fit notes:** Arguably the *purest* fit for the whole rubric: per-instance randomization ⇒ strong contamination resistance and a hard floor; closed-form-in-prompt ⇒ known ~100% ceiling; component transforms ⇒ independent latent rules with diagnostic failures; examples measurably help ⇒ MIPROv2 demo axis is alive. Weaknesses: entirely bespoke (no external validation of difficulty curve — you must calibrate floor/ceiling yourself on 10-20 evals); must tune complexity so the gap is neither one-shot nor unlearnable.

### 12. List-function / rule-induction tasks (Rule 2020; MIR-Bench many-shot induction)
- **What it is:** Infer a hidden function mapping input lists → output lists (or a hidden numeric rule) from examples, then apply to a query. MIR-Bench extends to many-shot in-context inductive reasoning; List Functions (Rule 2020) is the classic source of parameterized list-transformation rules.
- **Citations:** Rule. *The Child as Hacker* (List Functions dataset), 2020 (used widely as an induction benchmark). · *MIR-Bench: Benchmarking LLM's Long-Context Intelligence via Many-Shot In-Context Inductive Reasoning.* arXiv:2502.09933, 2025. https://arxiv.org/html/2502.09933v1
- **Adaptation required:** Restrict to *numeric* list functions whose output is a single integer (e.g., "apply hidden per-element remap then sum") so scoring is exact and output bounded. Randomize the hidden rule per instance. Number-of-shots and rule composition become the MIPROv2/GEPA-relevant levers.
- **Rubric-fit notes:** Strong on few-shot-helps (it's an induction task — demos are the whole point), latent-rule decomposition, and contamination resistance via randomized rules. Weaknesses: canonical list-function outputs are *lists* (multi-token, partial-credit-prone) — you must constrain to a scalar; some rules are trivially guessable (ceiling in one shot) while others are near-impossible, so curation to a resolvable-gradient subset is required.

---

## Cross-cutting observations for selection (not a ranking)

- **Strongest "use the generator" donors:** GSM-Symbolic templating (#2), length-generalization generators (#4), NUMCoT unit systems (#7), NumericBench (#8). None is a drop-in — each needs a notation/output-bounding layer.
- **Strongest "already the right shape" candidates:** Wu base-arithmetic (#1), InductionBench (#5), and the two synthetic-family designs — balanced ternary (#10) and keyed novel-operator (#11) — because per-instance randomization directly buys the known-ceiling + contamination-resistance + low-floor combination.
- **Universal adaptations the rubric forces:** (a) invented glyphs / invented operator semantics to defeat the "convert to base-10" shortcut and memorization; (b) hard-bound output to a single integer/token so exact scoring and temp-0 resolvability hold; (c) compose ≥2 latent rules so the gap is multi-step (not one-shot guessable) and diagnostic (expected-X-got-Y reveals the missed rule); (d) synthetic regeneration for any candidate whose source text is web-published (#1, #2, #3, #7, #8, #9, #12).
- **2025-2026 research momentum (tie-breaker):** counterfactual/non-standard arithmetic and inductive rule-learning are actively studied (GSM-Symbolic ICLR 2025 and its 2026 re-evaluations, InductionBench ACL 2025, MIR-Bench 2025, In-Context Algebra 2025, length-generalization ICLR 2025). A keyed novel-operator / nonstandard-base quick test could plausibly grow into a prompt-optimization-for-notation-induction research direction.

## Selected sources
- Wu et al., Reasoning or Reciting? — https://arxiv.org/abs/2307.02477
- Mirzadeh et al., GSM-Symbolic (ICLR 2025) — https://arxiv.org/pdf/2410.05229
- Li et al., GSM-Plus (ACL 2024) — https://aclanthology.org/2024.acl-long.163.pdf
- Zhou et al., Teaching Algorithmic Reasoning via ICL — https://arxiv.org/abs/2211.09066
- Cho et al., Position Coupling (NeurIPS 2024) — https://proceedings.neurips.cc/paper_files/paper/2024/file/27aa3a0e6d63db269977bb2df5607cb8-Paper-Conference.pdf
- Arithmetic Transformers Length-Generalize (ICLR 2025) — https://arxiv.org/abs/2410.15787
- The Lookahead Limitation (2025) — https://arxiv.org/pdf/2502.19981
- Hua et al., InductionBench (ACL 2025) — https://arxiv.org/abs/2502.15823
- Enumerating Cryptarithms (DFA) — https://arxiv.org/pdf/1807.11580
- Wang et al., NUMCoT (2024) — https://arxiv.org/html/2406.02864
- NumericBench (2025) — https://arxiv.org/pdf/2502.11075
- Yuan et al., LLM arithmetic (2023) — https://arxiv.org/pdf/2304.02015
- Balanced ternary (canonical) — https://en.wikipedia.org/wiki/Balanced_ternary ; Tekum (2025) — https://www.arxiv.org/pdf/2512.10964
- In-Context Algebra (2025) — https://arxiv.org/pdf/2512.16902 ; Algorithmic generalization (2024) — https://arxiv.org/pdf/2411.05943
- MIR-Bench (2025) — https://arxiv.org/html/2502.09933v1
