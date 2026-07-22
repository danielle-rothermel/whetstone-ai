# Research: Synthetic Rule-Induction & Compositional Generalization for Prompt/ICL Optimization

**Domain owner:** breadth-focused literature scan
**Date:** 2026-07-21
**Purpose:** Find existing tasks / benchmarks / dataset-generators that could be USED or ADAPTED as the *quick-test* task for the whetstone-ai prompt-optimizer harness (COPRO, MIPROv2, GEPA, Codex agent), scored against `design/quick-test-rubric.html`.

## Framing: what the rubric actually demands

The rubric is the authority. The candidates below are judged against these load-bearing criteria:

- **C1/C2/C3 (cheap):** single LLM call, exact deterministic programmatic scoring (no sandbox), output bounded to multiple-choice or a few words / short token sequence.
- **C4 (incremental gap):** default prompt scores *well below* a known ceiling, and the gap closes in *steps* — the best prompt is not one-shot guessable by a competent prompt engineer.
- **C5 (resolvable at n=10-20, temp 0):** effect size of a ≥10pt prompt change must exceed residual noise on a tiny eval.
- **C7/C8 (synthetic, hundreds of instances, contamination-resistant):** program-generated with exact ground truth.
- **C9 (known ceiling/floor + reference prompt).**
- **C10 (difficulty decomposes into independent latent rules; which rules a prompt found is inspectable).**
- **C11 (failures diagnostic for a reflection LM).**
- **C12 (few-shot demos measurably help).**
- **C13 (failure paths exercised — parse errors, empty gen).**

**Recurring adaptation pattern across every candidate below:** because almost all published benchmarks are *fixed corpora*, contamination (C8) and one-shot-guessability (C4) are the two dimensions that most often need engineering. The winning move for most is: (a) keep the task *generator* but discard the published split, (b) re-key / re-seed the latent rules per run, (c) bound the output to a single MC letter or one short token, and (d) tune rule count so the naive-prompt floor is well below the write-the-rules-out ceiling.

**Tie-breaker note (2025-2026 research heat, lower priority):** in-context rule induction is unusually active right now — InductionBench (2025), ICL Ciphers (2025), HERO'S JOURNEY (2026), SCFG in-context transduction (2026), LingOly / Rosetta-to-Match-Up (2024-2026). A quick-test built on a synthetic rule-induction generator has a natural path to becoming its own research artifact.

---

## Candidates (breadth, NOT ranked)

### 1. SCAN / MiniSCAN (compositional command → action grammar)
- **What:** Lake & Baroni's SCAN maps synthetic navigation commands ("jump twice and walk left") to action sequences via ~20 phrase-structure grammar rules over ~20 words. MiniSCAN / "few-shot rule learning" variants shrink it to a handful of invented primitives learned purely from in-context examples.
- **Sources:** Lake & Baroni, ICML 2018 (arXiv:1711.00350); least-to-most prompting solves SCAN few-shot (arXiv:2205.10625, 2022); Lake, "human few-shot learning of compositional instructions" (MiniSCAN).
- **Adaptation for rubric:** SCAN outputs are long action sequences — violates C2/C3. Convert to a *single-step* variant: give k demonstration (command, action) pairs with an INVENTED grammar, then ask for only the FIRST action token or a single MC among candidate action sequences. Regenerate grammar per run (dodges C8; SCAN itself is heavily contaminated). Independent rules = per-primitive mappings + modifiers ("twice", "thrice", "left") → clean C10 decomposition.
- **Rubric fit:** Strong on C7/C10/C12 (built for compositional demos). Weaknesses: canonical SCAN is memorized (C8) and long-output (C2/C3) — both fixed only by the MiniSCAN-style rewrite. As-is it fails C2/C3.

### 2. gSCAN / ReaSCAN (grounded grid-world compositional instructions)
- **What:** gSCAN grounds SCAN commands in a grid world; tests 8 compositional splits (novel adjective-noun, verb-adverb, target locations). ReaSCAN adds relational reasoning.
- **Sources:** Ruis et al., NeurIPS 2020 (arXiv:2003.05161); ReaSCAN (arXiv:2109.08994, 2021); repo github.com/LauraRuis/groundedSCAN.
- **Adaptation:** Full gSCAN wants an action sequence over a rendered grid — too heavy (C1/C2/C3 fail). Usable only if collapsed to a *textual* single-question form: describe grid in text, ask one MC ("which cell is the referent?"). The synthetic generator + 8 named generalization types is the asset (C10). Heavy adaptation.
- **Rubric fit:** Excellent latent-rule decomposition (C10) and generator control (C7). But grid rendering + sequence output make it the most expensive to bound; likely not worth the adaptation cost versus lighter candidates.

### 3. COGS / SLOG (compositional semantic parsing)
- **What:** COGS maps procedurally-generated English sentences to logical forms; 21 generalization categories (18 lexical, 3 structural). SLOG extends structural generalization.
- **Sources:** Kim & Linzen, EMNLP 2020 (COGS); SLOG (arXiv:2604.26157 / earlier 2023 EMNLP).
- **Adaptation:** Output is a logical form (structured string) — partial-credit temptation violates C2. Bound to exact-match of a short canonicalized logical form, or convert to MC over candidate parses. Generator dodges C8 if re-run. Latent rules = lexical roles + structural rules (C10).
- **Rubric fit:** Good C7/C10; the 18-vs-3 lexical/structural split is a natural incremental-difficulty knob (C4). Weakness: exact-match on logical forms is brittle (format-violation heavy — arguably good for C13 but noisy for C5). Long-ish outputs pressure C3.

### 4. PCFG SET (Hupkes — string-edit transduction with 5 named generalization axes)
- **What:** Sequences from a probabilistic CFG must be translated to their "meaning" by applying string-edit operators (copy, reverse, shift, echo, append...). Five explicit test axes: systematicity, productivity, substitutivity, localism, overgeneralization.
- **Sources:** Hupkes et al., JAIR 2020, "Compositionality Decomposed" (arXiv:1908.08351).
- **Adaptation:** Fully synthetic generator already exists → strong C7/C8. Output is a sequence; bound to first-token or fixed-short-length output, or MC. Each operator is an independent latent rule (C10) and the five axes give graded difficulty (C4). Re-seed the operator inventory per run.
- **Rubric fit:** One of the best structural fits: explicitly designed so difficulty decomposes into independent, nameable rules (C10) with an incremental path (C4), and it is generator-native (C7/C8). Weakness: output-length bounding still required (C3); operators like "reverse" produce longer outputs.

### 5. InductionBench (subregular function/transducer induction) — 2025
- **What:** LLMs must INDUCE the underlying rule of a string-to-string function drawn from the simplest subregular complexity classes (ISL/OSL/etc.), from input-output demonstrations. Reports that even o1/o3 fail the simplest classes.
- **Sources:** Hua et al., "InductionBench: LLMs Fail in the Simplest Complexity Class" (arXiv:2502.15823, 2025); code on GitHub.
- **Adaptation:** Very close to ideal. It is *native rule-induction from demos* (C12), synthetic with exact ground truth (C7/C8), and complexity-classed (C4/C10). To fit: bound output to applying the induced rule to ONE held-out input (few-token exact match) rather than emitting the rule; re-generate keys per run. The complexity hierarchy gives a principled floor/ceiling (C9).
- **Rubric fit:** Among the strongest overall (C4/C7/C8/C10/C11/C12). Weakness: "simplest classes still fail" means the ceiling may be hard to reach even with a great prompt — must verify a reference prompt can hit ≈100% (C9) on the *easiest* classes, else C4's headroom collapses to "nobody succeeds."

### 6. List Functions (Rule et al. — program/concept induction over integer lists)
- **What:** 250 human-authored list→list functions (map, filter, sort, dedupe, arithmetic). A gold standard for few-shot concept induction with human baselines. Present in BIG-Bench as `list_functions`.
- **Sources:** Rule PhD thesis (2020); Alet et al., "A large-scale benchmark for few-shot program induction and synthesis," ICML 2021; BIG-bench `list_functions`.
- **Adaptation:** Give k input→output list demos, ask for the output list on ONE query (short exact-match) — no sandbox needed, the label is precomputed (C2). Regenerate the *parameters* of each function family per run for C8 (the 250 concepts are published → contamination risk). Concept families are the independent latent rules (C10); Boolean/algorithmic complexity grades difficulty (C4).
- **Rubric fit:** Strong C7/C10/C11/C12 and clean exact-match scoring (C2). Weakness: some functions produce long list outputs (C3) — restrict to length-bounded output families; published concepts need re-parameterization for C8.

### 7. In-Context Boolean Concept Learning (complexity-graded) — 2024
- **What:** LLMs learn a hidden Boolean concept over feature vectors from labeled in-context examples; task accuracy correlates with the concept's Boolean complexity, mirroring the human simplicity bias.
- **Sources:** "Minimization of Boolean Complexity in In-Context Concept Learning" (arXiv:2412.02823, 2024).
- **Adaptation:** Almost turnkey for the rubric. Output is a single YES/NO or MC label (C2/C3 satisfied natively). Fully synthetic feature vectors + Boolean formula → exact ground truth, unlimited instances, contamination-proof (C7/C8). Boolean complexity is a direct, dial-able floor→ceiling knob (C4/C9); each literal/clause is an independent latent rule (C10); few-shot is the entire mechanism (C12).
- **Rubric fit:** Possibly the tightest fit to C1-C5 of any candidate (binary output, synthetic, complexity-graded, demo-driven). Weakness: with only a binary label per item, C5 noise is high per-instance — needs enough query items per eval to resolve; failures are less individually diagnostic than sequence tasks (C11 weaker — "wrong label" reveals less than "wrong token in position 3").

### 8. ICL Ciphers / Rashid (substitution-cipher in-context learning) — 2025-2026
- **What:** Re-map the token/vocabulary of a task via a bijective substitution cipher so the model must LEARN the mapping in-context rather than recall pretraining knowledge. Explicitly built to quantify genuine ICL and to be contamination-proof.
- **Sources:** "ICL CIPHERS: Quantifying Learning in In-Context Learning via Substitution Ciphers" (arXiv:2504.19395, 2025); "Rashid: A Cipher-Based Framework for Exploring In-Context Language Learning" (arXiv:2603.22497, 2026).
- **Adaptation:** The cipher key IS the latent rule; regenerate per run → C8 is solved by construction (its headline feature). Bound to decoding ONE short ciphered token/answer, exact match. Compose multiple independent sub-ciphers (e.g., per-position or per-class keys) for C10 decomposition and a graded floor→ceiling (C4).
- **Rubric fit:** Best-in-class on C8 (contamination-resistance is the paper's raison d'être) and strong on C12. Weakness: a single monoalphabetic cipher IS one-shot solvable once a prompt engineer says "it's a substitution cipher" (fails C4) — needs *composed/keyed* ciphers or hidden structure so no single instruction unlocks it.

### 9. HERO'S JOURNEY (complex rule induction in text games) — 2026
- **What:** Agents infer hidden rules from demonstrations in goal-directed episodic text games; 8 tasks across attribute-induction and procedural-induction families, each with 4 structural rule forms, controllable lexical grounding, and identifiability conditions. Finds current LLM rule induction limited and uneven; procedural induction is the hard open gap.
- **Sources:** Zheng, Misra, Beaver, Li, "HERO'S JOURNEY: Testing Complex Rule Induction with Text Games" (arXiv:2606.02556, June 2026).
- **Adaptation:** Multi-step execution violates C1/C2 as-is. Extract the *single-decision* form: show demos of the hidden rule, ask for ONE next action / MC. The generator's controllable lexical grounding is exactly the "swap surface to dodge contamination" knob (C8); attribute vs. procedural families + 4 structural forms give C10/C4 structure. The paper reports "surface semantics has minimal effect" — useful evidence the latent-rule signal survives lexical randomization.
- **Rubric fit:** Very strong on C7/C8/C10/C11 and freshest research heat. Weakness: procedural family may be too hard (ceiling unreachable → C9/C4 risk), and reducing to a single decision loses some of what the benchmark measures. Newest → tooling less mature.

### 10. Re-ARC / 1D-ARC / MiniARC / ARC-GEN (procedural ARC-style grid rule induction)
- **What:** ARC = infer a grid transformation from 2-3 demos, apply to a test grid. Re-ARC (Hodel 2024) is a *procedural generator* with configurable difficulty for all 400 training tasks; 1D-ARC linearizes to arrays (18 categories, 901 tasks); MiniARC is 5x5; ARC-GEN (2025) is a mimetic procedural generator.
- **Sources:** Chollet ARC (2019, arXiv:1911.01547); Hodel, "Addressing ARC via Procedural Example Generation" / re-arc (arXiv:2404.07353, 2024; github.com/michaelhodel/re-arc); 1D-ARC (arXiv:2305.18354); ARC-GEN (arXiv:2511.00162, 2025).
- **Adaptation:** Full 2D-grid output is verbose and hard to bound (C3) and exact-match on grids is format-fragile. Prefer 1D-ARC linearized to short arrays, output bounded to the transformed array (short exact-match) or MC over candidate outputs. Re-ARC's difficulty dial + infinite generation directly serve C4/C7/C8. Each transformation primitive is an independent latent rule (C10).
- **Rubric fit:** Excellent C7/C8 (Re-ARC is a generator with difficulty control) and C10. Weaknesses: grid serialization inflates tokens (C3); ARC is famously hard so ceiling reachability (C9) is a real risk unless you restrict to easy 1D transforms; grid exact-match causes many format failures (good for C13, noisy for C5).

### 11. BIG-Bench Hard — symbolic subset (Boolean expressions, logical deduction, tracking shuffled objects, Dyck languages, web-of-lies)
- **What:** 23-27 hard BBH tasks; several are *algorithmically generatable* with MC / short answers: `boolean_expressions`, `logical_deduction`, `dyck_languages`, `tracking_shuffled_objects`, `web_of_lies`, `multistep_arithmetic`.
- **Sources:** Suzgun et al., "Challenging BIG-Bench Tasks and Whether CoT Can Solve Them" (arXiv:2210.09261, 2022); BIG-bench repo.
- **Adaptation:** Use only the synthetic subset and *regenerate* instances from the underlying generators (published BBH is contaminated → C8). Bound to MC letter / single token (many already are). Rules decompose: e.g., logical_deduction = ordering constraints; tracking_shuffled_objects = swap-tracking (C10). CoT vs. direct is a known incremental-headroom lever (C4).
- **Rubric fit:** Strong C1/C2/C3 (many are MC already), good C4 (default-vs-CoT gap is well documented), decent C10. Weakness: heavy contamination as published (C8) — MUST regenerate; latent-rule decomposition is per-task and less clean than the grammar/cipher families.

### 12. LingOly / Rosetta-to-Match-Up (Rosetta-stone linguistic puzzles) — 2024-2026
- **What:** Olympiad linguistic puzzles: from paired examples in a low-resource/invented language, induce morphology/syntax rules and translate. LingOly (1,133 UKLO puzzles, NeurIPS 2024) reports SOTA ~21.7% exact-match. Rosetta-to-Match-Up (2026) pairs human + LLM benchmarks.
- **Sources:** Bean et al., "LINGOLY" (arXiv:2406.06196, NeurIPS 2024); "From Rosetta to Match-Up" (arXiv:2605.13408, 2026); "Can LLMs Solve and Generate Linguistic Olympiad Puzzles?" (arXiv:2509.21820, EMNLP 2025); LingBench++ (arXiv:2507.16809, 2025).
- **Adaptation:** Published puzzles are fixed and partly contaminated (C8). Use a *synthetic conlang generator* (morphological rules + lexicon) instead of UKLO corpora, ask to translate ONE short word/phrase (short exact-match) or MC. Morpheme rules = independent latent rules (C10); rule count grades difficulty (C4). The 2025 "generate puzzles" work and ConlangCrafter (arXiv:2508.06094) provide generation machinery.
- **Rubric fit:** Very strong C4 (low SOTA = huge headroom), C10 (morphology decomposes cleanly), C12 (Rosetta format IS few-shot). Fresh research heat. Weakness: hand-built puzzles fail C7/C8 — the whole value depends on building/adopting a *synthetic* conlang generator, which is real engineering; exact-match on morphology is noisy (C5).

---

## Bonus / adjacent leads (not full candidates)

- **Synchronous CFG in-context transduction** (arXiv:2604.07320, 2026): evaluate ICL by translating between two formal languages defined by a synchronous CFG — a clean generator for compositional, contamination-proof few-shot transduction. Strong latent adaptation base overlapping #4/#5.
- **"Understanding the Relationship between ICL and Compositional Generalization"** (arXiv:2403.11834, 2024): methodological grounding for why demos help compositional tasks (supports C12 design choices).
- **Grammar prompting for DSL generation** (arXiv:2305.19234): BNF-constrained output — a technique to bound outputs (helps C2/C3/C13) rather than a task itself.
- **ConlangCrafter** (arXiv:2508.06094, 2025) and **"Creating ConLangs to Probe Metalinguistic Knowledge"** (arXiv:2510.07591, 2025): generators to make #12 synthetic.

---

## Cross-cutting recommendations for the harness designer

1. **Most rubric-aligned families are generator-native rule-induction:** #4 PCFG SET, #5 InductionBench, #6 List Functions, #7 Boolean Concept Learning, #8 ICL Ciphers. These minimize adaptation because they already have exact ground truth + demo format + a complexity dial (C4/C7/C8/C10/C12). Prioritize these for a first prototype.
2. **The two universal adaptations** are (a) bound output to one MC letter / one short token for C2/C3, and (b) re-seed the latent key per run for C8. Almost every candidate needs both.
3. **Watch the ceiling-reachability trap (C9):** the freshest/hardest tasks (InductionBench, HERO'S JOURNEY procedural, full ARC) may have *no* prompt that reaches ~100%, which kills C4's "close the gap" story. Restrict to the easy end of each complexity hierarchy and *verify a reference prompt hits ~100%* before adopting.
4. **The one-shot-guessability trap (C4):** single monoalphabetic ciphers, single-rule concepts, and "just use chain-of-thought" tasks are one-shot solvable by a competent prompt engineer. Use *composed* independent rules (multiple ciphers/operators/morphemes) so partial prompts earn partial score and the best prompt is genuinely non-obvious.
5. **Failure-path coverage (C13)** comes free from sequence/array outputs (parse failures happen naturally); pure-MC tasks (#7, #11) may need deliberate malformed-output injection.
