# Capability Synthesis — c18: Depth-Controlled Synthetic Deduction (True/False/Unknown)

Scope: what 7 verified dossiers collectively say models CAN and CANNOT do on depth-controlled
synthetic deduction, and what that predicts for c18's reseed-only PrOntoQA baseline (True/False
exact-match, existing hop-depth settings, no new axes) at the cheap model tier, temperature 0.

Dossier keys used:
- `2210.01240-prontoqa` — PrOntoQA / "Greedy Reasoners" (Saparov & He, ICLR 2023)
- `2305.15269` — PrOntoQA-OOD (Saparov et al., NeurIPS 2023)
- `2308.07336-fld` — FLD (Morishita et al., ICML 2023)
- `2411.12498` — FLD×2 (Morishita et al., NeurIPS 2024)
- `2406.17169-multi-logieval` — Multi-LogiEval (Patel et al., EMNLP 2024)
- `2501.14851-justlogic` — JustLogic (Chen, Zhang & Tao, 2025)
- `2505.14615-satbench` — SATBench (Wei et al., 2025)

A structural caveat that colors every number below (EVIDENCE): several of these papers score the
**full proof chain**, not just a label (PrOntoQA, PrOntoQA-OOD, FLD "proof accuracy"). c18 scores the
**label only**. Label ("answer") accuracy is systematically and often dramatically HIGHER than proof
accuracy in the same setting (FLD GPT-4: answer 52.4 vs proof 12.8 on FLD; `2308.07336-fld` Table 4).
So proof-accuracy collapses in this literature do NOT translate one-to-one into label-accuracy
collapses. c18's floor/ceiling should be read against the label-only rows.

---

## 1. Single-forward-pass capability: where one-shot accuracy holds vs collapses

### What the evidence SHOWS

- **Shallow depth (1 hop / depth-1) is largely solved, even for older/cheaper models.**
  - PrOntoQA: text-davinci-002 "handles 1- and 3-hop examples well" (`2210.01240-prontoqa`, Sec 5.3).
  - Multi-LogiEval: depth-1 average label accuracy PL 85.4%, FOL 83.2% across a 6-model set that
    includes weak open models (`2406.17169-multi-logieval` Table 6). Even the below-random cases are
    at depth-5, not depth-1.
  - JustLogic depth-1 is the easy end of a monotonic decline (`2501.14851-justlogic` Sec 5.3/Fig 4).

- **Accuracy declines monotonically with depth within a single pass — this is the single most
  robust cross-paper finding.**
  - Multi-LogiEval: PL/FOL average drops d1→d5 from ~85%/83% to ~43%/33% (`2406.17169-multi-logieval`
    Table 6; Abstract "~68% at depth-1 to ~43% at depth-5").
  - JustLogic: "model accuracies generally decrease as depth increases" (`2501.14851-justlogic`
    Sec 5.3), extended to depths 8–11 where even DeepSeek R1 falls to 65% (Table 11).
  - PrOntoQA: 5-hop top-down fictional falls to **chance** for the best model tested
    (`2210.01240-prontoqa` Sec 5.3, Fig 4).
  - PrOntoQA-OOD: "performance decreases with increasing depth ... both ID and OOD accuracies
    decrease" (`2305.15269` Sec 4.2.3).
  - FLD: proof accuracy collapses to ~0% by depth 6–8 with the full axiom set (`2308.07336-fld`
    Sec 7.1, Table 7).

- **The collapse point depends on rule-set breadth, not depth alone.** FLD's O(A^d) argument: with
  implication-only rules (A≈2, PrOntoQA/RuleTaker-like) shallow-trained provers still generalize to
  depth 4–8; with the full axiom set (A≈10) accuracy hits ~0% by depth 6–8 (`2308.07336-fld` Sec 5.1,
  7.1). **c18 (PrOntoQA lineage) sits in the A≈2, if-then-chaining regime — the shallower-collapse,
  easier regime.**

- **Difficulty can also collapse accuracy below random in one branch of the label space.** SATBench
  (a *search/CSP* task, not chaining): o4-mini only 65.0% on hard UNSAT (≈50% random); gpt-4o-mini
  13.2% on UNSAT-Hard while 90.7% on SAT-Hard (`2505.14615-satbench` Table 3). Multi-LogiEval:
  several models score below the per-depth random baseline (`2406.17169-multi-logieval` Sec 4.2).

### What I INFER

- For c18 specifically (PrOntoQA if-then chaining, fictional ontology, label-only), one-shot accuracy
  should **hold high at D0–D1**, **degrade gradually across D2–D4**, and **approach but not
  necessarily hit chance at D5** — the PrOntoQA "5-hop → chance" result is (a) the *hardest* ordering
  (top-down) and (b) a much weaker 2022 model. Modern cheap models should do better than text-davinci-002.
- A three-way T/F/Unknown label has random floor 33% (not 50%), so "collapse to chance" means ~33%,
  and the depth at which cheap models reach it is likely D5 or beyond for the plain PrOntoQA rule set.

### Conflict to surface

- PrOntoQA says 1- and 3-hop are "handled well" and only 5-hop top-down collapses to chance; but that
  is *proof* accuracy on a 2022 model, top-down ordering (the paper's hardest condition). Multi-LogiEval's
  label accuracy already shows substantial d3 degradation (~60%). These are not contradictory once you
  account for (metric: proof vs label), (ordering: top-down vs bottom-up), and (model vintage). Depth
  effect direction agrees everywhere; the absolute collapse depth does not, and depends on those knobs.

---

## 2. Multistep / interactive / many-shot: what breaks, does interaction/more shots help?

### What the evidence SHOWS

- **Genuine interactive/multi-turn loops are rarely tested and mostly do NOT help; they often hurt
  smaller models.** The only real interactive ablation across the corpus is JustLogic's tree-of-thought:
  it helped only GPT-4o (71.4% vs 65.6% CoT) and *hurt* Llama3-8B (38.6% vs 57.8%) and GPT-4o-mini
  (48.6% vs 51.8%) (`2501.14851-justlogic` Table 10, App H.1). Authors' conclusion: "prompting techniques
  more expensive than vanilla CoT offer little to no performance advantage" and ToT is "too complex for
  most models to utilize." SATBench is 0-shot only and explicitly tests no interactive loop
  (`2505.14615-satbench` Sec 4).

- **More demonstrations help modestly but saturate ~8 and never remove the depth degradation.**
  - PrOntoQA / PrOntoQA-OOD used 8-shot "since adding further in-context examples did not improve
    performance" (`2305.15269` footnote 5; `2210.01240-prontoqa` Sec 5.1).
  - Multi-LogiEval: 3-shot raised accuracy at every depth (e.g. GPT-4 FOL d4 59.2%→68.3%) but "do not
    fully mitigate the inherent challenges ... in higher depths" (`2406.17169-multi-logieval` App G).
  - JustLogic: few-shot *without* CoT rationale can UNDERPERFORM zero-shot (Llama3-8B few-shot 41.8% <
    zero-shot 49.8%) (`2501.14851-justlogic` Table 6). More demos are not monotonically safe.

- **Sampling/aggregation tricks do not fix planning.** PrOntoQA self-consistency (40 samples) = 0.56
  vs 0.545 greedy, not significant; DFS-style demonstrations 0.55, also not significant; probability
  aggregation actually favors *wrong* proofs on items the model gets wrong (`2210.01240-prontoqa`
  App A.7). JustLogic self-consistency-CoT rarely beat vanilla CoT (`2501.14851-justlogic` Table 10).

- **The failure that breaks in multistep is planning / error propagation, not single-step validity.**
  PrOntoQA: individual steps are 93.2% strictly-valid at 5 hops; the failure is choosing a valid-but-
  *misleading* step at branch points and not recovering (`2210.01240-prontoqa` Sec 5.3–5.4). Multi-LogiEval:
  "mistakes in the initial reasoning step propagate"; longer chains do not correlate with correctness
  (`2406.17169-multi-logieval` Sec 4.3). SATBench: satisfiability bias / context inconsistency /
  condition omission — failures of exhaustive search and self-consistency (`2505.14615-satbench` Sec 5.3).

- **Fine-tuning helps but is out of scope for a prompt-only baseline.** FLD×2 ALT gains up to 30 pts on
  logic benchmarks (`2411.12498`); SATBench LoRA moved a 14B model only 51.9%→53.6% (`2505.14615-satbench`
  Sec 6.2). Not relevant to c18's reseed-only prompt baseline except as evidence that base-model prompting
  is the binding constraint.

### What I INFER

- For c18: an interactive or many-shot elaboration is unlikely to change the go/no-go picture. The
  binding lever is CoT vs no-CoT within a single pass (Section 4), not turns or demo count. Adding
  interaction risks *lowering* cheap-tier scores (ToT evidence).

---

## 3. Model-tier breakdown (cheap tier matters most)

### Frontier reasoning models (EVIDENCE)

- DeepSeek R1: 80.9% on JustLogic (only model above the 73.0% human average), "moderate" depth decline
  (`2501.14851-justlogic` Table 6). SATBench: R1 87.8% overall, o4-mini 89.3% overall but 65.0% on hard
  UNSAT (`2505.14615-satbench` Tables 3–4). o1 underperforms R1 (72.9%) via a depth-7 "answer Uncertain"
  degeneracy (`2501.14851-justlogic` App H.2). So even frontier reasoning models have a hard tail at high
  depth / UNSAT.

### Cheap tier — the closest available proxies (EVIDENCE)

None of the dossiers test the *exact* c18 quick-test tier (Gemini-2.5-Flash / GPT-5-mini / Claude Haiku 4.5
/ DeepSeek-Chat). The nearest proxies, on 3-way T/F/Uncertain or close tasks:

- **JustLogic (3-way T/F/Uncertain, exact-match label, closest task shape to c18):**
  - GPT-4o-mini: 0-shot 53.0%, few-shot 54.7% (best), CoT 51.8% (`2501.14851-justlogic` Table 6).
  - Llama3-8B: 0-shot 49.8%, CoT 57.8% (best) (Table 6).
  - Llama3-70B: 0-shot 53.1%, CoT 64.6% (Table 6).
  - Random floor = 33.3%; human avg 73.0%; best model (R1) 80.9%.
- **Multi-LogiEval (binary Yes/No, easier label space):** cheap/open models (Mistral-7B, Yi-34B,
  Orca-2-13B) range from ~80% at d1 down to 6.7–20% at d5 FOL (`2406.17169-multi-logieval` Table 6);
  Gemini-Pro (a mid proprietary proxy) PL d1 90%→d5 60%, FOL 76.9%→53.3%.
- **FLD (3-way proved/disproved/unknown, label accuracy):** GPT-3.5-Turbo 35.8% (FLD) / 37.6% (FLD*),
  ~random; LongAlpaca-13B 21.2%/19.6%, below the 33.3% random floor (`2308.07336-fld` Table 4). These
  are hard *axiom-set* corpora (A≈10), not the PrOntoQA if-then regime c18 uses.
- **SATBench (binary, but search task):** gpt-4o-mini 53.9% overall (near-random) (`2505.14615-satbench`
  Table 3).

### What I INFER for the cheap tier

- On the c18 *label-shape* proxy (JustLogic 3-way), cheap-tier models cluster around **50–57%
  zero-shot** and **~52–60% with CoT** — i.e. clearly above the 33% floor but far below strong-model
  ceilings. JustLogic uses harder natural-language surface forms than PrOntoQA's clean fictional
  templates, so PrOntoQA-style surface should sit somewhat *higher*.
- The 2026 quick-test cheap tier (Gemini-2.5-Flash, GPT-5-mini, Haiku 4.5, DeepSeek-Chat) is
  meaningfully stronger than the 2024 GPT-4o-mini/Llama3-8B proxies. INFERENCE: on plain PrOntoQA
  if-then chaining at shallow-to-moderate depth, this tier should land materially above the JustLogic
  proxies — the JustLogic/FLD numbers are a *lower*-leaning proxy for c18, not an upper bound.

---

## 4. Prompting strategies and representations that moved accuracy (with conditions)

EVIDENCE, ranked by leverage:

1. **CoT vs zero-shot / few-shot-without-rationale — largest reliable lever for non-reasoning models.**
   JustLogic CoT gains: Llama3-8B 49.8→57.8 (+8.0), Llama3-70B 53.1→64.6 (+11.5), GPT-4o 53.8→65.6
   (+11.8); exception GPT-4o-mini CoT 51.8 < few-shot 54.7 (`2501.14851-justlogic` Table 6). Multi-LogiEval
   is entirely zero-shot-CoT primary and still degrades with depth (`2406.17169-multi-logieval` Sec 4.1).

2. **Naming the logic / argument forms in the prompt helps.** JustLogic "CoT w/o propositional logic"
   underperformed standard CoT for 3 of 4 models (`2501.14851-justlogic` App H.1). SATBench error-aware
   prompting (naming the 4 failure modes, still single-shot) corrected 60.4% (o4-mini) / 73.2% (R1) of
   previously-failing cases (`2505.14615-satbench` Sec 6.2) — the single biggest single-shot prompt swing
   in the corpus.

3. **Sentence ordering (representation).** PrOntoQA: bottom-up (matches gold proof order) is easier than
   top-down (reversed); the gap widens with depth, and 5-hop top-down fictional → chance
   (`2210.01240-prontoqa` Sec 5.3). Directly relevant to c18: PrOntoQA's default ordering knob can swing
   deep-instance accuracy from workable to chance.

4. **Distractors are required to avoid a trivial shortcut.** Without distractors, InstructGPT predicted
   the T/F label "almost perfectly" by string-matching the queried property (`2210.01240-prontoqa`
   App A.2); PrOntoQA-OOD makes the same design point (`2305.15269` Sec 3). **CRITICAL for c18:** the
   reseed-only baseline must keep the codebase's distractors on, or the ceiling prompt trivially maxes out
   and the task measures shortcut-matching, not reasoning.

5. **Natural-language vs symbolic surface.** SATBench: raw CNF formula beats narrative NL for every model
   (o4-mini 94.3 vs 89.4; DeepSeek-V3 87.3 vs 84.0) (`2505.14615-satbench` Table 6). PrOntoQA/c18 use clean
   templated NL (between the two extremes), simpler than SATBench narratives or JustLogic GenericsKB text.

6. **Multi-turn / ToT / self-consistency — mostly neutral-to-negative** (see Section 2).

Representation caveat (EVIDENCE): none of these papers ran a JSON-vs-NL-vs-symbolic ablation *on the
PrOntoQA family itself*; the ordering and distractor knobs are the representation levers actually measured
for c18's lineage.

---

## 5. Implications for c18's reseed-only baseline (naive-prompt floor, ceiling-prompt score)

Setup being scored (per task spec): PrOntoQA instances regenerated as-is (fresh seeds, fresh fictional
ontologies), existing hop-depth settings, native question format, **True/False exact-match** (the
codebase's native closed-world 2-way label — NO added Unknown handling, no distractor/soft-rule strata
beyond what the repo natively emits). Cheap tier, temperature 0.

**Key structural facts that raise the predicted scores vs the literature proxies:**
- Native PrOntoQA closed-world output is **True/False (2-way, 50% random floor)**, NOT the 3-way
  33%-floor task most of these papers score (`2210.01240-prontoqa` — the base paper's PrOntoQA is
  closed-world T/F only; Unknown comes from ProofWriter/FLD, which c18 is NOT adding here).
- c18 is in the **easy A≈2 if-then regime** (`2308.07336-fld`), the shallow-collapse regime.
- Surface form is **clean fictional templates**, easier than JustLogic NL and SATBench narratives.
- 2026 cheap tier > the 2024 GPT-4o-mini/Llama3-8B proxies.

All four factors push c18's expected cheap-tier accuracy **above** the raw JustLogic/FLD numbers.

### Predicted naive-prompt FLOOR (zero-shot, no CoT, no logic-form naming; cheap tier, temp 0)

- **Estimate: ~60–78% overall (aggregated across the native depth range).** Reasoning: JustLogic 3-way
  zero-shot for GPT-4o-mini/Llama3-70B is ~53%; converting to 2-way, cleaner surface, easier rule set,
  and a stronger 2026 tier each add headroom. At shallow depth this is likely 80%+; at the deepest native
  hops it drops toward the 50% two-way floor.
- **Uncertainty: MODERATE-TO-HIGH.** No dossier tests this exact model tier, this exact 2-way native
  format, or PrOntoQA *label-only* accuracy for modern models — the closest label-only PrOntoQA numbers
  are *proof* accuracy on a 2022 model. There is a real risk the shortcut warning applies: if the
  regenerated instances lack strong distractors, even the naive prompt could score **very high (85–95%+)**
  by property-string-matching (`2210.01240-prontoqa` App A.2). Whether the repo's default settings emit
  distractors by default is the single biggest swing factor and must be checked in the codebase.

### Predicted ceiling-prompt SCORE (CoT + logic-form naming + best few-shot; cheap tier, temp 0)

- **Estimate: ~75–90% overall.** Reasoning: CoT adds ~+8 to +12 pts for non-reasoning models
  (`2501.14851-justlogic`); logic-form naming and error-aware prompting add more (`2505.14615-satbench`
  Sec 6.2). But the depth ceiling is real and unfixable by prompting: at the deepest native hops the
  cheap tier will still degrade (Multi-LogiEval, JustLogic depth curves), and even frontier reasoning
  models hit a hard tail (R1 65% at depth 8–11; o4-mini 65% hard-UNSAT). So the aggregate ceiling stays
  short of ~95%.
- **Uncertainty: MODERATE.** CoT lift direction is robust across papers; magnitude for this exact tier is
  extrapolated.

### Go/no-go read (INFERENCE)

- **If the reseed-only baseline keeps distractors ON and the native hop range reaches deep enough
  (≈D5):** expect a naive floor ~60–75% and a ceiling ~80–90%, with a **meaningful but not dramatic
  floor-to-ceiling gap and a live depth signal**. This is a usable but *soft* discriminator — the cheap
  tier is unlikely to saturate at the ceiling, but it also may not be hard enough to strongly separate
  models unless deep hops dominate the mix.
- **If distractors are OFF / weak in the native settings:** high risk the naive prompt already scores
  **85–95%** via shortcut-matching (`2210.01240-prontoqa` App A.2, `2305.15269` Sec 3), leaving little
  headroom and a near-ceiling floor → **the task likely needs added complexity** (distractors, deeper
  hops, or the Unknown/3-way axis) to be a good discriminator.
- The **2-way T/F native format** (50% floor, no Unknown-withholding test) is itself a limiting factor:
  the literature's most diagnostic failures (under-using "unknown" — `2308.07336-fld` Sec 5.2,
  `2411.12498` Sec 2.2; open-world withholding — JustLogic) are exactly what a 2-way reseed-only baseline
  does NOT probe. This is a design argument that added complexity (the Unknown axis) would materially
  increase the task's diagnostic value, independent of the raw accuracy numbers.

**Bottom line for the decision:** the evidence predicts the reseed-only 2-way baseline lands in a
"comfortable middle" for the cheap tier — clearly above floor, clearly below saturation *if and only if*
distractors are on and deep hops are included. Its main weakness is not that it's too easy in the
aggregate, but that (a) it may saturate at shallow depth / without distractors, and (b) 2-way T/F omits
the most diagnostic failure mode (Unknown-withholding). Both point toward adding at least the distractor
and/or Unknown axes if a strong model-discriminating signal is required.

---

## Executive summary (10 lines)

1. Robust cross-paper finding: single-pass accuracy declines MONOTONICALLY with proof depth; depth-1 is
   near-solved, deep instances degrade toward chance (`2406.17169`, `2501.14851`, `2210.01240`, `2305.15269`).
2. The collapse depth depends on rule-set breadth: c18's PrOntoQA if-then regime (A≈2) is the *easier*,
   shallower-collapse regime vs FLD's axiom set (A≈10) that hits ~0% proof-acc by depth 6–8 (`2308.07336`).
3. Failures are planning/error-propagation, not single-step validity: steps are ~93% valid at 5 hops but
   models pick misleading branches and don't recover (`2210.01240` Sec 5.3–5.4; `2406.17169` Sec 4.3).
4. Interaction rarely helps and often hurts cheap models: tree-of-thought hurt Llama3-8B (57.8→38.6);
   self-consistency ≈ greedy; more demos saturate ~8 (`2501.14851` Table 10; `2210.01240` App A.7).
5. Biggest single-shot levers are CoT (+8–12 pts) and naming the logic / error-aware prompting
   (recovers 60–73% of failures), not turns or demo count (`2501.14851`, `2505.14615` Sec 6.2).
6. Distractors are mandatory: without them models T/F-label "almost perfectly" by string-matching, not
   reasoning (`2210.01240` App A.2; `2305.15269` Sec 3) — the key design risk for c18's ceiling.
7. Cheap-tier proxies on 3-way T/F/Uncertain cluster ~50–57% zero-shot, ~52–65% with CoT (JustLogic
   Table 6); FLD 3-way puts GPT-3.5/LongAlpaca at/below the 33% random floor (`2308.07336` Table 4).
8. c18's native format is 2-way T/F (50% floor, not 3-way 33%), cleaner surface, easier rules, and a
   stronger 2026 tier — all push predicted accuracy ABOVE these proxies.
9. Predicted cheap-tier, temp-0 scores (aggregate over native depth, distractors ON): naive-prompt floor
   ~60–78%, ceiling-prompt ~75–90%; uncertainty moderate-to-high (no dossier tests this exact tier/format).
10. Go/no-go: usable but SOFT discriminator if distractors on + deep hops included; if distractors off it
    likely saturates at 85–95% (needs added complexity), and 2-way T/F omits the most diagnostic axis
    (Unknown-withholding) — arguing to add the distractor and/or Unknown axes.
