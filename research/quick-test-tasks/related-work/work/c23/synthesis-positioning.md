# c23 · Hidden-Rule String-Transform Induction Ladder — Positioning Synthesis

**Scope.** Positions c23-as-designed against its six related-work dossiers, then analyzes publishability of the planned two-phase run (un-optimized baseline first, prompt optimization second). Evidence and judgment are labeled separately throughout: **[E]** = drawn from a dossier / candidate file / trends doc / manifest; **[J]** = my inference.

**The six papers.**
- **IB** — InductionBench (Hua et al., ACL 2025; 2502.15823) — the codebase source and central design driver.
- **PCFG** — Compositionality Decomposed / PCFG SET (Hupkes et al., JAIR 2020; 1908.08351).
- **Re-ARC** — Procedural ARC example generation (Hodel, 2024; 2404.07353).
- **WILT** — Wason multi-turn inductive logic (Banatt et al., Oct 2024 / ICLR 2025; 2410.10998).
- **MIR** — MIR-Bench many-shot ICL (Yan et al., NeurIPS 2025 D&B; 2502.09933).
- **Falsify** — FalsifyBench rule-discovery games (Bertolazzi et al., 2026; 2606.04751).

---

## 1. Positioning map

**[J]** Three axes organize this family cleanly. Each is directly grounded in a dossier field (`protocol.mode`, `relation_to_candidate`, verifier descriptions).

### Axis A — Interaction structure (single-call vs. multi-turn evidence-gathering)
Where does the evidence come from, and how many model calls per instance?

| Pole | Papers | Evidence |
|---|---|---|
| **Single forward pass, evidence provided** | MIR, IB, **c23** | **[E]** MIR `mode:single-shot`, all n demos + one query in one call; IB `mode:single-shot`, zero-shot CoT, one rule-set emitted per prompt; c23 is explicitly single-call (candidate file: "single-call constraint"). |
| **Trained-from-scratch, no ICL channel** | PCFG | **[E]** PCFG dossier: "NOT an LLM-prompting paper"; supervised seq2seq over ~85k pairs; no prompt-time exemplars. |
| **Multi-turn, model generates its own evidence** | WILT, Falsify | **[E]** WILT `mode:interactive`, up to 30 self-proposed test triples + 1 guess; Falsify `mode:interactive`, up to 20 player-oracle rounds. |
| **No model in the loop** | Re-ARC | **[E]** Re-ARC `mode:not-applicable`; a generator+verifier toolkit, "reports zero LLM results." |

### Axis B — Output shape & verifier type
What does the model emit, and how is it graded?

| Pole | Papers | Evidence |
|---|---|---|
| **Apply rule → single transformed value, exact-match vs. deterministic oracle** | MIR, **c23** | **[E]** MIR: emit `y_new=f(x_new)`, string/`ast.literal_eval` exact match; c23: transformed short string (or 4-way MC), string equality vs. an independently-sampled deterministic transducer. |
| **State the rule → symbolic rule-set, set-overlap metrics** | IB | **[E]** IB grades precision/recall vs. unique minimal rule set + compatibility (rules re-applied to the *training* sample, not a held-out query). |
| **State the rule → free-form symbolic function, behavioral-equivalence check** | WILT, Falsify | **[E]** WILT: Python lambda checked over 64k-grid + 10k floats; Falsify: property string judged for semantic equivalence by an **LLM oracle** (kappa 0.34–0.98). |
| **Full-sequence exact match (or output-vs-output consistency)** | PCFG | **[E]** PCFG "sequence accuracy" (whole output must equal target) plus a "consistency" metric (output vs. own earlier output). |

### Axis C — Rule provenance & difficulty dial
Are rules latent (must be induced) or named? Is the difficulty dial *rule composition* or something else?

| Pole | Papers | Evidence |
|---|---|---|
| **Latent, composed K-independent sub-rules; dial = # rules stated/induced** | **c23** (as designed) | **[E]** candidate file: "K independent latent rules," smooth incremental ladder, floor (naive prompt) → ceiling (state all rules). |
| **Latent single rule; dial = formal complexity (k, \|Σ\|, # rules)** | IB | **[E]** IB dials context window k, alphabet \|Σ\|, and number of minimal rules; k is the largest difficulty driver. |
| **Latent single monolithic function; dial = shot count** | MIR | **[E]** MIR: arbitrary Python functions from HumanEval+/MBPP+/APPS; dial = 4→2048 shots; "no equivalent partial-rule-credit structure." |
| **Latent single boolean/category; dial = catalog tier / taxonomic distance** | WILT, Falsify | **[E]** WILT Easy/Medium/Very-Hard catalog of 50; Falsify WordNet taxonomic distance (close ≤4 edges / deep >4). |
| **Named (not latent) operators; dial = length/depth/# functions** | PCFG | **[E]** operator names are literal input tokens; no induction; dial = seq length, parse depth, # functions. |
| **Fixed transformation; dial = cardinalities (grid/object/symbol counts)** | Re-ARC | **[E]** RNG-/PSO-Difficulty over grid size, object/symbol counts; one fixed transform per task. |

### Crowded and empty regions **[J]**

- **Crowded — "state the latent rule as a symbolic artifact, grade the artifact":** IB, WILT, Falsify all live here. This is the well-populated evaluation mode of the induction family; it comes with expensive or noisy verifiers (symbolic diffing; 64k-grid equivalence; LLM-judge).
- **Crowded — multi-turn Wason-style rule discovery:** WILT and Falsify are near-duplicates on Axis A/B, differing mainly in domain (numbers vs. WordNet) and whether the oracle is deterministic or an LLM. **[E]** Falsify's own related work positions itself as extending WILT.
- **Near-empty — c23's target cell:** single-call **× apply-rule-to-held-out-query-with-exact-match × latent, *composed* K-independent rules with a partial-credit ladder.** **[E]** MIR is the closest neighbor (single-call, apply-and-exact-match) but its functions are monolithic with "no partial-rule-credit structure"; IB shares the subregular substrate but grades rule-set reconstruction against the training sample, not a held-out query. **[J]** No dossier paper occupies c23's exact cell — the combination of (apply, not state) + (exact-match on a fresh query) + (composed decomposable rules) is the genuinely open region.
- **Empty — cheap deterministic verifier + decomposable partial credit:** everyone with decomposable structure (IB) pays for it with a symbolic verifier; everyone with a cheap verifier (MIR, PCFG) has monolithic all-or-nothing scoring. **[J]** c23's design claim is precisely to sit in this empty intersection: string exact-match (cheap, unambiguous) *and* a decomposable rule ladder.

**[J] Load-bearing caveat that spans the whole map:** IB's headline finding — frontier models score ~5.69% weighted compatibility, 0.00 at moderate difficulty — is a documented reachability warning for anything built on the subregular substrate c23 reuses (see §3e). The "empty cell" is empty partly because it is hard to keep the ceiling reachable there.

---

## 2. Per-paper: what c23-as-designed adds (honest about overlap)

### IB — InductionBench (2502.15823) — highest overlap; the codebase source
**[E]** Deepest overlap of the six: same formal object (subregular ISL/L-OSL/R-OSL transduction), same seeded-generator contamination stance, same k/\|Σ\|/#-rules difficulty philosophy, and c23 literally vendors IB's `synthetic_data_generation.py` generator+oracle. **[E]** What differs: IB asks the model to *state* a full symbolic rule set and grades it by precision/recall/compatibility **against the training sample it was given** — it never tests application to a genuinely held-out query, and its output is grammar-induction-shaped, not a single transformed string. **[J/E]** c23-as-designed adds a held-out-query, apply-the-rule, string-exact-match protocol (trivial to grade vs. IB's symbolic rule-diffing), broadens beyond pure ISL/OSL edits to a multi-family ladder (casing, swaps, command→action, 1D-ARC), and — critically — commits to curating rule complexity to keep a reachable ceiling, directly answering IB's own 0–5.69% ceiling warning. **Honest overlap:** the *un-optimized baseline* as planned (fresh-seed InductionBench ISL/OSL, single-rule, I/O-demos+query) is very close to an IB instance re-scored under exact-match-on-a-query instead of compatibility; the novelty of the baseline phase is the scoring/format change and fresh seeds, not a new task substrate. The composition ladder — the part that is genuinely new relative to IB — is explicitly *deferred* out of phase 1.

### PCFG — Compositionality Decomposed (1908.08351) — taxonomic template, different paradigm
**[E]** Strong conceptual overlap on the operator taxonomy: PCFG's per-operator functions (copy, reverse, echo, swap, repeat, append…) are the template for c23's "K independent string-transform rules," and both score short transformed sequences by exact match with a smooth length/depth/#-function difficulty curve. **[E]** But PCFG trains seq2seq models from scratch on ~85k labeled pairs — there is no in-context channel — and its operators are **named tokens literally present in the input**, so no induction from demos is required (the dossier flags the "direct template" claim as *overstating* the mechanistic similarity for exactly this reason). **[J]** c23 adds the entire in-context, gradient-free, frozen-LLM induction mechanic PCFG lacks: unnamed latent rules inferable only from I/O demos, the stated/induced-rule ladder as the independent variable (vs. PCFG's training-distribution manipulations), a 4-way MC variant, and per-instance re-randomized invented vocabularies (PCFG's 520-symbol alphabet is fixed and public, so its released corpus is not contamination-safe for a modern LLM eval). **Honest overlap:** the operator taxonomy and the "decompose difficulty into independent axes" style are borrowed wholesale; c23's contribution here is porting them into the ICL-induction regime, not inventing the operators.

### Re-ARC (2404.07353) — tooling analogue, no evaluation content
**[E]** Overlap is "family resemblance only": both concern transformation induction with a tunable difficulty dial and an oracle verifier, and Re-ARC's RNG/PSO difficulty interval is the structural analogue of c23's rule-count ladder. **[E]** But Re-ARC defines no prompting protocol, no output shape, no model results — it is a 2D-grid generator+verifier toolkit (`mode:not-applicable`). **[J]** c23 adds everything model-facing that Re-ARC omits (few-shot demos+query format, exact-match/MC scoring, the composable-rule ladder) and works on short linear strings rather than 30×30 grids. **Honest overlap:** minimal and methodological — Re-ARC is useful mainly as prior art that "regenerate instances with fresh seeds from a procedural generator to defeat memorization" is an accepted pattern, which is exactly what c23's phase-1 plan does. It contests essentially none of c23's design space.

### WILT (2410.10998) — same family, opposite protocol pole
**[E]** Shares the "induce a latent rule and resist memorization by construction" goal and the tiered-difficulty-catalog framing; WILT's doom-loop/confirmation-bias failures parallel c23's "naive-prompt miss." **[E]** But WILT is fully interactive (up to 30 self-directed test turns + 1 guess), its evidence is model-generated (introducing hypothesis-space-reduction as a distinct skill), its output is a free-form Python lambda checked by an expensive 64k-grid behavioral-equivalence oracle, and its rules are single boolean predicates from a fixed 50-item human-authored catalog. **[E]** c23's own dossier already lists WILT as "generator only, not adopted" because its episodic form violates the single-call constraint. **[J]** c23 adds a single-call protocol that isolates *pure in-context rule induction* from WILT's dominant multi-turn confounds (evidence-gathering strategy, turn-budget management), a combinatorial K-rule space vs. WILT's fixed catalog, a cheap unambiguous string-match verifier, and an MC variant WILT has no analogue for. **Honest overlap:** conceptually adjacent but methodologically disjoint — they measure different things (induction-from-static-demos vs. interactive experimentation).

### MIR — MIR-Bench (2502.09933) — closest single-call neighbor
**[E]** After IB, the closest: both are single-call inductive tasks that generate I/O pairs from a latent function, require predicting a held-out query's output in one pass, and grade by exact match against an independently-computed deterministic ground truth; both are contamination-resistant by construction; MIR's related work cites the same SCAN/ARC/1D-ARC/List-Functions lineage. **[E]** Differences: MIR's dial is *many-shot* (4→2048, targeting attention-aggregation limits), its functions are arbitrary heterogeneous Python programs with **no decomposition into K independent scoreable sub-rules** (all-or-nothing per instance), and it has no MC variant. MIR's own findings are directly relevant confounds: no-CoT beats forced-CoT unanimously; performance saturates by ~256 shots; RAG-selected shots don't beat random. **[J]** c23 adds the decomposable K-independent-rule structure with an explicit partial-credit ladder (resolving *which* rule strata a model misses, not just pass/fail), a 4-way MC lower-variance signal, a tightly-scoped small string domain (cheap quick-test instances vs. MIR's long-context stress test), and deliberately *excludes* the many-shot axis MIR is built around. **Honest overlap:** this is the paper c23 most risks being seen as a special case of — "MIR-Bench with a smaller string domain and a rule-count ladder." The differentiator that must hold is that the ladder produces genuinely decomposable partial credit, which MIR explicitly does not offer.

### Falsify — FalsifyBench (2606.04751) — WILT's semantic cousin, opposite pole
**[E]** Descends from the same "infer a hidden rule, don't surface-pattern-match" lineage and cites SCAN/ARC as ancestors; its "surface-level linguistic hypothesis" failure mode is thematically close to c23's "one-shot guessable" risk. **[E]** But it is fully interactive (≤20 player-oracle turns), the model constructs its own test triples, output is multi-field JSON, and — importantly — the verifier is an **LLM oracle** validated only against a 1,200-instance human sample with kappa as low as 0.34; rules are single WordNet categories dialed by taxonomic distance, with no composed-sub-rule structure. **[J]** c23 adds a genuinely single-pass evaluation (isolating induction from interactive strategy, which Falsify shows dominates its score variance), a fully deterministic oracle with no LLM-judge noise, a quantifiable partial-credit ladder vs. Falsify's binary per-game outcome, invented-vocabulary contamination-proofing (vs. public WordNet), and far cheaper instances (~$150 bought only 1,200 interactive games). **Honest overlap:** small on method; Falsify mainly matters as evidence that the broader rule-discovery family is active into 2026 and that "does the model do real induction vs. shortcut" is a live research question c23 also probes.

---

## 3. Publishability analysis (conditional on prompt optimization producing large *verified* gains)

**[E] Planned phase 1 (baseline):** un-optimized, InductionBench-style subregular ISL/OSL instances **regenerated with fresh seeds** using the existing InductionBench codebase; **single-rule instances only** (no composition ladder, no multi-rule stacking); existing I/O-demos+query format; short-string exact-match. **[E] Phase 2:** optionally prompt-optimize with COPRO / MIPROv2 / GEPA. This analysis assumes phase 2 yields a large, verified accuracy gain.

### (a) The precise workshop-paper claim
**[J]** State it narrowly and mechanistically, not as a capability claim:

> "On a fresh-seed, single-rule InductionBench-style subregular (ISL/OSL) string-transform induction task with I/O-demos-plus-query format and exact-match scoring, an un-optimized baseline prompt achieves X% exact-match; automated prompt optimization (best of COPRO/MIPROv2/GEPA) raises this to Y% (Δ = Y−X points, N held-out instances, seeds S, on model M), and we verify the gain is (i) held-out-seed generalizing, (ii) not attributable to demo selection alone, and (iii) driven by the optimized instruction inducing more of the latent rule."

**[J]** What the claim must **not** say: that LLMs "can/cannot do inductive reasoning" (IB already owns the negative capability claim, and one task cannot support the positive one); that the *composition ladder* was tested (it is deferred to phase 2's task design, not the baseline). The publishable unit is a *measured optimizer-driven delta on a controlled, contamination-proof induction task*, with the delta's mechanism verified.

### (b) Which publication line
**[J]** Two candidate lines; the honest answer depends on where the delta is largest and most defensible:

- **Prompt-optimizer benchmarking line (primary, if the finding is "optimizer X moves this task by Δ").** **[E]** This line is active and receptive: GEPA is an ICLR 2026 **oral** (2507.19457; beats MIPROv2 ~13% aggregate, ~35× fewer rollouts) and **MAS-PromptBench** (2606.23664) is "a benchmark specifically for comparing optimizers." **[E]** The trends doc's own "bottom line" says publishing a chosen task "as a small prompt-optimizer benchmark is itself an active, receptive line of work right now." **[J]** A single contamination-proof, exact-match, few-shot-sensitive induction task where COPRO/MIPROv2/GEPA visibly diverge is exactly the artifact this line consumes. This is the stronger fit for a *large-gain* result.
- **This family's own evaluation line (secondary).** **[E]** Areas #4/#5 of the trends doc (in-context rule learning; rule induction / hypothesis search) are "moderate-high" activity with 2025–2026 papers (MIR, Falsify, WILT, the 2509.01016 error-sources paper). **[J]** But a single-rule, single-task result is thin for this line without the composition ladder that differentiates c23 from IB/MIR — so absent phase-2 ladder data, the evaluation line is a weaker home than the optimizer line.

**[J] Recommendation:** frame a large verified gain primarily as an **optimizer-benchmarking result** (a new small, clean, decomposable-in-principle test instance for COPRO/MIPROv2/GEPA), with the induction-family framing as motivation rather than the headline.

### (c) Baselines, ablations, comparisons reviewers will demand
**[J]**, grounded in specific dossier findings:

1. **Optimizer coverage + a fair floor.** All three named optimizers (COPRO, MIPROv2, GEPA) with matched compute budgets, plus a non-trivial baseline (a competent hand-written prompt, not a strawman). **[E]** GEPA's own oral-level claim is *relative to MIPROv2*, so reviewers expect the head-to-head.
2. **Demo-selection vs. instruction ablation.** **[E]** MIR shows RAG-selected shots don't beat random and IB shows few-shot count helps only in easy settings — so reviewers will demand you separate "optimizer found better demos" from "optimizer found a better instruction." A gain that is purely demo-selection is a different (weaker) claim.
3. **CoT confound.** **[E]** MIR: no-CoT beats forced-CoT *unanimously* across 21 models. You must fix or explicitly sweep the CoT axis, or the "optimization gain" may be an artifact of the optimizer stumbling onto direct-answer prompting.
4. **Held-out-seed generalization.** **[E]** c23's own repo review found the generator is non-deterministic unless `PYTHONHASHSEED` is pinned; reviewers will demand the optimized prompt is evaluated on *fresh seeds disjoint from the optimization set*, or the gain is memorization of the optimization pool.
5. **Reachability / ceiling calibration.** **[E]** IB reports 0.00–5.69% at moderate difficulty; PCFG plateaus below 100% even with unlimited training. Reviewers will ask where your baseline and ceiling sit and whether the gain is real headroom vs. noise on an unreachable task. **[E]** MIR's precedent (excluding instances that are 0% for all models at all shot counts) is the expected mitigation.
6. **Statistical resolution.** **[E]** IB and PCFG both note per-instance binary exact-match; c23's measurement-skeptic note flags that a 10-point change needs enough query items to clear noise. Reviewers want CIs / seed-level variance, not a single point estimate.
7. **Model breadth.** **[J]** At least 2–3 models (one frontier, one mid) so the claim isn't "GEPA helps model M once."
8. **Degenerate-solution guard.** **[E]** IB warns a "memorize every pair as its own rule" strategy trivially maximizes compatibility; the analogue here is checking the optimized prompt didn't overfit to the demo strings rather than inducing the rule.

### (d) Realistic venues / workshops and why
**[J]**
- **A prompt-optimization / LLM-methods workshop at a major venue (ICLR/NeurIPS/ACL workshops).** **[E]** GEPA-oral and MAS-PromptBench establish an active, named community in exactly this space; a small clean optimizer-comparison task is on-topic and welcome. **Best fit** for a large-gain result.
- **DSPy / agents / eval-tooling workshops.** **[J]** COPRO/MIPROv2 are DSPy-native; a reproducible task+harness that discriminates optimizers is directly useful to that ecosystem.
- **An ICL / inductive-reasoning workshop (the trends doc's areas #4/#5).** **[E]** MIR (NeurIPS D&B), Falsify, WILT show the substrate is publishable; **[J]** but as a *secondary* venue, and stronger once the composition ladder exists.
- **[J] Not a main-conference paper on the phase-1 result alone.** A single-rule, single-task optimizer delta is workshop-scale; a main-track paper needs the multi-family ladder, multiple tasks, or a genuinely new optimizer insight.

### (e) What would NOT be publishable (recognize early)
**[J]**
1. **No gain / gain within noise.** If COPRO/MIPROv2/GEPA don't beat a competent baseline beyond seed variance, there is no result. **[E]** Plausible outcome — IB found few-shot help is "negligible" past low complexity and MIR found many-shot saturates; both suggest this substrate can be optimization-inert.
2. **Gain that vanishes on fresh seeds.** **[E]** Given the `PYTHONHASHSEED` non-determinism, a gain that doesn't transfer to held-out seeds is overfitting, not a finding.
3. **Gain fully explained by demo-selection or CoT-toggling.** **[E]** If the ablations (c-2, c-3) show the delta is entirely "picked better demos" or "switched to no-CoT," the interesting claim (optimizer induced more of the rule) collapses into a known effect.
4. **Unreachable ceiling.** **[E]** If the task sits in IB's 0–5% regime, both baseline and optimized are near-floor and the delta is uninterpretable — the reachability-calibration step must pass *before* freezing.
5. **"LLMs can do induction" over-claim from one task.** **[J]** Not publishable as stated; a single single-rule task cannot support a general capability claim, and reviewers familiar with IB/MIR/WILT will reject the over-reach.
6. **Pure IB re-run.** **[J]** If the baseline is indistinguishable from InductionBench (same substrate, same-ish scoring) with no optimizer story and no ladder, it is a replication, not a contribution.

---

## 4. Research-currency verdict

### Evidence (OpenAlex `cited_by` / `by_year` / `since_2025`, fetched 2026-07-22)
| Paper | cited_by | since_2025 | by_year |
|---|---|---|---|
| PCFG (1908.08351, JAIR 2020) | 16 | 0 | 2019:2, 2020:6, 2021:8 |
| IB (2502.15823, ACL 2025) | 0 | 0 | — |
| Re-ARC (2404.07353, 2024) | 0 | 0 | — |
| WILT (2410.10998, Oct 2024) | 0 | 0 | — |
| MIR (2502.09933, NeurIPS 2025 D&B) | 0 | 0 | — |
| Falsify (2606.04751, 2026) | 0 | 0 | — |

### Stated caveats (required)
**[E]** These are **OpenAlex** counts, fetched **2026-07-22**. OpenAlex **undercounts relative to Google Scholar** (it misses many preprint-to-preprint and workshop citations). For **2026 papers the count is near-zero by construction** (indexing lag); the same lag depresses late-2025 papers. So the zeros for IB, Re-ARC, WILT, MIR, and Falsify are **not** evidence of low impact — IB is ACL 2025 main, MIR is NeurIPS 2025 D&B, both peer-reviewed at top venues; the zero reflects OpenAlex coverage lag, not neglect. PCFG's 16 (all pre-2022, `by_year` stopping at 2021) is an OpenAlex-partial trail for a well-known 2020 JAIR paper, again an undercount.

### Verdict **[J]**
The OpenAlex numbers are **too lag-biased to use as a currency signal for this family** and should be treated as a floor, not a measurement. The **qualitative** currency evidence is what matters and it is strong: **[E]** the candidate file and trends doc independently establish that in-context rule induction is unusually active in 2025–2026 (trends areas #4 "moderate-high" and #5 "moderate-high"; IB ACL 2025, MIR NeurIPS 2025 D&B, Falsify a 2026 preprint, WILT ICLR 2025, plus the 2509.01016 error-sources-in-rule-induction paper). **[E]** The adjacent prompt-optimization line c23's phase-2 targets is *highly* current (GEPA ICLR 2026 oral; MAS-PromptBench 2026). **[J] Net: high research currency, established via venue/recency and the trends taxonomy — explicitly not via the citation counts, which are undercounted and near-zero-by-construction for the 2025–2026 papers.**

---

## Executive summary (10 lines)

1. Three axes organize this family: interaction structure (single-call vs. multi-turn), output+verifier (apply-and-exact-match vs. state-the-rule-and-symbolic/LLM-judge), and rule provenance (latent-composed vs. named/single).
2. c23-as-designed targets a near-empty cell: single-call × apply-to-held-out-query × exact-match × latent *composed* K-independent rules with a partial-credit ladder.
3. The crowded regions are "state the rule as a symbolic artifact" (IB, WILT, Falsify) and "multi-turn Wason discovery" (WILT, Falsify near-duplicates); c23's empty cell buys a cheap deterministic verifier *and* decomposable credit at once.
4. Highest overlap is InductionBench (same subregular substrate, vendored generator); c23 adds held-out-query exact-match, a multi-family ladder, and reachability curation — but the phase-1 baseline is close to a re-scored IB instance.
5. MIR-Bench is the closest single-call neighbor; the differentiator that must hold is genuine decomposable partial credit (MIR has none). PCFG donates the operator taxonomy but uses named, not latent, rules. Re-ARC/WILT/Falsify are adjacent but methodologically disjoint.
6. If optimization yields a large verified gain, state it narrowly: an optimizer-driven exact-match delta on a fresh-seed, single-rule, contamination-proof induction task on model M — not a capability claim, and not a ladder claim (ladder is deferred).
7. Primary publication line is prompt-optimizer benchmarking (GEPA ICLR 2026 oral; MAS-PromptBench 2026); the induction-family line is secondary and weak until the composition ladder exists.
8. Reviewers will demand: all three optimizers vs. a fair floor, demo-selection-vs-instruction and CoT ablations, held-out-*seed* generalization, reachability/ceiling calibration, and per-seed variance/CIs.
9. NOT publishable: no gain / within-noise, gains that vanish on fresh seeds or reduce to demo-selection/CoT-toggling, an unreachable-ceiling task, a bare IB re-run, or an "LLMs can do induction" over-claim from one task.
10. Research currency is HIGH on venue/recency grounds (IB ACL 2025, MIR NeurIPS 2025 D&B, Falsify 2026, active optimizer line); the OpenAlex citation counts (fetched 2026-07-22) are near-zero-by-construction for 2025–2026 papers and undercount vs. Google Scholar, so they are a floor, not a signal.
