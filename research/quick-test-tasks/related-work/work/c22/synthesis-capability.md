# Capability Synthesis — c22 · Stacked Verifiable-Constraint Micro-Generation

Scope note. This synthesis answers the five capability questions for c22 **as scoped to its reseed-only baseline**: standard IFEval checker atoms ONLY (reuse the `google-research` IFEval checker library as-is), no OOD/invented atoms, no conditional Selection constraints yet — 3–5 atoms sampled fresh per instance over trivial micro-tasks, all-checkers-pass 0/1 scoring. This is a critical framing: the reseed-only baseline **removes exactly the two mechanisms** (Selection/discovery and OOD atoms) that carry c22's anti-enumeration story, so the evidence that applies most directly is the IFEval/IFBench-in-domain / VFF / FollowBench body of work, where every constraint is stated plainly in the prompt.

Evidence dossiers (11): `2311.07911-ifeval`, `2507.02833-ifbench`, `2307.08689-collie`, `2310.20410-followbench`, `2401.03601-infobench`, `2407.03978-complexbench`, `2502.04498-vff`, `2505.16234-lifebench`, `2404.13208` (instruction hierarchy), `2603.04738-if-rewardbench`, `2604.04443-deonticbench`.

Throughout, **[SHOWS]** = directly measured in a dossier; **[INFER]** = my reasoning from the evidence (flagged with its uncertainty). A recurring limitation is stated up front: **almost no dossier reports numbers for the exact cheap tier c22 runs at (Gemini-2.5-Flash / GPT-5-mini / Claude Haiku 4.5 / DeepSeek-Chat class) on a strict all-pass, deterministic-checker, micro-task setup.** The cheap-tier estimates in Q3/Q5 are therefore interpolations, and I say so.

---

## Q1 — What models CAN and CANNOT do in a SINGLE forward pass

### What the evidence SHOWS

The whole corpus is dominated by single-shot generation (IFEval, IFBench single-turn, COLLIE zero-shot, FollowBench, InFoBench, ComplexBench, VFF, LIFEBench are all single-forward-pass on the generation side). The consistent finding: **models can satisfy one stated verifiable constraint reasonably well, but strict all-pass accuracy collapses as independent constraints are stacked, and the collapse is present at every model tier.**

Concrete single-pass, strict-all-pass numbers most relevant to c22 (all deterministic checkers, constraint stated in prompt):

- **IFEval** (`2311.07911-ifeval`, Table 3): 1–3 stacked verifiable atoms, prompt-level **strict** accuracy (= all atoms pass, exactly c22's scoring). GPT-4 **76.89%**; PaLM 2 Small **43.07%**. Instruction-level (per-atom) is higher (GPT-4 83.57%, PaLM-2-S 55.76%) — the paper's own numbers thus demonstrate the all-pass compression effect directly (§3, metrics 1–2). IFEval's construction range is exactly 1–3 atoms per prompt (§2.1).
- **VFF** (`2502.04498-vff`, Table 2): strict conjunctive Python-checker scoring (`I = ∏ F_k`), levels = 1/2/3 stacked constraints. GPT-4-turbo: **76.29 / 53.33 / 35.31** (L1/L2/L3). GPT-3.5: **62.93 / 34.07 / 16.40**. 7B baselines (Mistral-7B, LLaMA-2-7B, LLaMA-3-8B) sit at L1 ≈ 50–60, L3 ≈ 9–16 (Sec 4.2). This is the cleanest analog to c22: deterministic checkers, strict AND, trivial-to-moderate base task, count as difficulty dial.
- **COLLIE** (`2307.08689-collie`, Sec 1/5.1): compositional count/position constraints, deterministic parser, binary all-pass. GPT-4 zero-shot **50.9%** average; GPT-3.5 far lower (pass@20 32% vs GPT-4 >63%).
- **ComplexBench** (`2407.03978-complexbench`, Table 5): GPT-4-1106 partial-credit DRFR = 0.800 overall but **fails ~20% of complex instructions**; on the stricter all-branches-correct Coherent Test even GPT-4 hits only **14.9%** on nested Selection (Sec 5.2.3).

Difficulty/size collapse pattern **[SHOWS]**:
- **Per-atom pass probability ≈ 0.75.** IF-RewardBench (`2603.04738`, Sec 3.3): 74.6% of (response, constraint) pairs are annotated "followed" across a mix of real LLM responses. This is a rare direct anchor for a *single* stated constraint's base pass rate.
- **Constraint category matters, not just count.** Numeric/counting and length atoms are systematically the hardest deterministic category across InFoBench (`2401.03601`, Fig 3: lowest on Number/Linguistic), ComplexBench (Length only 0.532 for the best model, Appendix I.2), IFBench (`2507.02833`, Table 2: words/sentence/count categories stay 48–71 even after targeted RL, vs casing/format 90+), and IF-RewardBench (Numeric/Format easiest to verify, Situation/Style hardest). Casing, format-wrapper, keyword-presence, and start/end-token atoms are the **easy** deterministic category.
- **Exact-match constraints are far harder than one-sided/loose ones.** COLLIE (exact char count > range > bound, Sec 5.2); LIFEBench (`2505.16234`, Sec 5.1): "Equal To" length is much harder than "At Most"/"At Least" (23/26 models < 60 LS on Equal-To; 19/26 > 90 on At-Most).

### What I INFER

- **[INFER, high confidence]** For c22's reseed-only baseline at 3–5 IFEval atoms, single-pass **strict all-pass accuracy will be substantially below the per-atom rate**, because strict AND multiplies failure. If atoms were independent with p≈0.75 each (IF-RewardBench anchor), naive expectation is 0.75³≈0.42, 0.75⁴≈0.32, 0.75⁵≈0.24. IFEval's own GPT-4 gap (76.89 prompt-level vs 83.57 inst-level at only 1–3 atoms) confirms the direction. **Independence is an over-pessimistic assumption** — a capable model that "gets" the format task tends to satisfy several easy atoms jointly (correlated successes), so real numbers for strong models will beat the naive product; but for weak models errors are also correlated (a model that ignores formatting misses several atoms at once), which can push below the product.
- **[INFER, high confidence]** c22's *trivial base task* ("name a color", few-words output) removes the content-generation confound and, importantly, **neutralizes the single biggest failure mode in this literature** — long-form length/counting collapse (LIFEBench's entire story is about long outputs; c22's output is tiny, so an exact-word-count atom on a 2-word answer is trivially satisfiable, unlike LIFEBench's 8,192-word targets). This makes c22 *easier* than VFF/IFEval on the length/count axis specifically.
- **[INFER, medium]** Where c22 stays hard even at micro-scale: forbidden-letter atoms (must avoid a letter across the whole short output), exact-word-count-equals-N (exact, not bounded), and joint satisfaction under strict AND. These are the atoms that will drive the floor.

**Bottom line Q1:** Single pass — models CAN satisfy 1–2 stated deterministic atoms on a trivial task at high rates; they CANNOT reliably satisfy 4–5 simultaneously under strict all-pass, and this holds from frontier down. The exact ceiling depends heavily on *which* atoms are sampled (casing/format easy, exact-count/forbidden-letter hard).

---

## Q2 — MULTISTEP / interactive / many-shot: what breaks, does interaction help?

### What the evidence SHOWS

The corpus is unusually consistent and unusually negative here: **more demonstrations, decomposition, or feedback loops do not reliably rescue constraint-following, and several forms actively hurt.**

- **Few-shot / one-shot demonstrations do NOT help (and can hurt).**
  - COLLIE (`2307.08689`, Appendix C.1): zero-shot vs one-shot statistically indistinguishable (GPT-4 40.7→39.4; GPT-3.5 23.1→23.6, on a pre-release internal set) — authors conclude the bottleneck is generation-under-constraint, not instruction comprehension.
  - InFoBench (`2401.03601`, Appendix A.2): adding few-shot examples to the judge prompt gave "no significant improvement."
  - FollowBench (`2310.20410`, §4.3): the "Example" constraint category (a 5-shot ICL task by design) is the *hardest* category — more/noisier demonstrations degrade rather than help; GPT-4 avg only 50.7 there vs 76–93 elsewhere.
- **Instruction decomposition into multi-turn execution HURTS.** ComplexBench (`2407.03978`, Table 6, Sec 5.2.3): manually decomposing composed instructions into step-by-step turns dropped GPT-3.5 DRFR 0.682→0.652, worse on more-composed categories, attributed to "cumulative errors in multi-round interactions." Explicit conclusion: "cannot be simply solved via instruction decomposition."
- **Iterative feedback helps early then plateaus, and does not solve.** COLLIE (`2307.08689`, Sec 5.2, Fig 7): with per-turn natural-language feedback stating which constraints failed, GPT-4 gains ~20% after round 2 then **plateaus at 66%** through rounds 3–4 — comparable to plain pass@5. Some constraint types stay unsolved regardless.
- **Internal reasoning / extended thinking only partially helps.** LIFEBench (`2505.16234`, Sec 5.3, Appendix I): reasoning models self-count and revise inside their trace, but this "only partially alleviates the problem for short length constraints and still fails under longer constraints." IF-RewardBench (`2603.04738`, Table 5): for *judges*, disabling thinking hurts (GLM-4.6 −32.7%) and self-consistency Maj@K helps but saturates beyond K=7.
- **IFBench multi-turn** (`2507.02833`, Table 7): "multi-turn" = constraint deferred to turn 3 with dialogue history; single-turn-trained models lose ~20–30 points when evaluated multi-turn. Only training on a mix of both formats balances them. This is about *training exposure to turn structure*, not inference-time iteration helping.

### What I INFER

- **[INFER, high confidence]** For c22's reseed-only baseline, which is defined as a **single LLM call**, the multistep literature says: do not expect that adding demonstrations or an interaction loop would raise the ceiling much, and decomposition would likely lower it. The one lever the literature shows *does* work is **repeated sampling (pass@k)** and **native model reasoning within one call** — both of which are properties of the model/decoding, not of the task design.
- **[INFER, medium]** Because c22 logs per-checker diagnostic verdicts, a hypothetical feedback variant would look like COLLIE's feedback loop — expect a one-shot ~15–20pp bump then a plateau well short of 100%, not a solve. This is relevant if the go/no-go later considers adding a reflection loop.

**Bottom line Q2:** Interaction/demonstrations/decomposition do NOT reliably help and often hurt; the only reliable multistep gains come from repeated sampling and native reasoning within a single call. This *supports* c22's single-call design but also warns that the task will not be "fixed" by prompting scaffolds — its difficulty is intrinsic to strict compositional constraint-following.

---

## Q3 — Breakdown by model tier (cheap tier matters most)

### What the evidence SHOWS — frontier reasoning tier

- Even the strongest models are far from ceiling on stacked strict constraints: GPT-4 IFEval prompt-strict **76.89%** at 1–3 atoms (`2311.07911`); GPT-4-turbo VFF L3 **35.31%** (`2502.04498`); GPT-4-1106 ComplexBench fails ~20%, nested-Selection Coherent 14.9% (`2407.03978`); IFBench frontier models (GPT-4.1, Claude 3.7/4 Sonnet, Qwen3-32B) score **<50%** on OOD constraints (`2507.02833`, Sec 1). Reasoning models lead length-following (o3-mini best on LIFEBench, `2505.16234`) but still <76 LS on exact-length.

### What the evidence SHOWS — cheap tier (sparse, indirect)

Direct data on the exact cheap tier is thin. The closest anchors:

- **GPT-3.5-turbo** (a reasonable proxy for the cheap tier's capability band): IFEval-adjacent numbers not reported for GPT-3.5 in `2311.07911`, but VFF (`2502.04498`, Table 2) gives GPT-3.5 **62.93 / 34.07 / 16.40** at L1/L2/L3 strict; COLLIE pass@20 32% vs GPT-4 63% (`2307.08689`); FollowBench CSL 2.9 vs GPT-4 3.3 (`2310.20410`); InFoBench 86.7 DRFR (`2401.03601`, partial-credit, not strict).
- **Gemini-Pro** (older): InFoBench 85.6 DRFR partial-credit (`2401.03601`); **Gemini 2.5 Flash** appears by name only in DeonticBench (`2604.04443`) on a *different* (deontic rule-reasoning) task where it is weak, and in LIFEBench's model family discussion — not on a clean stacked-IFEval-atom strict test.
- **DeepSeek-V3 / GPT-5-mini / Claude Haiku 4.5** do not appear on a strict-all-pass stacked-IFEval-atom test anywhere in these 11 dossiers. GPT-5-mini appears only as a *judge* in IF-RewardBench (`2603.04738`, τ_b 0.211 on a hard subset) — not as a generator on c22-like tasks.
- **PaLM-2-Small** (`2311.07911`): IFEval prompt-strict **43.07%** at 1–3 atoms — a concrete small/cheap-model floor on exactly c22's scoring, and the single most transferable cheap-tier datapoint in the corpus.
- **7B–8B open models** (VFF `2502.04498`, IFBench `2507.02833`, FollowBench `2310.20410`, InFoBench `2401.03601`): consistently L1 ≈ 50–65, L3 ≈ 9–16 strict; CSL ≈ 1.4–2.4 (≈2 constraints reliably); IFBench pre-RLVR 16–31.

### What I INFER

- **[INFER, medium-high]** Today's cheap-tier hosted models (Gemini-2.5-Flash, GPT-5-mini, Claude Haiku 4.5, DeepSeek-Chat) are **meaningfully stronger at instruction-following than GPT-3.5-turbo / PaLM-2-Small and than 7B open models**, but weaker than frontier reasoning models. On a strict all-pass stacked-IFEval-atom micro-task I place them **between GPT-3.5-turbo and GPT-4** — i.e., closer to GPT-4 on easy atoms (casing/format/keyword) and closer to GPT-3.5 on the hard atoms (exact-count/forbidden-letter) and under deep stacking. Uncertainty is real because none of these models is directly measured on this setup here.
- **[INFER, medium]** The cheap tier will show the **steepest count-driven decay** of the deployable tiers: expect near-frontier at 1–2 atoms, then a sharper drop than frontier as count rises to 4–5, mirroring the VFF GPT-3.5 curve (63→34→16) more than the GPT-4 curve (76→53→35).

**Bottom line Q3:** Frontier ≫ cheap ≫ 7B-open, but all three collapse under stacking. The cheap tier is the decisive band for c22 and is exactly where the corpus is weakest on direct evidence — the estimates below are interpolations anchored on GPT-3.5/PaLM-2-S/7B data, not measurements.

---

## Q4 — Prompting strategies and input representations that moved accuracy

### What the evidence SHOWS (with exact conditions)

Most "representation" results in this corpus are on the **judge/scoring side**, not the generator side. c22 uses a deterministic code oracle, so judge-side findings are largely moot for its scoring but do inform whether an LLM-judge fallback is ever advisable (answer: no).

Generator-side, stated-constraint prompting:
- **Few-shot / one-shot exemplars: ≈0 effect on generation.** COLLIE zero→one-shot GPT-4 40.7→39.4, GPT-3.5 23.1→23.6 (`2307.08689`, Table 2). [Condition: one fixed I/O example per structure, pre-release set.] FollowBench Example-category (5-shot by design) is the worst category (`2310.20410`, §4.3).
- **Constraint parameter range (train-time) swings generalization.** IFBench (`2507.02833`, Sec 4.3): training on a "wider range" of constraint values ≈ or beats exact-test-range; a fully disjoint ("different") range consistently underperforms. **Implication for c22's generator: sample atom parameters (word counts, keywords, letters) from a wide range, not one narrowly matched to eval.**
- **Training-time stacking depth is non-monotonic; ~3 is a sweet spot.** IFBench (`2507.02833`, Table 1): training on 3 stacked constraints gave best OOD IFBench (59.5); non-monotonic thereafter (49.4 at 4, 55.8 at 5, dips at 6+). IFEval in-domain peaked at 6.
- **Decoding temperature has a mid-range "sweet spot."** FollowBench (`2310.20410`, §5.4) qualitative; exact deltas not extractable. Most benchmarks use temp 0 / greedy for determinism (FollowBench, InFoBench, ComplexBench, IFBench).
- **VFF one-shot negative-demonstration for sampling** (`2502.04498`, Table 4): showing a self-generated *wrong* response before resampling raised correct-yield 55.14%→78.80% cumulative — but this was for *training-data collection*, not evaluation-time prompting, and per-round yield decayed sharply (70.4%→2.5% by round 4).

Judge-side (relevant only if an LLM judge were ever used — c22 does not):
- Showing the constraint checklist / evolution path to a judge helps a lot: FollowBench multi-level-aware judge 88% vs 67% holistic (`2310.20410`, Table 3); ComplexBench RAL rule-augmented 95.36% vs 62.02% on rule-defined questions (`2407.03978`, Table 4); IF-RewardBench: judges do better with an explicit checklist than raw instruction (`2603.04738`, Sec 4.2).
- **Deterministic Python checking dominates LLM judging outright.** VFF (`2502.04498`, Tables 6–7): Python 100% acc / free / ~100× faster vs GPT-4o 70%, GPT-4o-mini 59%, with **25–52% self-inconsistency even at temp 0.1**. IFEval strict/loose split (`2311.07911`) exists precisely because naive checking has false negatives (markdown, intro/outro lines).

### What I INFER

- **[INFER, high]** For c22's reseed-only baseline, the levers that will move the *measured* score are (a) **which atom types get sampled** (casing/format/keyword vs exact-count/forbidden-letter) — the single largest driver, per every category-difficulty result; (b) **constraint count** (the difficulty dial, monotone-ish downward); (c) **strict vs loose checking** — IFEval shows strict is ~2–4pp below loose, i.e. a chunk of "failures" on real outputs are formatting artifacts (bold, intro lines). **c22 uses strict-only, so it inherits IFEval's false-negative risk without the loose mitigation** — a naive prompt's floor will be a few points lower than it "should" be purely from formatting incidentals.
- **[INFER, high]** Prompt exemplars / few-shot will NOT be the ceiling-prompt lever here; the ceiling comes from an *engineered instruction* that explicitly enumerates each stated atom and its exact parameter, plus format hygiene (no markdown, exact output) — because in the reseed-only baseline every constraint IS stated (no discovery), so a well-drafted prompt can in principle enumerate all of them.

**Bottom line Q4:** The accuracy movers are atom-type mix, count, and strict-vs-loose — not demonstrations. Deterministic checking is strongly vindicated (never use an LLM judge). c22's strict-only choice mildly depresses scores via formatting false-negatives.

---

## Q5 — Predicted naive-prompt FLOOR and ceiling-prompt SCORE for the reseed-only baseline (cheap tier, temp 0)

**Setup being predicted:** standard IFEval checker atoms only (no Selection, no OOD atoms), 3–5 atoms sampled fresh per instance, trivial micro-task, strict all-pass 0/1, single call, cheap tier (Gemini-2.5-Flash / GPT-5-mini / Claude Haiku 4.5 / DeepSeek-Chat class), temperature 0.

### Anchoring evidence (all [SHOWS])
- IFEval prompt-strict at 1–3 atoms: GPT-4 76.89%, PaLM-2-S 43.07% (`2311.07911`).
- VFF strict at L3 (3 atoms): GPT-4-turbo 35.31%, GPT-3.5 16.40%, 7B ≈ 9–16% (`2502.04498`).
- Per-single-constraint base pass ≈ 0.75 (`2603.04738`).
- Category effect: casing/format/keyword easy (90+ even for mid models); exact-count/forbidden-letter/length hard (`2401.03601`, `2407.03978`, `2507.02833`).
- Trivial micro-task removes the length/counting-collapse confound that dominates VFF/LIFEBench (`2505.16234`) — a c22-specific *easing* factor.

### Reasoning to the estimate [INFER]

Two opposing adjustments from the VFF/IFEval anchors:

**Easing factors (push c22 above VFF-L3 / IFEval-small-model numbers):**
1. Micro-task output makes any length/word-count atom near-trivially satisfiable (a 2-word answer trivially meets "≤ N words"; exact-count still needs care but on a tiny target). This is the biggest single mover and is *not* present in VFF (open-ended Alpaca answers) or IFEval (essays/poems).
2. Cheap-tier 2026 models are stronger than GPT-3.5/PaLM-2-S/7B, the closest measured proxies.
3. No content-quality confound; the model can spend all its "attention budget" on constraints.

**Hardening factors (push c22 below the easy reading):**
1. Strict AND over **4–5** atoms (VFF/IFEval mostly test ≤3); each added atom multiplies failure. IFBench's own eval set is only 1–2 atoms *because* more is hard.
2. Strict-only checking (no loose mitigation) costs a few points of formatting false-negatives.
3. If the atom sampler includes the hard category (exact-word-count-equals-N, forbidden-letter across the whole output), those atoms cap the joint.
4. Cheap tier decays faster with count than frontier.

**Synthesis.** The naive prompt (states the atoms in plain concatenated prose, no format hygiene, no per-atom emphasis) behaves like the "prompt lists only the stated atoms" miss case in c22's own example. With per-atom compliance for a *decent* model in the ~0.80–0.90 range on easy atoms and ~0.55–0.75 on hard atoms, and 3–5 atoms with partial positive correlation, the joint lands in the low-to-mid range.

### Predicted numbers (cheap tier, temp 0, strict all-pass)

| Condition | Floor (naive prompt) | Ceiling (engineered prompt) |
|---|---|---|
| **3 atoms**, easy-skewed mix | 0.35 – 0.55 | 0.70 – 0.88 |
| **3–5 atoms**, mixed atom types (expected default) | **0.20 – 0.45** | **0.55 – 0.80** |
| **5 atoms**, includes 1+ hard atom (exact-count / forbidden-letter) | 0.08 – 0.25 | 0.40 – 0.65 |

**Point estimates for the expected default (3–5 mixed atoms):**
- **Naive-prompt floor ≈ 0.25–0.40** (central guess ~0.30). Rationale: between VFF-L3 GPT-3.5 (0.16, but that's open-ended output + only 3 atoms) and IFEval-1–3-atom small-model band (0.43), nudged up by the micro-task easing but down by the 4–5-atom strictness and formatting false-negatives.
- **Ceiling-prompt score ≈ 0.60–0.80** (central guess ~0.70). Rationale: an engineered prompt that enumerates each *stated* atom with its exact parameter and enforces output hygiene should approach a strong model's per-atom competence, since **nothing is hidden in the reseed-only baseline** — the ceiling is limited mainly by the hard atoms and residual count decay, not by discoverability. Anchored near GPT-4's IFEval-small-atom 0.77 but held down for cheap tier and 4–5 atoms.

### Uncertainty and the go/no-go implication

- **Uncertainty is HIGH and asymmetric.** No dossier measures the exact cheap tier on this exact setup; the estimate rests on GPT-3.5/PaLM-2-S/7B proxies plus a micro-task easing adjustment that is *inferred, not measured*. The floor could be as low as ~0.10 (if the sampler is hard-atom-heavy and cheap models decay fast) or as high as ~0.50 (if easy-atom-skewed and cheap models are near-frontier on formatting). The ceiling is more stable (0.55–0.85).
- **Key go/no-go signal:** the reseed-only baseline is **explicitly missing the Selection/discovery atoms and OOD atoms** — the two mechanisms c22's own verdict says carry the entire anti-enumeration load (`candidates/c22.md` Verdict; corroborated by the fact that in the reseed-only setup *every constraint is stated*, so a ceiling prompt can enumerate all of them, exactly the "one clever sentence closes the ladder" risk). Therefore:
  - **[INFER, high]** A **naive-to-ceiling gap of ~0.30–0.45** (0.30 floor → ~0.70 ceiling) is plausible and would give a usable optimization signal — this is a real, exploitable gap.
  - **BUT [INFER, high]** most of that gap is closable by a single well-drafted enumerate-the-atoms prompt, because there is nothing to discover. That is the crux: the reseed-only baseline likely produces a **wide but shallow / one-shot-closable** gap. Once a prompt enumerates the stated atoms and adds format hygiene, the incremental ladder flattens — precisely the criterion-4 failure mode c22 is designed to guard against, and which the reseed-only baseline (by construction) does NOT guard against.
  - **This predicts the reseed-only baseline will look promising on floor-vs-ceiling spread but weak on *sustained* optimization difficulty** — pointing toward "needs added complexity" (the Selection/discovery and OOD atoms) unless the count/hard-atom strata alone are found to hold up a residual gap.

**Bottom line Q5:** Cheap tier, temp 0, strict all-pass, 3–5 stated IFEval atoms: **naive floor ~0.25–0.40, ceiling ~0.60–0.80** (both wide, high uncertainty, cheap-tier is interpolated). The gap is real but likely one-shot-closable because the reseed-only baseline strips the anti-enumeration mechanisms — evidence leans toward the task needing the Selection/OOD complexity to be a durable target.

---

## Conflicts and caveats surfaced

1. **Direct cheap-tier evidence is essentially absent.** Q3/Q5 cheap-tier numbers are interpolations from GPT-3.5-turbo, PaLM-2-Small, and 7B-open proxies. Flagged repeatedly; not a dossier conflict but the largest evidence gap.
2. **Strict vs partial-credit scoring is inconsistent across the corpus.** IFEval prompt-strict and VFF `∏F_k` are true all-pass (match c22); InFoBench DRFR, ComplexBench DRFR, IFBench category scores, LIFEBench LS are partial-credit/graded. All cross-benchmark accuracy comparisons must respect this — partial-credit numbers (e.g., ComplexBench 0.800, InFoBench 89) are **not** comparable to c22's strict 0/1 and would overstate expected c22 scores. I used only the strict/all-pass anchors (IFEval, VFF-`∏F_k`, COLLIE) for the Q5 estimate.
3. **COLLIE one-shot numbers are from a pre-release internal set**, not COLLIE-v1 (`2307.08689` Appendix C.1 correction) — used only for the qualitative "one-shot doesn't help" claim, not as absolute anchors.
4. **Selection difficulty evidence (ComplexBench, IF-RewardBench) is about *stated* conditions**, not c22's *hidden/discover* conditions — the papers do not test the hidden-property variant (both dossiers' `corpus_claim_verdict` flag this). So they support "Selection composition is hard" but do NOT directly validate c22's discovery mechanism; that remains an inference.
5. **The reseed-only baseline (this task's scope) deliberately excludes Selection and OOD atoms**, so several dossiers' most c22-relevant mechanisms (IFBench OOD generalization <50%, ComplexBench Selection) inform the *fuller* c22, not the reseed-only floor/ceiling being predicted. I kept these separate.

---

## Executive summary (10 lines)

1. Across 11 dossiers, single-pass models reliably satisfy 1–2 stated deterministic constraints on a trivial task but collapse under strict all-pass as constraints stack to 4–5 — at every tier.
2. Hard anchors for c22's exact scoring: IFEval prompt-strict (all-pass) GPT-4 76.9% / PaLM-2-Small 43.1% at 1–3 atoms; VFF strict at 3 atoms GPT-4-turbo 35% / GPT-3.5 16% / 7B ~9–16%.
3. Per single stated constraint, base pass rate ≈ 0.75 (IF-RewardBench); casing/format/keyword atoms are easy (90+), exact-count/forbidden-letter/length atoms are the hard floor-drivers.
4. Multistep does NOT rescue: few-shot ≈ 0 effect (COLLIE), decomposition hurts (ComplexBench), feedback plateaus ~66% (COLLIE); only repeated sampling and native reasoning within one call help.
5. c22's trivial micro-output uniquely neutralizes the length/counting-collapse that dominates VFF/LIFEBench, easing it relative to those benchmarks on the count/length axis.
6. Cheap tier (Gemini-2.5-Flash/GPT-5-mini/Haiku-4.5/DeepSeek class) is NOT directly measured anywhere here; estimates interpolate between GPT-3.5/PaLM-2-S/7B and GPT-4 — the corpus's biggest gap.
7. Predicted reseed-only baseline, cheap tier, temp 0, strict all-pass, 3–5 stated IFEval atoms: naive-prompt floor ≈ 0.25–0.40, ceiling-prompt ≈ 0.60–0.80 (both wide, high uncertainty).
8. Accuracy movers are atom-type mix, constraint count, and strict-vs-loose (c22's strict-only choice costs a few points to formatting false-negatives) — not demonstrations; deterministic checking beats LLM judging outright (VFF: Python 100% vs GPT-4o 70%).
9. Go/no-go signal: the reseed-only baseline strips Selection/discovery and OOD atoms — the two anti-enumeration mechanisms — so its floor-to-ceiling gap is real but likely one-shot-closable by a single enumerate-the-atoms prompt.
10. Recommendation lean: reseed-only will show a promising spread but shallow, one-shot-closable optimization difficulty; evidence points toward needing the Selection/OOD complexity for a durable target unless count/hard-atom strata alone hold a residual gap.
