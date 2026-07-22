# Capability Synthesis — c23: Hidden-Rule String-Transform Induction Ladder

Scope: c23's reseed-only baseline is InductionBench-style subregular (ISL/OSL) string-transform
instances, regenerated with FRESH seeds using the existing InductionBench codebase, **single-rule
instances only** (no composition ladder, no invented multi-rule stacking), the existing
I/O-demos-plus-query format, short-string exact-match output. This synthesis reads the 6 verified
dossiers against that specific target and predicts a naive-prompt floor and ceiling-prompt score at
the cheap tier (Gemini-2.5-Flash / GPT-5-mini / Claude Haiku 4.5 / DeepSeek-Chat class), temp 0.

Dossier keys used:
- **IB** = `2502.15823` InductionBench (the direct design driver / codebase source)
- **MIR** = `2502.09933` MIR-Bench (many-shot ICL function I/O prediction)
- **PCFG** = `1908.08351` Compositionality Decomposed / PCFG SET (trained seq2seq, string-edit ops)
- **WILT** = `2410.10998` WILT (multi-turn Wason 2-4-6 boolean rule discovery)
- **FALSIFY** = `2606.04751` FalsifyBench (multi-turn WordNet rule-discovery games)
- **REARC** = `2404.07353` Re-ARC (procedural ARC data generator; no model eval)

A key caveat up front: **only IB directly targets c23's exact task and protocol** (subregular
string-transform induction, single forward pass). MIR and PCFG are strong analogues. WILT, FALSIFY,
and REARC are family relatives whose protocols (multi-turn / trained-from-scratch / no-model) differ
enough that their numbers are directional, not load-bearing, for c23's single-call design. Where I
lean on them I say so.

---

## Q1. Single forward pass — what models CAN and CANNOT do; where one-shot accuracy holds vs collapses

**What the evidence SHOWS (single-shot, on-target = IB and MIR):**

- IB is the on-point evidence and it is bleak. In one forward pass (zero-shot CoT) the model is given
  the function class, context window `k`, alphabet `Σ`, and a sample I/O set, and must emit a rule set.
  At the moderate setting **k=4, |Σ|=4, rules=3, sample=2x**, *compatibility collapses to 0.00 for all
  6 models except o3-mini* (o3-mini 10/10/30 for ISL/L-OSL/R-OSL) (IB Table 1, Sec 5.1).
- **Where one-shot accuracy HOLDS (IB):** only at genuinely easy settings. IB Appendix Table 5 / Table 8:
  ISL with **k=2, vocab=2, rules=1**, GPT-4o reaches compatibility 40 (recall 60, precision 43);
  Llama-3.3-70B reaches 60 at k=2 and 60/70/80 with 1/2/3-shot demos (IB Table 8). So single-rule,
  small-k, small-vocab ISL is the regime where non-frontier models get partial traction.
- **Where it COLLAPSES (IB):** `k` is the single largest difficulty driver — moving k from 2 to 4
  "markedly reduces recall, precision, and compatibility across all models" (IB Sec 5.2, "Impact of k").
  At k=4 essentially everything except o3-mini goes to 0 compatibility. Number of rules also collapses
  scores: "sometimes causing scores to plummet to near 0 when a second or third rule is added" at large
  k/|Σ| (IB Sec 5.2, "Impact of the Number of Rules"). |Σ| matters less than k.
- **MIR corroborates the single-pass ceiling** on the more general "predict y=f(x) from I/O demos"
  formulation: best model o1-mini <0.7 accuracy on MIR-Extended; most models (e.g. GPT-4o) <0.4
  (MIR Sec 4.1). No model saturates. MIR deliberately filtered out instances that are 0%-accuracy for
  all five factor-analysis models across all shot counts (MIR Sec 3.2), so even its retained pool has a
  reachable-but-low ceiling.
- **PCFG** (trained seq2seq, not prompting) gives the difficulty-shape prior: sequence accuracy is
  ~1.0 at low length/depth/#functions and decays smoothly as any of these rises (PCFG Sec 6.1.1, Fig 6);
  best model plateaus at 0.92 task / 0.72 systematicity / 0.50 productivity even with unlimited training
  (PCFG Table 1). Even a *trained* model does not reach ceiling on the harder axes.

**What I INFER:** For c23's single-rule, reseed-only slice, the "holds" regime (IB k=2, vocab=2,
rules=1 ISL) is exactly what c23 restricts itself to. The collapse regime (k=4, rules=3) is explicitly
excluded by "single-rule instances only." So c23's on-target evidence says: at the easy end of IB,
non-frontier models score in the tens-of-percent, not zero, and not near-ceiling.

---

## Q2. Multistep / interactive / many-shot — what breaks; does interaction / more demos help or hurt?

**More demonstrations in a single call (the axis c23 actually has — few-shot demo count):**

- **IB (on-target):** few-shot demos help *only* in simple settings. Compatibility rises 60→60/70/80
  with 0/1/2/3-shot at ISL/k=2, but "as complexity increases … the benefits of additional few-shot
  examples become negligible" (IB Sec 5.2, "Impact of Examples"; Sec 5.3). Worse: adding *more raw
  characteristic-sample data* (2x→5x) makes compatibility DROP steeply — "compatibility decreases
  steeply as the number of provided input-output examples increases," plummeting near zero at 4x-5x
  (IB Sec 5.2, "Robustness", Fig 5). Repeating the same sample (padding context) causes only a small
  decline (Fig 6), so it is *new information*, not context length, that overwhelms the model. This
  directly contradicts the theory (more characteristic-sample data should make the function *more*
  recoverable). **This is the single most important design warning for c23: more demos is not free;
  it can hurt.**
- **MIR (strong analogue):** many-shot (4→2048) saturates by ~256 shots, often well before context
  limits, attributed to attention dispersion, not retrieval failure (MIR Insight 1, Sec 4.1/4.3).
  So the demo-count lever has sharply diminishing then negative returns.

**Interaction / multi-turn (NOT c23's protocol, but the family's evidence on it):**

- **WILT:** even frontier models cap at 28% (Claude 3.5 Sonnet 14/50) on the full multi-turn split;
  reasoning models (o1) reason well per-turn but lose by gathering less evidence — "a multi-turn
  capabilities failure despite not being a failure on any turn in particular" (WILT Sec 4.1).
  Interaction introduces *new* failure modes absent from single-shot: doom loops (re-proposing tested
  cases) and confirmation bias (WILT Sec 4.1, Appendices B.1/B.3).
- **FALSIFY:** same story on WordNet games — best model 75%, no model near optimal; success is driven
  by falsification-seeking behavior, and the dominant failure is confirmation bias
  (rho=-0.779 with success) (FALSIFY Sec 4).
- **Does interaction help vs one-shot?** WILT's function-inversion ablation shows composing a
  multi-turn evidence-gatherer with a separate single-turn inverter beats any single model
  (o1-mini 19/50 given chatgpt-4o's stripped tests vs 14/50 best full-task) (WILT Sec 4.3) — i.e., the
  interactive burden *hurts* the model that must both gather and deduce. For c23 this is reassuring:
  by removing interaction, c23 strips out the doom-loop/confirmation-bias confounds and isolates pure
  induction.

**CoT (a within-call multistep lever that DOES apply to c23):**

- **MIR:** across all 21 models, **no-CoT (direct/transductive) accuracy is unanimously and
  significantly HIGHER than forced-CoT (inductive)** — o1-mini 0.696 vs 0.334; DeepSeek-R1 0.757 vs
  0.298; Claude-3.5-Sonnet 0.775 vs 0.585 (MIR Table 2, Insight 2). The gap widens with more shots.
  Exception: long-CoT "thinking" models tolerate CoT better. **For a cheap non-thinking tier at temp 0,
  this predicts direct-answer prompting will beat forced-CoT.** (Note IB itself used zero-shot CoT
  throughout and still got its low numbers — so CoT is not obviously helping there either.)

---

## Q3. Model-tier breakdown (cheap tier matters most)

**CONFLICT/CAVEAT:** No dossier tests c23's *exact* cheap tier (Gemini-2.5-Flash / GPT-5-mini /
Claude Haiku 4.5 / DeepSeek-Chat). The closest same-family, same-protocol proxies:

**Frontier reasoning tier (on-target IB, single-shot):**
- o3-mini is the only model with nonzero compatibility at IB's moderate setting (10/10/30, IB Table 1);
  on the full standard leaderboard still only **5.69% weighted compatibility** (33.93% under log-weight
  that dampens k=4's dominance) (IB Table 3/4, Sec 6.1). o1-mini second-best but 0.00 compatibility at
  the moderate setting (IB Table 1).
- MIR: o1-mini/o1-preview clearly top, but still <0.7 (MIR Sec 4.1).

**Cheap / mid tier — the directly relevant proxies:**
- **GPT-4o (proxy for GPT-5-mini-ish non-reasoning):** IB Table 1 compatibility 0.00/0.00/0.00 at
  moderate setting; recall only 10-17, precision 3-7. But at easy ISL (k=2, vocab=2, rules=1)
  compatibility 40 (IB Table 5). MIR: acc 0.540 (no-CoT) / 0.488 (CoT) on MIR-Core (MIR Table 2).
- **DeepSeek-V3 / DeepSeek-Chat class:** IB Table 1 compatibility 0.00/0.00/0.00 at moderate;
  recall 3-23, precision <3 (IB models). DeepSeek-R1-Distill-Llama-70B on the IB leaderboard:
  compatibility 0.12 linear / 8.63 log-weighted (IB Table 3/4). WILT: DeepSeek-v2.5-chat 6/50,
  v2-chat 3/50 (WILT Table 1). MIR (DeepSeek-R1): 0.757 no-CoT / 0.298 CoT (MIR Table 2) — but R1 is a
  reasoning model, so this over-states a plain DeepSeek-Chat.
- **Claude Haiku class:** two independent red flags. **MIR: Claude-3.5-Haiku "achieves surprisingly
  low accuracy" — it "often does not understand our prompt and sees the target input as part of an
  incomplete data, thus refusing to answer"** (MIR Sec 4.1). **WILT: Claude 3 Haiku 1/50**, flagged for
  failing to even use its turns (WILT Table 1). Haiku-class models show *prompt-comprehension/refusal*
  failure on I/O-block formats, independent of reasoning difficulty. This is a concrete parsing/format
  risk for c23's Haiku 4.5 runs.
- **Gemini Flash class:** WILT: Gemini 1.5 Flash 7/50, Flash-8b **0/50** (WILT Table 1). MIR notes
  Gemini-2.0 Flash as one of two models with anomalous behavior in the erroneous-shots study
  (MIR Sec 4.5).
- **Llama-3.3-70B (open, cheap-ish):** the *most encouraging* on-target datapoint — IB Table 8, ISL
  k=2, vocab=2, rules=1: compatibility 60→80 with more shots; but "near 0-10 regardless of shot count"
  at k=4 (IB Table 8).

**Tier takeaway:** Frontier reasoning models barely clear zero at IB's moderate difficulty; the cheap
tier is at or near the floor for anything beyond single-rule/small-k/small-vocab. In the easy ISL
regime (which is c23's target), cheap/mid models land in the ~40-60% compatibility band, with
Haiku/Flash carrying additional *format-refusal* risk that can drag them to single digits regardless of
difficulty.

---

## Q4. Prompting strategies & input representations that moved accuracy — with conditions

Ranked by magnitude of effect, on-target first:

1. **Context window `k` (IB):** k=2→k=4 is the largest single accuracy driver; collapses most models
   from partial success to 0 compatibility (IB Sec 5.2). *This is a task-difficulty dial, not a prompt
   trick, but it dominates everything.*
2. **No-CoT vs forced-CoT (MIR, large effect):** no-CoT beats forced-CoT for *all 21 models*, e.g.
   +0.36 for o1-mini (0.696 vs 0.334), +0.46 for DeepSeek-R1 (0.757 vs 0.298), +0.19 for
   Claude-3.5-Sonnet (0.775 vs 0.585) (MIR Table 2). Condition: single call, function I/O prediction.
   Non-thinking cheap models should benefit most from direct-answer prompting.
3. **Few-shot demo count (IB):** helps at easy settings (ISL k=2: 0→3-shot lifts Llama-3.3-70B
   60→80), negligible past a complexity threshold; and adding *new* sample data beyond minimal
   overwhelms (2x→5x drops toward 0) (IB Sec 5.2). Net: a *small* demo count is best; more can hurt.
4. **Output/rule notation (IB):** IB adopted "condition ∘ target → output" notation specifically
   because verbose functional descriptions are harder to generate/parse (IB Sec 3.5). No head-to-head
   accuracy delta reported, but chosen for parse robustness. For c23 (short-string exact-match output),
   the analogous lever is strict output-format enforcement + robust extraction (see MIR Appendix B.6:
   find last "Output:", strip markdown, ast.literal_eval — a reference extraction pattern).
5. **RAG shot selection (MIR):** embedding-based selection of the 64 most-similar shots vs random =
   *no significant difference* (MIR Table 5, Insight 5). Do not expect demo-selection cleverness to help.
6. **Meta-shot / out-of-domain CoT exemplars (MIR):** only marginal, inconsistent gains (MIR Table 6).
7. **Erroneous shots (MIR):** models robust even at 3/4 error rate (MIR Insight 3) — a noisy-demo dial
   would barely move accuracy.
8. **Strict JSON schema + retry (FALSIFY, WILT):** both interactive papers needed strict tagged/JSON
   output plus retry-on-malformed to reliably parse (FALSIFY App C.3; WILT App A.1.2). Relevant for
   c23's answer extraction robustness even in single-call mode.
9. **Representation format (JSON vs prose vs table):** *no paper ran this ablation.* MIR/IB fixed format;
   PCFG doesn't prompt. **This is an evidence gap** — c23 cannot cite prior work on I/O serialization
   format effects; it would have to measure them itself.

---

## Q5. Predicted naive-prompt floor & ceiling-prompt score for c23's reseed-only single-rule baseline (cheap tier, temp 0)

**Restating the exact target:** InductionBench-style subregular ISL/OSL, fresh seeds via the existing
IB codebase, **single-rule only**, existing I/O-demos-plus-query format, short-string exact-match.
Critically, c23's format is **demos + one held-out query, exact-match on the transformed string** —
which is EASIER to score and plausibly easier to succeed at than IB's own protocol (which grades a full
rule-set reconstruction via precision/recall/compatibility). So IB's compatibility numbers are a
*conservative-to-pessimistic* proxy: a model can sometimes transform one held-out string correctly
without articulating the complete minimal rule set. I flag this as the largest source of upward
uncertainty.

**Anchoring datapoints (single-rule, easy end, cheap/mid models):**
- IB ISL, k=2, vocab=2, rules=1: GPT-4o compatibility 40; Llama-3.3-70B 60 (0-shot) → 80 (3-shot)
  (IB Table 5, Table 8).
- IB at k=4, rules=3 (EXCLUDED by c23's single-rule restriction): 0 for these models.
- The reseed-only baseline stays at rules=1, and c23 controls k. If c23 uses **small k (2-3) and small
  vocab**, the anchor is the 40-80 compatibility band; if it inherits IB's harder default k=4, the
  anchor is ~0-10.

Predicted ranges (accuracy = exact-match on the held-out query string):

- **NAIVE-PROMPT FLOOR (bare task statement, no format scaffolding, direct query, temp 0):**
  **~5-25%.** Reasoning: (a) the floor is depressed by *format/comprehension* failures on the cheap
  tier — MIR's Claude-Haiku refusal behavior and WILT's Haiku 1/50 and Gemini-Flash-8b 0/50 show
  cheap models can score near zero purely from misreading the I/O block (MIR Sec 4.1; WILT Table 1);
  (b) but single-rule small-k ISL is genuinely partially solvable, and held-out-query exact-match is
  easier than full-rule reconstruction, pulling above zero. **Uncertainty is high and skewed low**:
  if c23 inherits IB's k=4 default or if the cheap models hit the refusal mode, the floor is ~0-10%;
  if k is small and format is legible, ~15-25%. Model spread will be large (Haiku/Flash likely near the
  bottom of the range; DeepSeek-Chat/GPT-5-mini nearer the top).

- **CEILING-PROMPT SCORE (best feasible prompt: explicit task framing, worked format, direct-answer /
  no forced-CoT, a small well-chosen demo count, strict output format, temp 0):**
  **~35-65%.** Reasoning: (a) the on-target easy-ISL anchor is 40-80 compatibility for GPT-4o /
  Llama-3.3-70B, and held-out-query exact-match should meet-or-exceed compatibility; (b) MIR shows
  no-CoT prompting alone can add ~0.1-0.4 over forced-CoT, and a good format removes the Haiku-style
  refusal loss; (c) but MIR also shows *no cheap model saturates* even the retained-instance pool, and
  IB's "more demos hurts" plus "few-shot negligible past threshold" cap how far prompt engineering can
  push. The ceiling will NOT approach ~100% at the cheap tier; a realistic best is high-30s to low-60s,
  with the upper end reachable only for the easiest (k=2, small-vocab, single ISL rule) instances and
  the stronger cheap models. **Uncertainty: moderate-high.** If c23's single-rule instances skew toward
  the very easiest IB configs, ceiling could touch ~65-70%; if they sit at IB's default hardness even at
  rules=1, ceiling may be stuck in the ~20-40% band.

**Floor-to-ceiling gap (the go/no-go signal):** predicted **~20-40 percentage points** of headroom at
the cheap tier. This is a *usable* dynamic range — big enough that prompt quality clearly separates,
but the whole band sits well below 100%, consistent with IB's "even the simplest complexity class" thesis
and MIR's "no model saturates." **Two design risks that would compress the range and argue for adding
complexity/curation:** (1) if reseed-only inherits IB's k=4/multi-rule default hardness, the entire band
collapses toward zero (no separation, no-go without easing difficulty); (2) if cheap-tier format-refusal
(Haiku/Flash) dominates, the floor and ceiling both crater for those models specifically — a
representation problem, not a difficulty problem, fixable by prompt format work before any complexity is
added. **Recommendation implied by the evidence (not a directive):** calibrate `k`, vocab size, and demo
count on a small pilot BEFORE freezing, exactly as IB and MIR both did (MIR excluded all-zero instances,
Sec 3.2; IB dialed k/|Σ|/rules), and fix the CoT axis to direct-answer for the cheap non-thinking tier.

**Conflicts / gaps surfaced:**
- IB and MIR agree "more evidence can hurt/saturate," but IB is starker (new data drops toward 0 by
  4x-5x) than MIR (plateau by ~256 shots). Both point the same direction for c23: keep demo count small.
- No dossier measured input *serialization format* effects — a genuine gap; c23 must measure this itself.
- No dossier tested c23's *exact* cheap models; all cheap-tier numbers are same-family proxies
  (GPT-4o for GPT-5-mini; Claude-3.5-Haiku / Claude-3-Haiku for Haiku 4.5; Gemini-1.5/2.0-Flash for
  2.5-Flash; DeepSeek-V3/-R1 for DeepSeek-Chat). Treat all point predictions as ranges, not values.
- IB's protocol (full rule-set reconstruction) is stricter than c23's (held-out-query exact-match), so
  IB compatibility numbers likely UNDER-predict c23 accuracy — the main reason the ceiling range extends
  above IB's raw compatibility figures.

---

## Executive summary (10 lines)

1. Only InductionBench (IB, 2502.15823) directly tests c23's task+protocol; MIR-Bench and PCFG are strong analogues; WILT/FalsifyBench/Re-ARC are family relatives on different (multi-turn / trained / no-model) protocols.
2. Single forward pass: models succeed only at easy subregular settings — IB ISL k=2/vocab=2/rules=1 yields ~40-60% compatibility for GPT-4o/Llama-3.3-70B; at k=4/rules=3 all-but-o3-mini collapse to 0 (IB Table 1/5/8).
3. c23's reseed-only slice is single-rule and restricts k, so it lives in IB's "partial-success" easy regime, not the zero-collapse regime.
4. Context window k is the dominant difficulty driver; rule count and adding NEW demo data both push accuracy toward zero (IB Sec 5.2) — more demonstrations is not free and can hurt.
5. MIR: no cheap model saturates; no-CoT beats forced-CoT for all 21 models (e.g. +0.36 o1-mini), so direct-answer prompting is favored for the cheap non-thinking tier (MIR Table 2, Insight 2).
6. Cheap-tier proxies are weak: GPT-4o/DeepSeek-V3 hit 0 compatibility at IB's moderate setting; Claude-Haiku and Gemini-Flash additionally show format-refusal failures (MIR Sec 4.1; WILT Table 1) that crater scores independent of difficulty.
7. Interaction (WILT 28% ceiling, FalsifyBench 75% ceiling) adds doom-loop/confirmation-bias confounds; c23's single-call design usefully removes these but inherits none of their help.
8. Predicted naive-prompt floor at cheap tier, temp 0: ~5-25% (skewed low; ~0-10% if k=4 inherited or Haiku/Flash refuse; ~15-25% if k small and format legible).
9. Predicted ceiling-prompt score: ~35-65% (upper end only for easiest k=2 single-ISL instances and stronger cheap models; no cheap model approaches 100%). IB's rule-reconstruction metric under-predicts c23's easier held-out-query exact-match, biasing the ceiling upward.
10. Go/no-go: a ~20-40pt floor-to-ceiling gap is usable IF difficulty is curated (small k/vocab, small demo count, direct-answer prompting) and format-refusal is fixed first; if reseed-only inherits IB's default hardness, the band collapses to zero and the task needs added easing (not added complexity) to separate — calibrate on a pilot before freezing, as both IB and MIR did.
