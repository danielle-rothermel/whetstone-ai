# c22 Positioning Synthesis — Stacked Verifiable-Constraint Micro-Generation

Scope: positions c22 (as designed) against the 11 related-work papers in
`work/c22/dossier-*.json`, using the candidate spec in `candidates/c22.md`, the trend
scan `trends-2025-2026.md` (esp. area 9, prompt-optimization benchmarking), and the
OpenAlex citation data in `papers/manifest.json`.

**Convention throughout: EVIDENCE = what a dossier/paper/manifest states (cited to
source). JUDGMENT = my inference for c22. The two are labeled separately in every
section.**

c22 as-designed, restated for reference: a deliberately trivial base micro-task
("name a color"), gated by 3–5 composed, deterministically-checkable constraints drawn
from the IFEval/IFBench checker library, scored strict all-pass 0/1, with per-checker
verdicts logged as diagnostics; a synthetic seeded generator for an infinite
contamination-resistant pool; and one distinguishing constraint type — conditional
**Selection** ("if input has property P obey rule A else B") that must be *discovered,
not copied*. The **first run is an un-optimized baseline** using **Standard IFEval
checker atoms ONLY** (google-research IFEval library reused as-is; no OOD/invented
atoms, no Selection constraints yet): 3–5 atoms sampled fresh per instance over trivial
micro-tasks, all-checkers-pass 0/1 scoring. Optimization (COPRO/MIPROv2/GEPA) is a
later, separate step.

---

## 1. Positioning map

I organize the family on three axes that the dossiers repeatedly distinguish papers by.

### Axis A — Verifier type (how a response is scored)
- **A1 Deterministic code, per-atom** — Python/checker functions, no LLM judge.
- **A2 Hybrid (code + LLM judge)** — deterministic where possible, LLM judge for
  open-ended constraints.
- **A3 LLM-as-judge / model-based** — scoring is itself a model call.
- **A4 External symbolic execution** — a real solver runs generated code.

### Axis B — Constraint/rule provenance & shape (how the governing rule reaches the model)
- **B1 Stated atoms** — every constraint written verbatim in the prompt (concatenated).
- **B2 Stated composition** — constraints plus an explicit composition structure
  (And/Chain/Selection), condition still stated in-prompt.
- **B3 Hidden/discover** — the governing rule (or a branch condition) is deliberately
  NOT stated and must be inferred.
- **B4 Source-priority** — which instruction governs is decided by message
  provenance/authority (system > user > tool).

### Axis C — Output shape & scoring aggregation
- **C1 Micro-output, strict all-pass 0/1** — few words, instance scored 1 iff every
  atom passes.
- **C2 Free-form output, strict all-pass (prompt-level)** — long generation, 0/1 per
  instance requiring all atoms.
- **C3 Free-form output, partial-credit / graded** — DRFR ratio, macro-F1, HSR/SSR,
  continuous length score, or ranking.

(A fourth axis — single-shot vs. multi-turn generation — is nearly degenerate for this
family: **every one of the 11 papers is single-shot on the generation side.** Where
"multi-turn"/"multi-step" appears it is (i) an evaluation-side judge protocol
[InFoBench, IF-RewardBench], (ii) a difficulty ladder across separate instances
[FollowBench], (iii) a training curriculum [VFF, IFBench IF-RLVR], or (iv) a diagnostic
ablation that *hurt* performance [ComplexBench, COLLIE feedback plateau]. I note it per
paper rather than as a live positioning axis, because c22 is also single-shot.)

### EVIDENCE — placement table

| Paper (dossier) | A: verifier | B: provenance/shape | C: output/scoring | Cited-by (OpenAlex, 2026-07-22) |
|---|---|---|---|---|
| IFEval (2311.07911) | A1 deterministic, 25 checkers | B1 stated atoms (1–3) | C2 free-form, prompt-level strict all-pass (+loose) | 30 |
| COLLIE (2307.08689) | A1 deterministic parser (count/pos) | B1/B2 stated, AND/OR only, no conditional | C2 word→passage, True/False all-hold | 1 |
| FollowBench (2310.20410) | A2 hybrid (rule for closed, GPT-4 judge for open) | B1 stated, incremental level 1→5 | C3 HSR/SSR/CSL (partial + all-pass HSR) | 1 |
| InFoBench (2401.03601) | A3 LLM-as-judge, per-atom YES/NO | B1 stated (decomposed post-hoc) | C3 DRFR ratio (partial credit) | 1 |
| ComplexBench (2407.03978) | A2 hybrid RAL (~17% rule, ~83% judge) | B2 And/Chain/**Selection**/Nested, condition STATED | C3 DRFR w/ dependency aggregation | 0 |
| VFF (2502.04498) | A1 deterministic Python, ~60 meta-constraints | B1 stated atoms, level 1–3 | C2/C1 free-form, strict AND (∏Fₖ) | 0 |
| LIFEBench (2505.16234) | A1 deterministic word/char counter | B1 stated (single length atom) | C3 continuous Length Score 0–100 | 0 |
| IFBench (2507.02833) | A1 deterministic, 58 OOD + 29 train checkers | B1 stated atoms (1–2 per eval instance) | C2 free-form, strict/loose prompt-level | 1 |
| Instruction Hierarchy (2404.13208) | A3/heuristic (GPT-4 judge + string overlap) | **B4 source-priority** (system>user>tool) | C2 free-form, obey/ignore | 8 |
| IF-RewardBench (2603.04738) | meta: scores judges; ground truth A1+human | B2 taxonomy incl. Selection, condition STATED | C3 per-atom pass-rate + ranking (τ_b) | 0 |
| DeonticBench (2604.04443) | A4 SWI-Prolog execution | B3-ish (rule *selection* from long statutes) | C3 macro-F1 / numeric acc | 0 |
| **c22 (as designed)** | **A1 deterministic (IFBench checkers), per-atom diagnostics** | **B1 baseline; B3 Selection later** | **C1 micro-output, strict all-pass 0/1** | — |

(Manifest caveat, stated as required: these are **OpenAlex** `cited_by`/`by_year`/
`since_2025` counts fetched **2026-07-22**. They **undercount vs. Google Scholar** — the
IFEval "30" is implausibly low for a benchmark of its stature and should be read as an
OpenAlex-indexing artifact, not a true citation count — and are **near-zero by
construction for 2026 papers** [ComplexBench, VFF, LIFEBench, IF-RewardBench,
DeonticBench all show 0, which reflects OpenAlex lag and 2026 publication dates, not
irrelevance]. Treat these numbers as a coarse recency/indexing signal only.)

### EVIDENCE — crowded and empty regions of the map

**Crowded (A1/A2 × B1 × C2/C3): the "stated stacked verifiable constraints on free-form
output" corner.** IFEval, COLLIE, VFF, IFBench, FollowBench, and ComplexBench (its rule
subset) all live here. Every one of these:
- stacks multiple independently-checkable constraints and uses **constraint count as the
  difficulty dial** (IFEval 1–3; COLLIE 2–4; FollowBench 1–5; VFF 1–3; ComplexBench avg
  4.19; IFBench 1–2 eval / up to 6 train);
- confirms the **strict-AND floor effect** (VFF: even GPT-4-turbo only 35.31% at level-3;
  ComplexBench GPT-3.5 collapses 0.845→0.083 at 1 And+1 Chain+2 Selection; IFBench
  frontier models <50% on 58 OOD constraints; IF-RewardBench per-constraint base pass
  rate ~0.75, so ~4–5 strict atoms floor near 0.24–0.32 in expectation);
- confirms **deterministic > judge** on the checkable subset (VFF: Python 100% acc vs.
  GPT-4o 70%, GPT-4o-mini 59%, with 25–52% judge self-inconsistency; ComplexBench: rule
  component adds 13+ points on rule-defined questions; IF-RewardBench: Numerical/Format
  are the easiest, most-reliable category for judges).

**Empty / sparsely occupied region 1 — B1/B3 × C1 (strict all-pass 0/1 over a *trivial
micro-output*).** No paper deliberately trivializes the base task to isolate
constraint-following. IFEval/ComplexBench/VFF/IFBench/InFoBench/DeonticBench all use
*substantive* base tasks (essays, recipes, statutes), and their own analyses show
task-quality and constraint-following interact (IFBench Sec 5 reward-hacking; VFF Sec 5.4
content-vs-format entanglement). c22's "few words" output is the one design choice with
no direct precedent in the set.

**Empty region 2 — B3 hidden/discover for *format* constraints.** The Selection *operator*
exists (ComplexBench defines it; IF-RewardBench catalogs it), but **in every documented
example the branch condition is stated in the prompt** (ComplexBench dossier
`corpus_claim_verdict`: "the selection condition is always stated explicitly within the
instruction text … the model is never required to infer or discover an unstated/hidden
property"; IFBench and VFF have no Selection at all). The Instruction Hierarchy has a
*discover-which-instruction-governs* flavor but via source authority (B4), not an
inferred property of the input. DeonticBench has *rule selection from long statutes*
(B3-ish, Housing 96.8% "Wrong Rule" errors) but at heavyweight symbolic-execution scale.
**c22's "discover-not-copy" conditional constraint over trivial micro-outputs occupies a
genuinely empty cell.**

**Empty region 3 — A1 × synthetic infinite seeded generator.** Every paper here ships a
*fixed, frozen, released* dataset (IFEval 541 prompts; COLLIE 2,080; FollowBench 820;
InFoBench 500; ComplexBench 1,150; VFF fixed HF release; LIFEBench 10,800; IFBench 300;
IF-RewardBench 842; DeonticBench 6,232). None ships a seeded on-the-fly generator.
c22's infinite/contamination-resistant pool is a real gap — but (JUDGMENT) it is the
*build burden*, not the novelty claim; the checker code is reused, the generator is
net-new (per c22.md implementation lens: ~120–200 LOC, and the shipped `config.seed` is a
decoy).

---

## 2. Per-paper: what c22-as-designed adds (honest about overlap)

**IFEval (2311.07911).** *Overlap is maximal and direct: this is c22's oracle.* c22
reuses IFEval's deterministic checker library, and its "score 1 iff ALL checkers pass"
is exactly IFEval's prompt-level strict-accuracy; the baseline run uses *only* these
atoms. So on the un-optimized baseline, **c22 adds almost nothing to IFEval
scientifically** — it is IFEval's checkers re-sampled 3–5 per instance over a trivial
task. The honest additions are (a) a seeded generator vs. IFEval's frozen 541 prompts,
(b) deliberate base-task triviality to isolate constraint-following from content, and
(c) per-instance per-checker diagnostic logging (IFEval only reports aggregate
category-level accuracy). The *conceptual* additions (Selection, OOD atoms) are
explicitly out of the baseline scope. Note IFEval itself added a *loose* metric to fix
false negatives from formatting artifacts; c22 as specced inherits the checkers but not
the loose mitigation — a real, unglamorous gap to watch.

**COLLIE (2307.08689).** Overlap: grammar-defined, deterministically-checkable,
compositional constrained generation scored all-hold; COLLIE already stacks 2–4
base-constraints. c22 adds: a conditional Selection operator (COLLIE's grammar is AND/OR
only, Eq. 2 — no conditional branch); a base task *decoupled* from the constraints
(COLLIE's constraint *is* the generation target); a broader IFEval/IFBench atom taxonomy
(casing/forbidden-letter/keyword/end-token vs. COLLIE's count/position only); and a
seeded generator (COLLIE is corpus-extracted and frozen). Honest caveat: COLLIE is
*more* than c22 in one respect — it runs a genuine interactive feedback loop (which
plateaus at 66%), a comparison point c22 does not attempt. On the baseline specifically,
c22 adds nothing over COLLIE beyond the atom taxonomy and micro-output framing.

**FollowBench (2310.20410).** Overlap: "stack constraints, difficulty = count" is the
shared spine; c22 explicitly borrows FollowBench's level-ladder. Both log per-constraint
verdicts; both report the strict-AND compression (HSR is FollowBench's all-pass analog).
c22 adds: a *fully deterministic* verifier (FollowBench needs a GPT-4 judge for ~half its
open-ended instructions, and its own ablation shows judge agreement swings 67–88% by
prompt design); genuinely hidden Selection (FollowBench's constraints are all stated;
its only "discovery" burden falls on the judge, not the generator); micro-output; and
synthetic generation vs. a frozen 820-instruction set. Honest overlap: the *difficulty
mechanic* is essentially identical, so c22's incremental-difficulty story is not novel —
only its determinism and hidden-branch are.

**InFoBench (2401.03601).** Overlap: decompose a compound instruction into atomic
checkable sub-requirements and log per-atom verdicts — DRFR is the ancestor of c22's
diagnostic logging (c22.md credits it as the per-atom scoring model). c22 adds: a
judge-free deterministic oracle (InFoBench's per-atom check is an LLM judge, and its own
error analysis documents counting/spatial/echo failures of that judge); strict per-
instance 0/1 rather than DRFR's dataset-level partial-credit ratio; a synthetic infinite
pool (InFoBench flags manual authoring as an explicit scaling limitation); a hidden
Selection type (InFoBench decomposed questions are always derived from the visible
instruction). Overlap is mostly *philosophical* (decomposition), not mechanical.

**ComplexBench (2407.03978).** *This is the source of c22's single distinguishing
mechanism.* Overlap: ComplexBench introduces And/Chain/**Selection**/Nested and shows
Selection is empirically the hardest, most bias-prone composition (14.9% GPT-4 coherent-
test on nested Selection; ~70% branch-position bias corrected via Selection Branch
Expansion). c22's whole anti-enumeration argument rests on this operator. **But the honest
delta is precise and load-bearing: ComplexBench's Selection condition is always STATED in
the instruction** (dossier verdict: "'discover, not copy' is NOT literally supported by
the paper"). c22's hardening — the branch condition is a *hidden, unstated property of the
input* — is a genuine extension ComplexBench does not test. c22 also adds: pure-rule
scoring (ComplexBench is ~83% LLM-judge); strict all-pass vs. dependency-aggregated DRFR;
micro-output; synthetic generation. The Selection Branch Expansion result is a direct
warning c22's generator must heed to avoid a guessable positional/frequency bias.

**VFF (2502.04498).** *Structurally the closest to the baseline.* Overlap is very high:
fixed pool of parametrized meta-constraint templates, multi-level stacking (1–3), strict
conjunctive scoring I = ∏Fₖ (exactly c22's all-pass), fully deterministic Python
checkers, per-constraint diagnostics — and VFF's Sec 5.3 is the strongest published
justification for c22's determinism choice (Python 100% vs. judges 59–70%, 100× faster,
free, judges 25–52% self-inconsistent). c22 adds: hidden Selection (VFF has none — its
meta-constraints are static template-fills, "the single largest structural difference");
trivial micro-output (VFF answers open-ended Alpaca questions); a seeded infinite
generator (VFF is a frozen HF release); IFBench-style OOD atoms. Honest assessment: **on
the un-optimized baseline, c22 is close to a VFF-eval-slice re-implemented over micro-
tasks with the IFEval atom set** — the novelty is entirely in the later Selection/OOD/
generator layers.

**LIFEBench (2505.16234).** Overlap is narrow: length is one of c22's atoms, both use
deterministic non-judge scoring, both find exact ("Equal To") constraints far harder
than loose ones. c22 adds: multi-constraint composition (LIFEBench studies *one* length
constraint per instance); hidden Selection; strict binary all-pass (LIFEBench uses a
continuous 0–100 score); synthetic generation. Crucially, LIFEBench's central failures
(length-awareness deficits, "lazy" refusals, collapse at long outputs) are *artifacts of
long generation* that c22's micro-output design deliberately sidesteps — so LIFEBench is
mostly a cautionary data point (don't ask for exact long lengths), not a competitor.

**IFBench (2507.02833).** *The live frontier and c22's OOD-atom source.* Overlap:
deterministic per-constraint checking, stacking (up to 6 in training), strict/loose
framing, and an atom taxonomy that includes exactly c22's example atoms
(word-count-range, keyword, casing, start/end-with, forbidden-word). c22 adds:
evaluation-time 3–5 stacking (IFBench's *shipped eval* is only 1–2 constraints — the
deeper stacking is training-only); a Selection type (absent from all 58+29 IFBench
constraints); micro-output vs. substantive WildChat prompts (IFBench's own Sec 5 shows
task-quality/constraint interaction causes reward hacking, a confound c22 avoids);
first-class per-checker diagnostic logging (IFBench reports category-level aggregates);
and a synthetic pool vs. its frozen 300-prompt release. IFBench's Table 1 stacking sweep
(IFBench score peaks at n=3, non-monotonic after) is direct empirical support for c22's
3–5 dial. Honest overlap: the baseline *is* IFEval atoms, which IFBench shows are
saturated/overfit — a currency risk noted in §4.

**Instruction Hierarchy (2404.13208).** Low, mostly conceptual overlap: both care about
*which rule governs* when instructions coexist. But this is B4 (source-priority: system>
user>tool) resolved by a *training* intervention (SFT+RLHF), scored by GPT-4 judge/string
heuristics — categorically unlike c22's per-atom deterministic conjunction over output
properties. c22 adds: deterministic 0/1 scoring, compositional per-atom structure, a
tunable difficulty dial, an infinite synthetic pool, and the "discover the hidden rule"
property located at the *format-constraint* level rather than adversarial social-
engineering. Honest note: c22.md cites an "authority-priority variant" inspired here, but
that is not part of the current design and the dossier flags a citation-hygiene issue
(cited via OpenReview id, and the paper never mentions "IHEval").

**IF-RewardBench (2603.04738).** Different axis: it benchmarks *judges*, not generators.
Overlap: it adopts ComplexBench's Selection operator (again with the condition stated,
not discovered) and independently corroborates two c22 design bets — Numerical/Format
constraints are the most reliably checkable category, and Chain/Selection are harder to
verify than Single/And. Its ~0.75 per-constraint base pass rate is a useful calibration
anchor for c22's floor risk. c22 adds: it evaluates a *generator* directly against a
deterministic checker (closing the loop IF-RewardBench studies only from the judge side);
strict all-pass vs. mean-pass-rate; a trivial contamination-resistant task vs. curated
naturalistic instructions ($15k human annotation). Overlap is corroborative, not
competitive.

**DeonticBench (2604.04443).** Low overlap, adjacent family (rule-application). Both are
"verifiable" in the deterministic sense and both have a *rule-selection* flavor (its
Housing "Wrong Rule" errors echo c22's Selection). But DeonticBench uses external
Prolog execution (A4) over thousand-token statutes with macro-F1 aggregation — the
opposite of c22 on scale, verifier weight, and output shape. c22 adds: per-atom automatic
diagnostics (DeonticBench's error taxonomy is manual/post-hoc), a cheap pure-Python
oracle (vs. compiling Prolog with 20s timeouts), a tunable difficulty dial, synthetic
infinite generation, and strict per-instance 0/1. Mostly a contrast reference.

---

## 3. Publishability analysis — IF prompt-optimization produces large *verified* gains

Premise for this section: the un-optimized baseline runs first (IFEval atoms only, 3–5
per trivial instance, all-pass 0/1). *Then* COPRO/MIPROv2/GEPA is applied and — the
conditional we analyze — produces **large, verified gains** on this task.

### (a) The workshop-paper claim, stated precisely

EVIDENCE-grounded, JUDGMENT-authored claim:

> "On a controlled, deterministically-verified, contamination-resistant stacked-
> instruction-following task (3–5 composed IFEval checker atoms per trivial micro-task,
> scored strict all-pass 0/1 with zero label noise), reflective prompt optimization
> (GEPA) raises all-pass accuracy from X% (un-optimized baseline prompt) to Y% at a
> budget of Z rollouts, outperforming MIPROv2/COPRO by Δ points; because every point of
> gain is confirmed by an execution-based oracle with no LLM-judge in the loop, the
> improvement is attributable to the optimizer rather than to reward-model noise or
> judge drift."

The load-bearing, defensible parts (each traceable to evidence): (i) **zero-label-noise
measurement** — VFF Sec 5.3 and ComplexBench Table 4 establish that deterministic
checking is exactly reproducible where LLM-judges are 25–52% self-inconsistent, so gains
here cannot be judge artifacts; (ii) **controlled difficulty dial** — IFBench Table 1,
VFF levels, FollowBench CSL all establish constraint-count as a smooth, studied dial;
(iii) **contamination resistance** — synthetic seeded generation, motivated by
IF-RewardBench needing 300 *fresh* instructions because prior benchmarks are contaminated.

What the claim must NOT overreach to: it is **not** a claim that optimizers improve
instruction-following *capability* in general (that needs OOD atoms + real tasks), only
that they improve *this measured metric* on *this task*.

### (b) Which publication line

JUDGMENT: **Primarily the prompt-optimizer-benchmarking line (trends area 9), not this
family's own evaluation line.** Reasoning from evidence:
- The un-optimized baseline is, by construction, "IFEval atoms re-stacked" — the
  per-paper analysis (§2) shows it adds essentially nothing to IFEval/VFF/IFBench *as an
  instruction-following benchmark*. So a claim framed as "a new IF benchmark" would be
  correctly rejected as incremental over a crowded A1×B1×C2 corner.
- Its value under the large-gains conditional is as a **clean testbed for comparing
  optimizers**: deterministic oracle, tunable difficulty, contamination-proof, cheap.
  That is precisely the MAS-PromptBench 2026 use case (trends area 9: "a benchmark
  specifically for comparing optimizers … exactly whetstone-ai's own use case"), and
  GEPA is an ICLR 2026 oral whose headline IF task is IFBench (c22's lineage). A result
  of the form "here is a controlled task on which GEPA's advantage over MIPROv2 is
  large/small and *cleanly measurable*" slots into this active, self-referential line.
- Secondary/contributing framing to the IF line is possible ONLY if the Selection/OOD
  layers are added and shown to resist optimization differently than stated atoms — i.e.,
  not the baseline, and not the pure large-gains-on-baseline result.

### (c) Baselines, ablations, comparisons reviewers would demand

From the dossiers' own methodological practice:
1. **Optimizer baselines, not just one:** COPRO vs. MIPROv2 vs. GEPA vs. a strong manual/
   few-shot prompt, at matched rollout budgets (GEPA's own selling point is
   35× fewer rollouts — reviewers will demand the budget-matched frontier, not a single
   point).
2. **A no-optimization floor and a demonstration/few-shot control.** COLLIE (one-shot ≈
   zero-shot), IFEval, InFoBench (few-shot "no significant improvement"), and ComplexBench
   all report that naive demonstrations don't move this kind of task — so reviewers will
   ask whether the "optimizer gain" is just recovering what a couple of exemplars already
   give. Must show optimizer > few-shot control.
3. **Difficulty-stratified curves,** not a single aggregate. Report all-pass accuracy vs.
   constraint count (per IFBench Table 1, VFF levels, FollowBench CSL) to show the gain is
   not concentrated where the baseline was trivially near-ceiling or near-floor.
4. **Floor/ceiling and monotonicity checks.** Given the strict-AND floor (IF-RewardBench
   ~0.75 per-atom → ~0.25 at 4–5 atoms), reviewers will demand evidence the metric wasn't
   floored/saturated such that "gains" are noise. Report per-atom (loose/DRFR-style)
   accuracy alongside strict all-pass, as IFEval/FollowBench/InFoBench all do.
5. **Held-out generalization of the optimized prompt** to fresh seeds and to atom
   combinations unseen during optimization — the IFBench overfitting finding (frontier
   models 80%+ on 25 IFEval atoms, <50% on 58 OOD) is the reviewer's first suspicion:
   did the optimizer just memorize the atom set?
6. **Multiple base models** (a small open model + a frontier model). VFF/IFBench/
   FollowBench all show small-model floors differ qualitatively; a single-model result
   won't generalize.
7. **Seeds/variance.** Instruction Hierarchy reports ±1 SD per eval; c22 must report
   confidence intervals over generator seeds and optimizer runs (DeonticBench uses
   B=1000 bootstrap).

### (d) Realistic venues / workshops and why

JUDGMENT, grounded in trends area 9 and the citation recency data:
- **Workshop tier (most realistic):** a prompt-optimization / LLM-agents / DL4C-style
  workshop at ICLR/NeurIPS/ACL 2026, or a benchmarks-and-evaluation workshop. Rationale:
  the contribution is a *clean measurement instrument + an optimizer comparison*, which is
  exactly workshop-shaped, and the optimizer community is unusually active right now
  (GEPA oral, MAS-PromptBench, DD-GEPA, AIR — all 2026 per trends §9). A tight,
  well-controlled optimizer-comparison paper is a natural fit.
- **Possible main-track path (harder):** NeurIPS D&B or an *LMSYS/eval* venue **only if**
  the task is elevated beyond the baseline — Selection + OOD atoms + the infinite generator
  positioned as "a contamination-proof controllable optimizer-stress benchmark." The
  precedent models are IFBench (NeurIPS 2025 D&B) and ComplexBench (NeurIPS 2024 D&B).
  This requires the full design, not the baseline.
- **Why not a pure IF-benchmark venue:** §2 shows the baseline is dominated by
  IFEval/VFF/IFBench on the IF-benchmark axis; reviewers there would (correctly) call it
  incremental.

### (e) What result would NOT be publishable (recognize early)

- **Large gains on the baseline that vanish on held-out seeds/atom-combos** → the
  optimizer memorized the fixed atom set (the IFBench overfitting failure mode). Not
  publishable; it is a known negative.
- **Gains that a 1–2-shot exemplar control also produces** → COLLIE/InFoBench/IFEval all
  show demos are cheap here; "optimizer beats zero-shot" without "optimizer beats few-
  shot" is not a result.
- **Gains only because the strict-AND metric was floored and the optimizer nudged a few
  instances off the floor** → measurement artifact, not capability; reviewers will catch
  it via the per-atom curve.
- **Optimizers rank the same as on existing benchmarks with no new insight** → then the
  task adds nothing over MAS-PromptBench/IFBench-as-GEPA-testbed; a redundant benchmark.
- **A pure "we built another verifiable IF benchmark" framing with no optimizer story and
  no Selection/OOD/generator novelty** → incremental over the crowded corner in §1;
  rejected on novelty regardless of the numbers.
- **Any result whose scoring leaned on an LLM-judge** → forfeits the one clean advantage
  (VFF/ComplexBench/IF-RewardBench all show judges are noisy); the whole point is the
  deterministic oracle.

Concise recognizer: *publishable ≈ (large gains) ∧ (survive held-out seeds+atoms) ∧
(beat a few-shot control) ∧ (not a floor artifact) ∧ (deterministic oracle) ∧ (framed as
an optimizer-comparison contribution).* Miss any conjunct → treat as not publishable and
pivot early.

---

## 4. Research-currency verdict (from citation data, with stated caveats)

EVIDENCE (OpenAlex `cited_by` / `since_2025`, fetched 2026-07-22):
IFEval 30 (22 since 2025) · Instruction Hierarchy 8 (7) · COLLIE 1 · FollowBench 1 ·
InFoBench 1 · IFBench 1 · ComplexBench 0 · VFF 0 · LIFEBench 0 · IF-RewardBench 0 ·
DeonticBench 0.

**Mandatory caveats (as instructed):** these are **OpenAlex** counts, which **materially
undercount vs. Google Scholar** — IFEval's "30" is not credible as a true count for the
canonical verifiable-IF benchmark and is an indexing artifact; and they are **near-zero
by construction for 2026 papers** (ComplexBench, VFF, LIFEBench, IF-RewardBench,
DeonticBench show 0 largely because of OpenAlex lag and recent publication, not
irrelevance). So the *absolute* numbers carry little signal; only coarse patterns do.

JUDGMENT — currency verdict: **The task family is squarely current, but c22's baseline
sits on its most saturated layer.** Two evidence strands beyond the raw counts:
- The lineage is live and accelerating: IFEval (2023) → ComplexBench (NeurIPS 2024 D&B) →
  VFF/LIFEBench/IFBench (2025, IFBench at NeurIPS 2025 D&B) → IF-RewardBench/DeonticBench
  (2026, ACL/COLM-targeted). The trend scan independently rates verifiable IF the
  "most heavily benchmarked LLM-eval subfield in 2025-2026," with new papers ~monthly and
  a live leaderboard. IFEval's 22-of-30 citations landing in 2025+ (even undercounted)
  confirms the *founder* is still actively built upon.
- But the same evidence flags the risk: IFBench exists *because* IFEval's 25 atoms are
  saturated/overfit (frontier <50% OOD). c22's un-optimized baseline uses *exactly those
  saturated atoms*. So the family is current; the **baseline's specific atom set is the
  least-current slice of it.** Currency is recovered only by the OOD-atom + Selection +
  generator layers — i.e., by moving c22 off the crowded corner.

Net: **research-currency is a genuine strength of the *family* and of c22's *full
design*, and a genuine weakness of c22's *baseline as specced*.** The publishable path
(§3) leans on the optimizer-benchmarking currency (GEPA/MAS-PromptBench, trends §9),
which is strong and self-referential, rather than on out-citing the IF-benchmark corner.

---

## Executive summary (10 lines)

1. Three axes organize this family: verifier type (deterministic→judge→symbolic),
   rule provenance (stated / stated-composition / hidden-discover / source-priority),
   and output-shape×aggregation (micro all-pass vs. free-form partial-credit).
2. Every one of the 11 papers is single-shot on the generation side; "multi-turn" is
   always a judge, curriculum, or difficulty-ladder artifact — so it is not a live axis.
3. Crowded corner: deterministic/hybrid scoring × stated stacked atoms × free-form output
   (IFEval, COLLIE, VFF, IFBench, FollowBench, ComplexBench) — all use constraint-count as
   the dial and confirm the strict-AND floor and deterministic>judge.
4. c22's genuinely empty cells: strict all-pass over a *trivial micro-output*; *hidden*
   ("discover-not-copy") Selection over format constraints; and a *seeded infinite*
   generator — none of which any paper occupies.
5. Honest overlap: the un-optimized baseline (IFEval atoms only) adds almost nothing over
   IFEval/VFF/IFBench as an IF benchmark; c22's novelty lives entirely in the later
   Selection/OOD/generator layers.
6. Selection is sourced from ComplexBench, but every prior Selection states its condition
   in-prompt; c22's hidden-condition hardening is the one real mechanistic extension.
7. If optimization yields large *verified* gains, the precise claim is "GEPA raises
   strict all-pass X→Y at Z rollouts, beating MIPROv2/COPRO by Δ, with zero label noise
   because scoring is execution-based" — a measurement claim, not a capability claim.
8. It belongs to the prompt-optimizer-benchmarking line (trends §9: GEPA ICLR-2026 oral,
   MAS-PromptBench 2026), realistically a workshop; a NeurIPS-D&B path needs the full
   Selection+OOD+generator design, not the baseline.
9. Reviewers will demand: multiple optimizers at matched budgets, a few-shot control,
   difficulty-stratified + per-atom curves, held-out seed/atom generalization, multiple
   base models, and variance — and will reject floor-artifact, memorized-atom, or
   judge-scored results.
10. Currency (OpenAlex, undercounted, 2026-near-zero-by-construction): the family is very
    current and the full design is timely, but the baseline's IFEval atom set is the
    saturated slice — so lean the publishable story on optimizer-benchmarking currency.
