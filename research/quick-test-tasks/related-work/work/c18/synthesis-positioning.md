# c18 Positioning Synthesis — Depth-Controlled Synthetic Deduction (True/False/Unknown)

Scope: positions c18-as-designed against the 7 related-work papers in this dossier
set, maps the design space, and assesses publishability of the planned
un-optimized baseline plus optional prompt-optimization (COPRO/MIPROv2/GEPA).

Convention used throughout: **[EVIDENCE]** = drawn from the dossiers / papers /
trends doc / citation data; **[JUDGMENT]** = my inference or recommendation.
Citation counts are OpenAlex, fetched 2026-07-22; see §4 caveats.

The seven papers:
- **P1 PrOntoQA** (Saparov & He, ICLR 2023) — arXiv 2210.01240
- **P2 PrOntoQA-OOD** (Saparov et al., NeurIPS 2023) — arXiv 2305.15269
- **P3 FLD** (Morishita et al., ICML 2023) — arXiv 2308.07336
- **P4 FLD×2 / "FLD*"** (Morishita et al., NeurIPS 2024) — arXiv 2411.12498
- **P5 JustLogic** (Chen, Zhang & Tao, 2025; ICML-2025-implied) — arXiv 2501.14851
- **P6 Multi-LogiEval** (Patel et al., EMNLP 2024 Main) — arXiv 2406.17169
- **P7 SATBench** (Wei et al., 2025; EMNLP-2025-claimed-but-unverified) — arXiv 2505.14615

---

## 1. Positioning map

I organize the family on three axes that the dossiers show actually separate these
papers (rather than axes on which they all coincide).

### Axis A — Output shape & what the verifier grades  [EVIDENCE]
- **Label-only, exact-match** (bare T/F/U or Yes/No; grader = generator's stored
  gold): JustLogic (P5), Multi-LogiEval (P6), SATBench (P7 label track).
- **Full proof-chain + step-level structural verifier** (grade the CoT, not just the
  label): PrOntoQA (P1), PrOntoQA-OOD (P2), FLD (P3), FLD×2 (P4). P1 and P2 argue
  *explicitly* that label-only accuracy is an inadequate measure of reasoning and
  build symbolic proof-parsers to grade the intermediate chain.
- **Label + LLM-judged free-text trace**: SATBench (P7 secondary track).

**c18 sits at the label-only, exact-match pole** — it deliberately drops full
proof-chain output ("keeps the label only, to bound output tokens and preserve
exact-match scoring", c18.md Dead branches). Its *planned* addition of an
independent forward-chaining fixpoint oracle re-derives the label but does not grade
a model-emitted chain, so it stays on this pole.

### Axis B — Content provenance / contamination-resistance  [EVIDENCE]
- **Fully fictional / nonce predicates** ("wumpus/yumpus", "impus/vumpus"):
  PrOntoQA (P1), PrOntoQA-OOD (P2) — and **c18** inherits exactly this surface.
- **Content-free randomized symbols / WordNet-templated abstract predicates**:
  FLD (P3), FLD×2 (P4).
- **Real-world sentences with randomized/decoupled truth values**: JustLogic (P5,
  GenericsKB sentences; ships a dedicated prior-knowledge-independence test, |Δ|=0.4).
- **Naturalistic real-domain entities** (finance, wildlife, etc.): Multi-LogiEval
  (P6) — the dossier flags this as *less* contamination-resistant by construction.
- **LLM-authored narrative wrapping of CNF**: SATBench (P7).

### Axis C — Reasoning regime & the core difficulty dial  [EVIDENCE]
- **Inference-rule chaining, depth = dial**: PrOntoQA (hops 1/3/5, P1), PrOntoQA-OOD
  (depth+width+rule-composition, P2), FLD/FLD×2 (proof-tree depth 1–8 + rule-set size,
  P3/P4), Multi-LogiEval (depth 1–5 × 33 rule types, P6), JustLogic (depth 1–7 × 7
  argument forms, P5). **c18** is squarely here: depth bins D0–D5 + distractor rules +
  CWA/negation-as-failure vs open-world.
- **Search / constraint satisfaction, clause-count = dial**: SATBench (P7, SAT/UNSAT).
  **c18's second (CSP/seating) stratum** reaches toward this pole but is unique-solution
  extraction, not SAT/UNSAT existence.

A useful sub-axis of C is **label space**: three-way T/F/Unknown (open-world) is used
by P1-adjacent ProofWriter lineage, **c18**, JustLogic (P5), FLD/FLD×2 (P3/P4:
proved/disproved/unknown); binary is used by base PrOntoQA closed-world (P1),
Multi-LogiEval (P6, Yes/No), SATBench (P7, SAT/UNSAT).

Protocol axis (single-shot vs multi-turn) is **near-degenerate** for this family
[EVIDENCE]: all seven papers are single-forward-pass at the model-under-test level.
The only genuine multi-turn probes are JustLogic's tree-of-thought ablation (P5) and
FLD's stepwise T5 prover (P3) — both secondary, and JustLogic finds ToT mostly *hurts*.
So I do not use it as a primary organizing axis; c18 is single-shot like everyone else.

### Where the map is crowded vs empty

**Crowded region** [EVIDENCE]: {inference-rule chaining × depth-as-dial × three-way
label}. FLD, FLD×2, JustLogic, and c18's core stratum all live here; PrOntoQA-OOD and
Multi-LogiEval are one axis away each. This is the most densely occupied cell in the
family and is exactly where c18's *core* stratum lands.

**c18's distinguishing position** [JUDGMENT]: c18 is the only entry that combines
(a) fictional nonce surface (P1/P2 pole of Axis B), (b) label-only exact-match output
(P5/P6 pole of Axis A), and (c) a live, seed-deterministic, byte-identical regenerable
generator with no model call in the loop. Every published peer picks a different
combination: P1/P2 pair fictional surface with *full-proof grading*; P5/P6 pair
label-only with *real-world/naturalistic* surface. The specific triple {fictional +
label-only + reseedable-no-LLM} is an **empty cell** among the seven.

**Empty regions in the family** [EVIDENCE + JUDGMENT]:
- **Open-world Unknown under a fictional surface, verified by an independent oracle.**
  P5 has 3-way labels but real-world sentences; P3/P4 have proved/disproved/unknown but
  grade full proofs and use released static splits; c18-as-designed is the only one
  aiming an *independent forward-chaining re-derivation* at a *label-only* fictional
  3-way task. (This is also its main soundness risk — the label is otherwise
  "definitional"; c18.md Measurement skeptic.)
- **A CSP/unique-solution stratum welded to a deduction-chaining stratum in one
  harness.** None of P1–P6 has any CSP stratum; SATBench (P7) is CSP but SAT/UNSAT-only
  and has no chaining stratum. c18's two-stratum design occupies this gap but at real
  implementation cost (c18.md I3/I4/I5 = 1).
- **Contamination-controlled naturalistic-language deduction** is *filled* by JustLogic
  (P5), so c18's fictional surface is not novel there — it is a deliberate trade of
  P5's measured NL-complexity for stronger anti-contamination and simpler surface.

---

## 2. What c18-as-designed adds beyond each paper (honest about overlap)

**P1 — PrOntoQA (2210.01240).** [EVIDENCE] This is c18's literal generator lineage:
fictional ontologies, hop-controlled depth, modus-ponens chaining, distractors, the
"impus/vumpus/not loud" surface in c18's own example is P1's style. Overlap is maximal
on Axes B and C. [JUDGMENT] What c18 adds is almost entirely on Axis A and label space:
P1's whole thesis is that *label accuracy is inadequate* and builds a step-level proof
parser; c18 does the opposite, keeping a single T/F/Unknown token for cheap exact-match,
and imports the open-world **Unknown** label (which base P1 lacks — it is closed-world
True/False) from the ProofWriter branch. c18 also adds a designer-known D0–D5 *ladder as
a deliverable* (P1 uses hops 1/3/5 only as an independent variable for its causal
analysis) and a CSP stratum P1 has no analogue for. Honest caveat: on the shared axes
c18 is a *reuse*, not an advance — the novelty is the label-only/ladder/CSP repackaging,
not the generator.

**P2 — PrOntoQA-OOD (2305.15269).** [EVIDENCE] Same codebase lineage; adds a complete
natural-deduction rule set, controllable depth+width, and compositional proofs, graded
by a formal step-checker. Its queries are always provable/disprovable — **no open-world
Unknown**. [JUDGMENT] c18 adds: (a) the explicit open-world Unknown / CWA-vs-NAF
distinction P2 does not model; (b) a bounded single-token scoring target vs P2's
full-proof formal-parse pipeline (much cheaper per instance, at the cost of P2's rich
step diagnosis); (c) a run-verified, byte-identical *reseedable* harness (P2 is a
research paper releasing a static `generated_ood_data.zip`, contamination-risked on
reuse); (d) the CSP stratum. Overlap is high on depth-as-dial and fictional surface;
c18 does *not* match P2's rule-completeness or compositional-proof breadth — that is
genuinely P2's territory and c18 does not attempt it.

**P3 — FLD (2308.07336).** [EVIDENCE] Principled complete-axiom generator with
proved/disproved/unknown labels, controllable depth (1–8) and distractors (0–~20);
scores full proof sequences (proof accuracy) as primary, label-only "answer accuracy"
as secondary. Its LLM few-shot numbers are a headline resource: **GPT-4 10-shot ≈ 52.4
answer-acc on FLD, ≈49.4 on FLD\*; GPT-3.5 and LongAlpaca near the 33.3% random floor;
proof-accuracy far lower**. [JUDGMENT] c18 adds a *frozen-model, inference-time,
label-only* protocol (P3's LLM eval is secondary to fine-tuned-prover study), a
fictional-ontology surface (P3 uses content-free WordNet templates), fresh-per-run
seeded regeneration (P3 ships static v1/v2 corpora), and a CSP stratum. Overlap is real:
both are depth+distractor-dialed three-way-label synthetic deduction. The honest point
is that P3 already *has* the stronger verifier (step-checked proof accuracy) c18 forgoes
for cost — c18's independent oracle only recovers label soundness, not P3's step-level
grade.

**P4 — FLD×2 / "FLD\*" (2411.12498).** [EVIDENCE] Upstream corpus+training paper (ALT
fine-tuning), *not* an inference-time eval: it explicitly **excludes its own corpus from
evaluation** and reports only downstream-transfer accuracy after fine-tuning. Three-way
proved/disproved/unknown labels; "unknown" logical-steps = "None"; 0–20 adversarial
distractors; s=1–8. [JUDGMENT] c18 adds the thing P4 never reports: **direct
single/few-shot accuracy of an off-the-shelf model on fresh depth-d T/F/Unknown
instances**. This is a clean, non-overlapping contribution — P4 has "essentially no
directly reusable number for how well an LLM answers a depth-d query out of the box."
c18 also adds fictional surface, reseedable no-LLM generation, and an independent oracle.
Overlap is confined to the shared label scheme and depth/distractor dials.

**P5 — JustLogic (2501.14851).** [EVIDENCE] The closest living competitor:
programmatically generated, depth 1–7 × 7 argument forms, **three-way True/False/
Uncertain, exact-match, single-shot CQO→A** — structurally almost identical to c18's
scoring design. Ships a prior-knowledge-independence test (|Δ|=0.4, essentially at the
33.3% floor), depth-stratified curves, human baseline (avg 73.0%, ceiling 100%), best
model DeepSeek R1 80.9%. Regenerable by design. [JUDGMENT] This is where c18's added
value is *thinnest and must be stated honestly*: JustLogic already delivers the 3-way
exact-match single-shot deductive benchmark with contamination control and depth
laddering. c18's remaining differentiators are narrow: (a) fully fictional/nonce surface
vs P5's real-sentence-randomized-truth surface (a trade, not a strict improvement — P5
even *instruments* its NL complexity, which c18 does not); (b) an explicit
CWA-vs-open-world / negation-as-failure toggle as a named dial (P5 uses a single
open-world convention); (c) the CSP stratum; (d) an independent forward-chaining oracle
(P5's failure analysis is qualitative over CoT traces). Overlap here is high enough that
c18 cannot claim task-shape novelty over P5.

**P6 — Multi-LogiEval (2406.17169).** [EVIDENCE] Depth 1–5 × 33 inference-rule types
across PL/FOL/non-monotonic, **binary Yes/No**, single-shot zero-shot-CoT, naturalistic
real-domain contexts, static human-validated release (~1.6k items, ~14% needed manual
correction). Reports a strong depth ladder (avg ~68%→~43%, several models *below* the
per-depth random baseline). [JUDGMENT] c18 adds: (a) three-way Unknown with explicit
open-world semantics (P6 is binary and does not probe judgment-withholding); (b)
fictional nonce predicates removing the real-world-prior shortcut P6's naturalistic
domains still allow; (c) a live reseedable generator with an *automated proof-engine*
gold label (P6's gold is human-validated LLM-generated instances); (d) the CSP stratum.
Overlap: both are depth-laddered synthetic rule-chaining with exact-match. P6's genuine
edge that c18 does *not* match is rule-type breadth (33 rules, non-monotonic) — c18's
core rule set is narrower.

**P7 — SATBench (2505.14615).** [EVIDENCE] 2100 CNF-derived narrative puzzles, binary
SAT/UNSAT, solver oracle, clause-count dial, 0-shot CoT; documented hard tail (o4-mini
65.0% on hard UNSAT vs 50% random; avg model ~53% on hard; NL framing costs ~5 pts vs
raw formula). Frames itself as *search-based*, explicitly contrasting with the
inference-rule-based RuleTaker/FOLIO lineage c18 belongs to. [JUDGMENT] c18 adds a
genuine deductive-chaining stratum with an orthogonal depth ladder (D0–D5) rather than
clause-count as the only knob; open-world three-way Unknown (P7 is strictly binary); and
a proof engine that names the failed inference step (finer than P7's UNSAT-core). Overlap
is mostly at the meta level (both are solver/oracle-backed regenerable synthetic
generators with a documented hard tail). [JUDGMENT] SATBench is best read as a
**reference/hard-tail complement** for c18's CSP stratum, not a competitor for its core
stratum — c18's CSP seating puzzles are conceptually adjacent to SATBench's CNF→narrative
construction and could borrow from it.

**Cross-cutting honest summary** [JUDGMENT]: c18's task *shape* is not novel — P5
(JustLogic) occupies nearly the same cell, and P1/P2/P3 own the generator lineage. c18's
defensible additions are (i) the exact combination {fictional surface + label-only exact
match + reseedable-no-LLM generator + independent forward-chaining oracle + CSP stratum},
none of which any single peer holds all of, and (ii) as an *artifact*, a run-verified,
~100-line-glue, deterministic harness. Those are integration/engineering contributions,
not a new benchmark concept.

---

## 3. Publishability analysis (if optimization yields large verified gains)

The plan: first run an **un-optimized baseline** (PrOntoQA regenerated as-is with fresh
seeds + fresh fictional ontologies, existing hop-depth settings, existing question
format, **True/False exact-match** output, **no Unknown-label handling beyond the
codebase's native support, no distractor/soft-rule strata**), then potentially
prompt-optimize with COPRO / MIPROv2 / GEPA. [EVIDENCE re baseline scope: task statement.]

Important framing constraint [JUDGMENT]: the *un-optimized baseline as specified is a
faithful regeneration of PrOntoQA*, not the full c18 design (no Unknown, no distractors,
no CSP, no independent oracle). So any first-round result is a statement about
**prompt-optimization on PrOntoQA-style depth-binned deduction**, not about c18's novel
axes. This directly shapes which claim and which publication line are available.

### (a) The precise workshop-paper claim (if large verified gains appear)

[JUDGMENT] State it as an *optimizer-behavior* claim on a controlled, contamination-free,
depth-parameterized deduction task — **not** as a new-benchmark or new-reasoning-science
claim:

> "On freshly regenerated, contamination-free PrOntoQA-style depth-binned deductive
> True/False tasks (fictional ontologies, hop-depth D∈{…}, exact-match), automated prompt
> optimization (COPRO / MIPROv2 / GEPA) raises exact-match accuracy by **X±CI points over
> a fixed non-optimized instruction baseline** on model M, with the gain **concentrated at
> depth ≥ d** (or: **uniform across depth**), and **GEPA outperforming MIPROv2/COPRO by
> Y points at Z× fewer rollouts**. The gain does/does not survive fresh-seed regeneration
> and transfer to held-out depths."

The load-bearing qualifiers [JUDGMENT]: fixed baseline, per-depth breakdown, fresh-seed
held-out replication (contamination-free is the whole point of the task), rollout-cost
accounting, and CI/seeds. Absent these, it is not a defensible workshop claim.

### (b) Which publication line  [EVIDENCE for line existence + JUDGMENT for choice]

Two candidate lines:
- **This family's own deductive-evaluation line** (RuleTaker→PrOntoQA→FLD→JustLogic→
  SATBench). [JUDGMENT] A prompt-optimization result does **not** belong here as a
  primary contribution: this line's currency is about *harder/cleaner deduction
  benchmarks and reasoning-failure science* (P5, P7 are the live 2025 entries), and an
  un-optimized-baseline regeneration of a 2023 generator adds no benchmark novelty (P5
  already fills the near-identical cell — §2).
- **Prompt-optimizer benchmarking (Area 9)** [EVIDENCE, trends §9]: GEPA is an **ICLR
  2026 oral** (arXiv 2507.19457, beats MIPROv2 ~13% aggregate, up to 20% over GRPO-RL at
  up to 35× fewer rollouts); **MAS-PromptBench 2026** ("When Does Prompt Optimization
  Improve…", arXiv 2606.23664) is a purpose-built optimizer-comparison benchmark; AIR and
  DD-GEPA are further 2026 entries. The trends doc explicitly says this line is "active,
  self-referential, and receptive" and that a quick synthetic task doubling as a small
  optimizer benchmark "slots straight in."

[JUDGMENT] **The result belongs in the prompt-optimizer benchmarking line (Area 9), not
the family's own evaluation line.** The task is the *instrument*; the optimizers are the
*object of study*. A depth-parameterized, contamination-free, exact-match task with a
tunable difficulty dial is precisely the kind of clean testbed MAS-PromptBench-style work
wants. The deduction family provides citable framing/lineage, not the headline.

### (c) Baselines, ablations, comparisons reviewers will demand  [JUDGMENT]

1. **A strong fixed-prompt baseline** — not a strawman zero-shot prompt. Include a
   hand-written CoT prompt and a few-shot prompt; JustLogic (P5) and Multi-LogiEval (P6)
   both show few-shot/CoT alone move accuracy a lot (e.g. P5 CoT: Llama3-8B 49.8→57.8),
   so "optimizer beats a bad baseline" is not publishable.
2. **All three optimizers head-to-head** (COPRO, MIPROv2, GEPA) under **matched rollout /
   token budget** — GEPA's headline claim is efficiency, so cost-normalized comparison is
   mandatory (trends §9).
3. **Per-depth breakdown** (does the gain concentrate at the hard tail? P1/P2/P5/P6/P7 all
   report depth ladders — reviewers expect one here).
4. **Fresh-seed / held-out-ontology replication** — the entire rationale of the task is
   contamination-resistance; a gain that evaporates on regeneration is an overfit to a
   fixed instance pool. This is the single most important ablation.
5. **Cross-model transfer** — does an optimized prompt from model M help model M′? (P4/P5
   both stress prompt/scale interactions.)
6. **Random / majority floor and, ideally, a human or oracle ceiling** — P5 gives 73%
   human avg / 100% ceiling; P6 gives per-depth random baselines *above which several
   models fail*. Reviewers will want the floor to interpret X points.
7. **Seed variance / CIs** — P1 uses 95% Wilson intervals on 400/condition; anything
   less will be flagged.
8. **A verifier-soundness check** — since c18's label is otherwise "definitional", the
   independent forward-chaining oracle (or at least a spot-check) is needed so a
   generation bug isn't mistaken for an optimizer gain.

### (d) Realistic venues / workshops and why  [EVIDENCE for venues + JUDGMENT]

- **A prompt-optimization / LLM-methods workshop at a major 2026 venue** (NeurIPS/ICLR/
  ACL efficient-methods, "advances in prompt optimization", DSPy/GEPA-adjacent) — best
  fit: the result is an optimizer-behavior finding, and GEPA's ICLR 2026 oral status
  (trends §9) means the community is primed for it. [JUDGMENT]
- **A benchmark/datasets workshop or the NeurIPS Datasets & Benchmarks track** (trends
  doc lists its contamination focus) — viable *if* the framing leans on the
  reseedable-contamination-free instrument rather than optimizer deltas. [JUDGMENT]
- **MAS-PromptBench-style venue / arXiv companion** as a small optimizer micro-benchmark
  [EVIDENCE: MAS-PromptBench 2026 exists]. [JUDGMENT] Realistic as a short paper.

[JUDGMENT] A **main-conference deductive-reasoning-benchmark** slot is *not* realistic:
the un-optimized baseline is a regeneration of a 2023 generator and P5/P7 already own the
2025 benchmark frontier. Workshop / short-paper is the honest ceiling.

### (e) What would NOT be publishable (recognize early)  [JUDGMENT]

- **Small or within-noise gains** (a few points, CIs overlapping across optimizers, or
  swamped by seed variance) — nothing to claim. Watch for this first.
- **Gains that vanish on fresh-seed regeneration** — indicates overfitting to a fixed
  instance pool; fatal for a contamination-framed task.
- **"Optimizer beats a weak zero-shot prompt" with no strong fixed-prompt baseline** —
  P5/P6 evidence that CoT/few-shot alone closes most of the gap makes this an easy
  reviewer rejection.
- **The un-optimized baseline alone**, presented as a deduction-benchmark result — adds
  nothing over PrOntoQA/JustLogic; no novelty (P5 fills the cell).
- **Ceiling/floor effects** — if base accuracy is already ~95% (P1 shows real-world/low-
  hop settings saturate) or already at chance (P3/P7 show deep instances near random),
  there is no headroom for an optimizer to move and the number is uninformative. Choose
  depths that sit in the informative mid-band.
- **A single model, single seed, single depth** — not a benchmark, not a finding.

---

## 4. Research-currency verdict  [EVIDENCE + stated caveats]

**Caveats on the numbers (state prominently):** all counts are **OpenAlex, fetched
2026-07-22**. OpenAlex **undercounts vs Google Scholar** (often 2–5×), and 2026 counts
are **near-zero by construction** (indexing lag) — a 2025/2026 paper reading 0 is *not*
evidence of no impact.

Citation data for the seven:

| Paper | cited_by | since_2025 | by_year highlights |
|---|---|---|---|
| P1 PrOntoQA (2023) | 39 | 6 | 2023:23, 2024:10, 2025:5, 2026:1 |
| P2 PrOntoQA-OOD (2023) | 17 | 6 | 2023:4, 2024:7, 2025:6 |
| P3 FLD (2023) | 2 | 1 | 2024:1, 2025:1 |
| P4 FLD×2 (2024) | 2 | 2 | 2025:2 |
| P5 JustLogic (2025) | 0 | 0 | — |
| P6 Multi-LogiEval (2024) | 0 | 0 | — |
| P7 SATBench (2025) | 1 | 1 | 2026:1 |

**Verdict** [JUDGMENT, on the evidence + caveats]: The family is **alive but maturing,
not surging.**
- The **anchor generator (PrOntoQA, P1) is still being cited in 2025 (5) and into 2026**,
  and its OOD extension (P2) actually shows a *rising* trend (2023:4→2024:7→2025:6) — the
  generator lineage c18 reuses is demonstrably current, not abandoned. This supports
  c18's "reuse a live codebase" premise.
- The **P3/P4 FLD line shows low OpenAlex counts (2 each)** — but these are almost
  certainly undercounts (FLD is a known ICML/NeurIPS line; GS would be far higher), so I
  treat the low numbers as weak evidence at most.
- **P5–P7 at 0–1 are uninformative by construction** (2024–2025 papers, indexing lag) —
  their *existence* as 2024–2025 entries is the real currency signal: JustLogic and
  SATBench are fresh 2025 competitors, confirming the sub-field is producing new
  benchmarks *right now*.

Net: the deductive-reasoning evaluation line is current enough that c18's lineage is not
stale, but it is a **crowded, well-trodden shape** with fresh 2025 entrants (P5, P7)
already occupying c18's near-neighborhood. That argues *against* pitching c18 as a novel
benchmark and *for* using it as an instrument in the more decisively-surging Area 9
(prompt-optimizer benchmarking: GEPA ICLR 2026 oral, MAS-PromptBench 2026) — where
currency is unambiguously high and self-referential to whetstone-ai's own method space.

---

## Executive summary (10 lines)

1. Three axes organize the family: output shape/verifier (A), content provenance (B),
   reasoning regime + label space (C); protocol (single vs multi-turn) is degenerate —
   all seven papers are single-shot.
2. The crowded cell is {rule-chaining × depth-dial × three-way label}: FLD, FLD×2,
   JustLogic, and c18's core stratum all sit there.
3. c18's distinguishing (empty) cell is the combination {fictional surface + label-only
   exact-match + reseedable-no-LLM generator + independent oracle + CSP stratum} — no
   single peer holds all of it.
4. JustLogic (P5) is the closest living competitor and nearly co-locates with c18's
   scoring design, so c18's task-*shape* novelty is thin and must be stated honestly.
5. P1/P2 own the generator lineage but grade full proofs; c18 trades that for cheap
   label-only exact match plus an imported open-world Unknown.
6. c18's cleanest non-overlapping add vs P3/P4 is a direct off-the-shelf inference-time
   T/F/Unknown accuracy number, which those training-focused papers never report.
7. If optimization yields large verified gains, the claim is an optimizer-behavior claim
   on a contamination-free depth-parameterized task — not a new benchmark or reasoning-
   science claim.
8. It belongs in the prompt-optimizer benchmarking line (Area 9: GEPA ICLR 2026 oral,
   MAS-PromptBench 2026), with the task as instrument, targeting a workshop/short paper —
   a main-conference benchmark slot is not realistic.
9. Reviewers will demand a strong fixed-prompt baseline, all three optimizers at matched
   rollout budget, per-depth breakdown, fresh-seed replication, floors/ceilings, and CIs;
   not-publishable outcomes are within-noise gains, gains that vanish on regeneration,
   wins over a weak baseline, or ceiling/floor-saturated depths.
10. Currency (OpenAlex, 2026-07-22, undercounts vs GScholar, ~0-by-construction for 2026):
    PrOntoQA 39 and PrOntoQA-OOD 17-rising confirm a live lineage, but 2025 entrants
    JustLogic/SATBench show a crowded frontier — favoring Area 9 as the publication path.
