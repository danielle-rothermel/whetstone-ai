# c19 — Positioning Synthesis (Grid-World State Prediction)

Scope: positions c19-as-designed against its five related-work papers, assesses publishability under an
optimization-gain scenario, and issues a research-currency verdict. **Evidence** (from dossiers, the c19
candidate file, the trends scan, and the citation manifest) is separated from **Judgment** (my inference)
throughout. Prepared 2026-07-22.

The five papers:

| ID | Short name | arXiv |
|----|-----------|-------|
| P1 | GRASP (Tang & Kejriwal) | 2407.01892 |
| P2 | LLM-BabyBench (Choukrani et al.) | 2505.12135 |
| P3 | GridRoute (Li et al.) | 2505.24306 |
| P4 | MazeEval (Einarsson, single-author) | 2507.20395 |
| P5 | Do LLMs Build Spatial World Models? (Li et al.) | 2604.10690 |

c19-as-designed (from the candidate file): full ASCII grid (~6x6) shown at once, initial robot pose, a
fixed command string mixing standard moves with 2-3 **invented commands** (jump-two, wrap) under varied
**wrap-vs-clamp / turn-relative / indexing conventions**; the model predicts exactly **one derived fact**
(final coordinate, heading, or carrying-flag) as a single token, scored **strict 0/1 exact-match** against
an independent Minigrid-based oracle. No path, no interaction loop.

---

## 1. The 2x2 map

Axes: **Observation** (full = whole environment given upfront; partial = only local readings + history)
x **Protocol** (single-shot = one LLM call produces the full answer, no environment feedback; interactive
= closed action loop with per-step environment feedback).

Note on classification (Evidence): all dossiers classify "single-shot" at the **API-call level** — one
call, no environment feedback loop — regardless of whether the *output* is a single fact or a full plan.
Under that convention:

```
                       SINGLE-SHOT (one call, no env loop)      INTERACTIVE (closed loop, per-step feedback)
                   +-------------------------------------------+---------------------------------------------+
  FULL             |  P1 GRASP        (output: full plan)      |                                             |
  OBSERVATION      |  P2 LLM-BabyBench/Predict (final state)   |   (empty)                                   |
  (whole grid      |  P3 GridRoute    (output: full path)      |                                             |
   given upfront)  |  P5 Spatial World Models (path / T-F)     |                                             |
                   |  >>> c19  (output: ONE derived fact) <<<  |                                             |
                   +-------------------------------------------+---------------------------------------------+
  PARTIAL          |                                           |  P4 MazeEval (coord + wall-distance +       |
  OBSERVATION      |   (empty)                                 |              full history, move() loop)      |
  (local readings) |                                           |                                             |
                   +-------------------------------------------+---------------------------------------------+
```

**Crowded cell (Evidence).** Full-observation x single-shot holds **four of five** papers (P1, P2, P3, P5)
plus c19. All four present the complete environment in one prompt and elicit the whole answer in one
generation; dossiers state this explicitly for each.

**Sparse cells (Evidence).** Partial x interactive holds exactly one paper (P4 MazeEval — `move()`
function-calling loop, local wall-distance only, never a global map). The two right-hand/lower cells
(full x interactive; partial x single-shot) are empty across this set. P2's *other* two splits (Plan,
Decompose) are simulator-executed for verification but the LLM still emits its whole output in one call
with no mid-generation feedback; dossier treats them as full-observation with interactive *verification*,
not an interactive LLM protocol — so they do not populate the interactive cell in the sense used here.

**Where c19 sits (Evidence + Judgment).** c19 is in the crowded full x single-shot cell. Its
**differentiator is not the cell but the output granularity within it**: it is the only design in the set
whose scored output is a **single derived fact under strict 0/1 exact match**. P2/Predict is the nearest —
final agent position+direction — but grades with Manhattan-distance partial credit and uses standard
BabyAI actions. P1 and P3 score full multi-step plans/paths (many-valid-answer ambiguity). P5 Task 1 scores
full-path exact match (conflates planning with state-tracking); its Task 2/3 True-False proximity probes are
closer in spirit but test cross-query reuse, not single-fact simulation.

**Judgment.** The genuinely open territory c19 occupies is *within* the crowded cell: single-derived-fact,
strict-exact-match, **plus an invented-command / wrap-vs-clamp convention layer that none of the five test**.
No paper in this set combines (a) single-fact exact-match scoring with (b) non-memorizable movement
semantics. That intersection is c19's positioning claim, and it is defensible against all five.

---

## 2. Per paper: what c19-as-designed adds (honest about overlap)

### P1 — GRASP (2407.01892)
**Overlap (Evidence).** Same cell (full x single-shot), both use coordinate-labeled ASCII grids with an
independent oracle, both are motivated by the same critique that description-based spatial QA (SpartQA,
StepGame) tests interpretation not use. Both note off-by-one/edge/obstacle-adjacent errors as diagnostic.
**What c19 adds (Evidence).** GRASP scores a **full 20-step foraging plan** on continuous energy/step-cost
metrics — a planning+optimization objective, not state tracking — on fixed 11x11 grids with a *conventional*
action set (UP/DOWN/LEFT/RIGHT/TAKE/DROP) and no invented-rule or wrap/clamp variation. c19 replaces the
multi-valued plan with one exact-match fact, swaps optimization for pure deterministic simulation, shrinks
to ~6x6 for surgical per-rule diagnosis, and adds the invented-command / convention strata GRASP lacks.
**Honest note (Judgment).** The gap is real and structural (foraging-optimization vs state-tracking), so
overlap is thematic, not methodological. The dossier's own verdict downgrades the candidate file's "direct
antecedent" language to "relative in the same family" — c19 should cite GRASP as motivation, not as the
task it reframes.

### P2 — LLM-BabyBench / Predict split (2505.12135)
**Overlap (Evidence).** This is the **highest-overlap** paper and the acknowledged template. Same cell,
"very high structural overlap": full/omniscient state + initial pose + ordered action sequence → predict one
final-state fact, seeded/procedural, with a representation-format comparison (Narrative/Structured/JSON).
**What c19 adds (Evidence).** (a) an **invented-command layer** — LLM-BabyBench uses only standard, publicly
documented BabyAI actions a model could have memorized; c19's jump/wrap semantics cannot be. (b) an explicit
**wrap-vs-clamp boundary stratum** with no counterpart in BabyAI's wall/door model. (c) a **literal ASCII
grid** observation — LLM-BabyBench never renders a character grid, only prose/bullet/JSON object-lists; its
format study has no rendered-grid arm. (d) strict **0/1 exact match** vs its Manhattan partial credit.
(e) much smaller/cheaper grids (~6x6 vs up to 22x22) and a run-verified Minigrid harness.
**Honest note (Judgment).** This is the paper c19 most needs to differentiate from, and the differentiation
rests almost entirely on the invented-command/convention layer and the ASCII-rendering arm. Strip those and
c19 is a smaller, cheaper re-run of the Predict split. The novelty is therefore **conditional on the strata
actually mattering** (i.e., the naive-prompt floor must drop *because of* invented commands, not despite
them). P2's own data — five of seven models at 85-97.5% on Easy standard-semantics Predict — is direct
evidence that standard semantics alone do NOT floor frontier models, which is exactly why c19 needs the
strata; the same data warns that without them the floor is too high.

### P3 — GridRoute (2505.24306)
**Overlap (Evidence).** Same cell; deterministic cardinal-movement grid with a programmatic oracle
(constrained-diagonal Dijkstra); both study how in-prompt scaffolding affects grid accuracy; both note
naive/vanilla prompting is not floor-level (GridRoute Vanilla FR up to 76 at 10x10).
**What c19 adds (Evidence).** GridRoute outputs a **full path** (Feasibility/Optimal-Ratio metrics with
path ambiguity), never renders an ASCII grid (obstacles given as rectangle-corner coordinates), has no
invented commands, no wrap/clamp, no heading or carrying-flag state, and uses 10-30 sized grids. c19 adds
single-fact exact match, ASCII rendering, invented/nonstandard semantics, orientation+carry state, and
small grids. GridRoute is centrally an **algorithm-in-prompt** study (AoP: A*/Dijkstra/DFS embedded in the
prompt) — a different research question from c19's.
**Honest note (Judgment).** Lowest methodological overlap of the four same-cell papers; the shared surface
is "cardinal grid + oracle." GridRoute is most useful to c19 as a **cautionary datum for the optimization
phase**: its Algo-Reasoning worked-trace prompt is a *trade-off* (helps GM/MSE, hurts FR/OR, and produces
0/0/0 on a 7B model), and AoP-DFS collapses GPT-4-Turbo to CR/FR/OR 1-3. In-prompt scaffolding can backfire —
directly relevant if c19 later tests optimizer-produced prompts.

### P4 — MazeEval (2507.20395)
**Overlap (Evidence).** Only the shared genre: deterministic, seeded, coordinate grid navigation, oracle-
verified, contamination-resistant by construction, cardinal moves. c19's own related-work note cites
MazeEval as "directly probing the grid state-prediction floor c19 depends on."
**What c19 adds / how it differs (Evidence).** **Opposite quadrant on both axes** — MazeEval is partial-
observation (local wall-distance + history, never the map) and interactive (up to O(n^2) `move()` calls,
loop/budget termination). Its dominant finding — 100% of non-O3 failures are *looping* from failure to
integrate history across many turns — is **structurally impossible in c19's single-shot design** (no turns
to lose track of).
**Honest note (Judgment).** c19 doesn't "add to" MazeEval so much as **isolate the sub-capability MazeEval's
own Discussion (5.4) flags as untested**: can a model derive one final-state fact from a fully-specified
map+command string in one shot, with no long-horizon interactive memory? MazeEval shows local per-step
reasoning is intact (low wall-hit rate) but cross-turn integration fails; c19 tests whether the same
integration must be done *inside one forward pass* and whether removing the interaction loop (while keeping
equivalent information) raises or lowers the failure rate. This is a clean complementarity story, and the
strongest "orthogonal contribution" framing available to c19 — but note MazeEval also introduces a
cross-linguistic axis and O3-as-clean-ceiling result that c19 does not touch.

### P5 — Do LLMs Build Spatial World Models? (2604.10690)
**Overlap (Evidence).** Same cell; small seeded text grid-mazes (5x5-9x9) shown in full, **representation-
format comparison** (adjacency-list vs visual-grid: 86% vs 34% on 5x5 CoT for Gemini — a 2-5x swing),
exact-match scoring, and the same "genuine world model vs surface heuristic" motivation. c19's candidate
file cites it as validating c19's "format-sensitivity and headroom claims."
**What c19 adds (Evidence).** P5's Task 1 scores **full-path exact match** (conflates planning with state
tracking); c19's single fact isolates state tracking. P5 uses **standard adjacency movement only** — no
invented commands, no wrap/clamp — so a model with memorized "standard maze semantics" is not stress-tested
against convention variation. P5's heavy apparatus (semantic-coverage / ROUGE-L / LLM-judge reuse probes) is
out of scope for a cheap single-fact task; c19's ~150-line Minigrid harness is deliberately lighter.
**Honest note (Judgment).** P5 is the **most current and most threatening** overlap: a 2026 paper already
delivering c19's headline empirical claim (format/prompt sensitivity + headroom on small grids). c19 cannot
claim "grids are format-sensitive" as a contribution — P5 owns that. c19's remaining differentiators are
narrow but real: single-fact exact-match isolation of state-tracking, and the non-memorizable convention
layer. If c19 wants a defensible spatial-reasoning contribution *distinct from P5*, it must lean on the
invented-command/convention manipulation, because the format-sensitivity finding is already published.

---

## 3. Publishability analysis (IF optimization produces large verified gains)

Setup (Evidence, from the task brief): Phase 1 is an **un-optimized baseline** — vanilla Minigrid, reseeded,
naive + ceiling prompts, strict 0/1. Phase 2 *potentially* prompt-optimizes with COPRO / MIPROv2 / GEPA.
This section addresses the branch where Phase 2 yields **large, verified** gains.

### (a) The workshop-paper claim, stated precisely (Judgment)
> "On a contamination-resistant, single-derived-fact grid-state-prediction task with non-memorizable
> movement conventions (invented commands + wrap/clamp strata), automated prompt optimization closes a large
> fraction of the gap between a naive prompt and a hand-written all-rules ceiling prompt — recovering X of Y
> points — WITHOUT any change to model weights or task data, and the recovered gain is attributable to the
> optimizer surfacing the latent convention rules rather than to memorized grid heuristics."

Precision requirements for the claim to hold: (i) gain measured against **both** a naive floor and a
hand-written ceiling on **held-out seeds** (train/test seed split, no instance leakage); (ii) gain replicated
across **>=2 optimizers** (e.g. MIPROv2 and GEPA) and **>=2 models** to avoid single-config luck; (iii)
temperature-0, fixed decoding, reported with seed-variance / CI; (iv) an ablation showing the gain **shrinks
on the standard-semantics stratum and concentrates on the invented-command/wrap-clamp strata** — i.e. the
optimizer is learning the *convention*, not exploiting a format artifact.

### (b) Which publication line (Judgment, grounded in trends area 9)
**Primary: prompt-optimizer benchmarking** (trends area 9). Evidence for receptiveness: GEPA is an **ICLR
2026 oral** (2507.19457; reports beating MIPROv2 by ~13% aggregate, GRPO by up to 20% at 35x fewer rollouts);
**MAS-PromptBench** (2606.23664, 2026) is a benchmark *specifically* for "when does prompt optimization help";
**DD-GEPA** and **AIR** (2026) show active forking. A clean, cheap, contamination-proof task where optimizers
produce a *large, decomposable, verifiable* gain is exactly what this line wants — c19's single-fact
exact-match + independent latent rules make the gain attributable and auditable in a way free-form tasks are
not.
**Secondary/weaker: spatial-reasoning evaluation.** Evidence against leading with this line: P5 (2026) already
published the format/headroom finding; GRASP/GridRoute/MazeEval/BabyBench cover the evaluation angle. As a
*spatial-reasoning* paper the marginal contribution is thin. **Judgment: lead with optimizer-benchmarking,
use spatial reasoning as the substrate/motivation, not the headline.**

### (c) Baselines, ablations, comparisons reviewers will demand (Judgment)
- **Baselines:** naive prompt (floor); hand-written all-rules ceiling prompt; a random/majority-class
  baseline for the single-fact target (heading has ~4 classes, carry-flag 2 — reviewers will check the fact
  isn't near-guessable); a simple CoT prompt (to show the optimizer beats trivial reasoning scaffolding, cf.
  P5 where CoT alone recovers Claude-Haiku from ~0 to 78%). **Critical:** a CoT/ceiling baseline is
  non-negotiable given P5 and P2 both show plain CoT/ToT already moves scores enormously.
- **Optimizer comparison:** at least MIPROv2 vs GEPA (the two the trends doc foregrounds), ideally + COPRO;
  report cost (rollouts/API calls) not just final score — GEPA's whole claim is efficiency.
- **Ablations:** per-stratum gain decomposition (standard vs invented-command vs wrap/clamp vs indexing);
  train/test seed split proving no instance memorization; model-transfer (does an optimized prompt transfer
  across models or overfit one); sensitivity to grid size; the representation arm (ASCII vs coordinate-list)
  that P2/P5 both found swings results 2-5x — reviewers will ask whether the gain survives representation
  change.
- **Comparisons:** position against P2/Predict and P5 numerically where possible (same small-grid regime);
  cite MAS-PromptBench methodology as the comparison template.
- **Backfire check:** report cases where optimization *hurt* (GridRoute's Algo-DFS collapse and 7B 0/0/0 are
  the cautionary precedents reviewers know) — showing you looked strengthens the paper.

### (d) Realistic venues / workshops and why (Judgment)
- **ICLR / NeurIPS / ACL workshops on LLM agents, prompt optimization, or evaluation** — best fit; the
  optimizer-benchmarking framing rides the GEPA/DSPy wave and a small clean benchmark is workshop-sized.
- **NeurIPS Datasets & Benchmarks track** (the trends sources list its contamination-focused CFP) — only if
  the task is released as a reusable benchmark with a leaderboard, not just a one-off study.
- **A DSPy / prompt-optimization–adjacent workshop or the venues citing MAS-PromptBench** — most receptive to
  "here is a task where optimizers produce a large, decomposable, attributable gain."
- **Why workshop not main-track:** the task is a reframing of a well-trodden family (dossiers: "careful
  reframing rather than a new problem"); the contribution is a clean instrument + an optimizer result, which
  is workshop-scaled unless it grows a full benchmark suite.

### (e) What would NOT be publishable — recognize early (Judgment)
- **Small or noisy gains.** If optimization moves the score only a few points, or within seed variance, there
  is no story — vanilla prompt-optimization papers reporting marginal gains are not novel in 2026.
- **Gain that is really a format artifact.** If the per-stratum ablation shows the gain comes from the
  *standard-semantics* stratum or from representation formatting (the P5/P2-documented 2-5x format swing),
  then c19 has re-derived a known result (format sensitivity), not shown convention-learning — not publishable
  as a new claim.
- **Naive floor already near ceiling.** If frontier models solve the naive-prompt version at 85-97% (as P2
  found for standard-semantics Predict), there is no gap to close and nothing for the optimizer to do — this
  is the single biggest early-kill risk and the exact soft spot round 1 flagged.
- **Single-config result.** A gain on one optimizer + one model, not replicated, reads as luck and will be
  rejected.
- **Un-attributable gain.** If you cannot show *which rule* the optimizer surfaced (the whole point of the
  independent-latent-rule design), the exact-match-diagnostic advantage over free-form tasks is wasted and the
  result is just another "optimizer helped on task X" data point.

**Early-warning gate (Judgment):** after Phase 1, if the naive floor is high (>~80% on frontier models) OR the
ceiling prompt barely beats naive, STOP before Phase 2 — the headroom that makes the optimizer story possible
is absent, and P2's Easy-level numbers say this is a live risk.

---

## 4. Research-currency verdict

**Citation evidence (Evidence).** OpenAlex counts, fetched **2026-07-22**. Stated caveats apply: OpenAlex
**undercounts** relative to Google Scholar, and 2026 papers are **near-zero-by-construction** (indexing lag —
a 2026 paper has had almost no time to accrue or have its citations indexed).

| Paper | arXiv | cited_by | since_2025 | by_year | Note |
|-------|-------|---------|-----------|---------|------|
| P1 GRASP | 2407.01892 | 3 | 2 | 2024:1, 2025:2 | 2024 paper; low but nonzero, trending up |
| P3 GridRoute | 2505.24306 | 1 | 1 | 2026:1 | mid-2025 paper |
| P2 LLM-BabyBench | 2505.12135 | 0 | 0 | — | mid-2025 paper, not yet indexed-cited |
| P4 MazeEval | 2507.20395 | 0 | 0 | — | mid-2025 paper, single-author |
| P5 Spatial World Models | 2604.10690 | 0 | 0 | — | **2026** — zero-by-construction |

Anchor/lineage papers for context (Evidence, same manifest): bAbI (1502.05698) cited_by 727 (since_2025 11);
EntNet (1612.03969) 157 (since_2025 0); Entity Tracking (2305.02363) 1 (since_2025 1).

**Verdict (Judgment).** The **direct 2025-2026 corpus is genuinely current but citation-immature**: four of
the five papers are 2025-2026 preprints, three with literally zero indexed citations. Read naively this looks
weak; read with the stated caveats it is the *expected* signature of a **live, fast-moving, recently-active
area** — papers are appearing faster than the citation graph indexes them, not an area that has gone quiet.
The clinching signal is P5, a **2026** paper still probing exactly c19's construct (small-grid format
sensitivity + headroom); an area does not spawn a fresh 2026 entry if it is dead. The deep lineage
(bAbI 727, EntNet 157) confirms the *problem family* is heavily established, so c19 sits on a well-cited
foundation with a still-active frontier.

**Two honest caveats on currency (Judgment).** (1) The activity is **crowded, not open** — the same recency
that proves currency (P2, P4, P5 all 2025-2026) also means c19's evaluation-angle contribution is largely
pre-empted, especially by P5. Currency cuts both ways: live area, but little unclaimed evaluation territory.
(2) OpenAlex zeros mean we **cannot** distinguish "ignored" from "too new" for P2/P4/P5 from this data alone;
a Google Scholar cross-check would be needed before making any strong "high-impact" claim. On this data the
defensible statement is "active and current," not "high-impact."

---

## Executive summary (10 lines)

1. Four of five papers (GRASP, LLM-BabyBench, GridRoute, Spatial-World-Models) plus c19 occupy one crowded
   cell: full-observation x single-shot; MazeEval alone sits in partial x interactive; two cells are empty.
2. c19's differentiator is not the cell but the output: it is the only design scoring a SINGLE derived fact
   under strict 0/1 exact-match, plus an invented-command / wrap-vs-clamp convention layer none of the five test.
3. Highest overlap is LLM-BabyBench/Predict (the acknowledged template); c19 adds non-memorizable commands, a
   wrap/clamp stratum, literal ASCII rendering, and exact-match — strip those and it is a cheaper re-run.
4. Most threatening overlap is the 2026 Spatial-World-Models paper, which already published c19's format-
   sensitivity + headroom finding; c19 must lean on convention-manipulation, not format-sensitivity.
5. MazeEval is the cleanest complementarity: c19 isolates the single-shot state-derivation MazeEval's own
   Discussion flags as untested, since MazeEval's looping failure is structurally impossible in one shot.
6. IF optimization gains are large+verified, the precise claim is: automated prompt optimization recovers a
   large fraction of the naive→ceiling gap with no weight/data change, attributable to surfacing latent rules.
7. Lead with the prompt-optimizer-benchmarking line (GEPA ICLR-2026 oral; MAS-PromptBench 2026), not the
   spatial-reasoning line, which P5 has largely pre-empted; target agent/optimization/eval workshops.
8. Reviewers will demand: naive+ceiling+CoT baselines, MIPROv2-vs-GEPA with cost, per-stratum gain
   decomposition, held-out seed split, model-transfer, and a representation-arm and backfire check.
9. NOT publishable: small/within-variance gains, gains that are format artifacts or on standard-semantics
   strata, a naive floor already near ceiling (P2 shows 85-97% is a real risk), or single-config results.
10. Currency verdict (OpenAlex, 2026-07-22, undercounts vs Scholar, 2026≈zero-by-construction): four 2025-26
    preprints, three with zero indexed citations = the signature of a LIVE but CROWDED area on a well-cited
    (bAbI 727, EntNet 157) foundation — "active and current," not yet demonstrably "high-impact."
