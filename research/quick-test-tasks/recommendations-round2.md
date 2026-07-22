# Quick-Test Task Recommendations — Round 2 (Implementation-Aware)

Date: 2026-07-21 · whetstone-ai

> **Amended 2026-07-21.** The round-1 tie-breaker=1 exclusion is dropped by user
> decision: **c10, c04, and c02** are folded into the official ranking on equal terms
> with the other 21 candidates — their I1–I6 scores were produced by the same pipeline
> (web discovery, desk scoring with the same rubric and exercises, deep active-path
> code reads under [`repos/`](repos/)). Run verification carries over at the repo
> level for c10 and c04: their best repo, open-thought/reasoning-gym, was run-verified
> during round 2 under c24 (same-seed byte-identical, 6/6 independent ground-truth
> checks), while their candidate-specific task logic (interlocking-cycle semantics for
> c10, rewrite-system tasks for c04) is authored on top of the reused infra; c02's
> NUMCoT scored I1=1 and never met the run-verification bar. No previously published
> relative order changes — the three insert at ranks 5 (c10), 7 (c04), and 12 (c02),
> giving 20 ranked + 4 gated = 24 — and the adoption call (**c19 adopt, c18 backup**)
> is unchanged.

This is the **round-2** recommendation: a rescoring of the 24 candidates by
**implementation difficulty**, layered on top of the round-1 rubric ranking.
(Initially published over the 21 candidates surviving the round-1 tie-breaker filter;
the amendment above folds the three tie-breaker-1 candidates back in.) Round 1
([`recommendations.md`](recommendations.md)) is preserved unchanged; this document
supersedes its adoption call but not its design analysis. Round 2 adds what round 1
deliberately deferred: per-candidate web discovery of reusable codebases, deep
active-path code reads, and run verification of the finalist generators.

## Bottom line

**Adopt c19 — Grid-World State Prediction, built on Farama-Foundation/Minigrid
(implementation total 14/16, the only candidate combining a run-verified I1=3 codebase
with verification-only skeptic debt).** Minigrid is Apache-2.0, actively maintained,
properly seed-threaded (verified byte-identical reproduction), and our entire diff is
~120–200 lines of config + oracle + scoring glue outside the repo — no edits to
Minigrid itself. **Backup: c18 — Depth-Controlled Synthetic Deduction on
asaparov/prontoqa (13/16)**, the other run-verified I1=3 generator (~100 lines of
external glue, `--seed` → deterministic JSON with gold answers). Notably, the prontoqa
harness built for c18 also serves **c21**, the round-1 winner (now #6 at 11/16), so the
backup path preserves the option of returning to round 1's theoretical favorite at
marginal cost. The round-1 top tier fell hard here: c21 drops on generator intricacy
and calibration burden, and c12/c14 (round-1 28-scorers) fall to the bottom because no
reusable codebase exists for either.

## Methodology

### The implementation-difficulty rubric (higher = easier/safer)

- **I1 (0–3, double-weighted) — Reusable existing codebase.**
  3 = ALL of: maintained; license permissive or missing-but-paper-attached; actively
  used in 2025–2026 work; our adaptation diff is config + scoring glue; active-path
  code well-written and reliable; likely to work as claimed based on a **direct deep
  code read** (never from README/stars/citations). 2 = genuinely reusable but at least
  one property missing. 1 = code exists but major surgery needed, or frozen dataset
  with partial generator code. 0 = no usable codebase.
- **I2 (0–2) — Oracle independence & verifiability.** 2 = ground truth checkable via a
  second independent path (brute force / exhaustive closure / trivially-different
  reference impl) AND human hand-verifiable in under a minute; 1 = independent check
  costly/partial; 0 = truth defined only by the generator.
- **I3 (0–2) — Moving-part count of the adapted design** (incl. round-1 mandated foil
  banks, rejection samplers, discoverability checks, stratified sampling). 2 = one
  sampler + tiny oracle; 1 = modest machinery; 0 = multi-component system.
- **I4 (0–2) — Calibration/pinning burden** (count of pre-freeze empirical gates:
  measured ceilings, per-stratum floors, dead-strata audits, red-team runs).
  2 = 0–1 gates; 1 = 2–3; 0 = 4+.
- **I5 (0–2) — Change-tolerance.** 2 = rules/strata independent, a change regenerates
  one stratum with a local manifest diff; 1 = partial coupling; 0 = any change forces
  full pool regeneration + recalibration.
- **I6 (0–2) — Redesign debt from round-1 skeptic verdicts.** 2 = verification-only
  fixes (or no repair flags); 1 = bounded redesign; 0 = structural redesign that
  changes what the task is.
- **TOTAL = 2·I1 + I2 + I3 + I4 + I5 + I6, max 16.**

### Why I1 is double-weighted

The user's stated priority: being able to reuse a clean existing benchmark codebase
with minimal edits — especially one currently in use in 2025–2026 work, which makes
fast publishing much likelier — is roughly **2× more important** than a task being
theoretically somewhat nicer. The double weight encodes that directly. The
implementation total is the **primary sort key**; round-1 rubric score and tie-breaker
only break ties.

### Round-1 verdict health is a gate, not a score

A candidate whose measurement story was **refuted without a concrete rescue** in round
1 cannot be recommended regardless of implementation ease. Four candidates are gated:
**c01, c05, c07, c15** (both skeptic lenses refuted at major severity; the only fixes
are structural redesigns that change what the task is). They appear in the table for
completeness, unranked. Bounded-redesign candidates stay eligible — their I6 already
prices the debt.

### The tie-breaker pool filter — applied initially, dropped by amendment

Round 2's pool was initially filtered to the 21 candidates carrying round-1
tie-breaker 2: c02, c04, and c10 (tie-breaker 1) were excluded from final
consideration before the ranking was built. The 2026-07-21 amendment drops that
exclusion by user decision, folding all three into the ranking on equal terms —
their I1–I6 cells come from the same pipeline (steps 1–5 below) as every other
candidate's.

### How the scores were produced

1. **Desk scoring (I2–I6)** from the round-1 corpus: `candidates-merged.md`,
   `landscape.md`, `recommendations.md`, and the per-domain research docs, with cited
   evidence per cell.
2. **Web discovery** per candidate (GitHub, arXiv, Papers with Code, HuggingFace) for
   re-seedable generators, oracles, and checker libraries; every repo verified to
   exist and characterized (seeded-generator / frozen-dataset / checker-library /
   reference-impl).
3. **Deep active-path code reads** of each candidate's best repo by dedicated agents:
   seed plumbing traced call-by-call, oracle independence assessed, tests run where
   present, adaptation diff estimated in LOC, red flags logged with file:line. Written
   up under [`repos/`](repos/).
4. **Run verification** of finalist generators in fresh environments: same-seed
   byte-identical determinism, different-seed variation, and independent ground-truth
   spot checks (verified for reasoning-gym, json-schema-faker, BeyondBench, prontoqa,
   Minigrid).
5. **I1 finalization**: start from the deep-read provisional I1; a failed run
   verification caps I1 at 1; a confirmed run keeps it; no repo found = 0.

**I1 adjustments made:** c19 kept at 3 (Minigrid run confirmed). c18/c21 kept at 3
(prontoqa run confirmed). c17 kept at 2 (BeyondBench run confirmed — including runtime
confirmation of its XOR-dropping bug). c01/c24 kept at 2 (reasoning-gym run confirmed).
c11 kept at 2 (json-schema-faker run confirmed, 519/519 tests green). c23 and c07 held
at 1: InductionBench has three **verified import-time crashes** and PYTHONHASHSEED
nondeterminism — the failed-run cap binds at 1 (and c07's fit to that repo is
"essentially nil" per the deep read). c12, c13, c14, c16 scored 0 — no plausibly
reusable repo found. Unverified provisionals kept as read: c03=2, c05=2, c06=2, c20=2,
c22=2, c08=1, c09=1, c15=1.

**Amendment fold-ins (2026-07-21):** c04 scored I1=3 — reasoning-gym ships `ab.py`, a
genuine term-rewriting-to-normal-form generator on the framework run-verified under
c24, so the repo-level verification carries over (the seeded rule-table generalization
is a ~150-LOC clone-and-generalize authored on top). c10 held at 2 on the same
run-verified reasoning-gym base: the infrastructure's determinism is confirmed, but
the invented interlocking-cycle task exists nowhere in-repo and must be authored on
top of the reused infra. c02 scored I1=1: NUMCoT's generator is unseeded dead code
with a provably buggy oracle (`solve_hard`), so it was never run-verified — below the
I1≥2 bar.

**One caveat:** c13's desk-scoring agent failed mid-workflow, so its cells were
initially estimated during synthesis. A dedicated follow-up agent desk-scored it from
the round-1 corpus afterward (I2=2 — trivial second oracle via exhaustive feature-grid
closure; I3=1 — shadowed-rule coverage check needed under first-match; I4=1 — ~2–3
gates; I5=1 — first-match ordering partially couples rules; I6=2 — no repair flags),
confirming the estimates cell-for-cell. Its I1=0 is firm — no reusable repo exists,
which alone leaves it in the bottom tier.

## Full rescore table

Sorted by implementation total (primary); ties broken by I1 (per the reuse priority),
then round-1 score, then round-1 tie-breaker. c09 edges c08 on salvageable-code
substance (~320 LOC of CipherBank primitives vs <30 LOC from LINGOLY). Movement is
relative to the candidate's round-1 tier (28-tier finalists / 27-tier / 26-tier); the
three amendment fold-ins are marked ◆.

| Rank | ID | Candidate | R1 score/TB | I1 | I2 | I3 | I4 | I5 | I6 | Impl total | Movement vs R1 |
|:---:|----|-----------|:----:|:--:|:--:|:--:|:--:|:--:|:--:|:----:|----------------|
| **1** | **c19** | **Grid-World State Prediction** | 27/2 | **3** | 2 | 1 | 1 | 2 | 2 | **14** | ▲▲ 27-tier mid-pack → adopt |
| **2** | **c18** | **Depth-Controlled Synthetic Deduction** | 27/2 | **3** | 2 | 1 | 1 | 1 | 2 | **13** | ▲▲ 27-tier → backup |
| 3 | c17 | Invented-Gate Circuit Evaluation | 27/2 | 2 | 2 | 2 | 1 | 2 | 2 | 13 | ▲ 27-tier → podium |
| 4 | c24 | Composite-Key Sorting & Canonical Ordering | 27/2 | 2 | 2 | 2 | 1 | 1 | 2 | 12 | ▲ 27-tier → top 5 |
| 5 | c10 | Interlocking-Cycle Calendar & Clock Arithmetic | 27/1 | 2 | 2 | 1 | 1 | 2 | 2 | 12 | ◆ folded in from TB-filter exclusion → #5 |
| 6 | c21 | Relational Micro-World (Invented Vocabulary) | 28/2 | 3 | 2 | 0 | 0 | 1 | 2 | 11 | ▼▼ round-1 winner → #6 |
| 7 | c04 | Term-Rewriting to Normal Form | 26/1 | 3 | 2 | 1 | 0 | 1 | 1 | 11 | ◆ folded in from TB-filter exclusion → #7 |
| 8 | c11 | House-Convention Canonicalizer | 28/2 | 2 | 2 | 1 | 1 | 2 | 1 | 11 | ▼ 28-tier finalist → #8 |
| 9 | c22 | Stacked Verifiable-Constraint Micro-Generation | 26/2 | 2 | 2 | 1 | 1 | 1 | 2 | 11 | ▲ 26-tier → #9 |
| 10 | c03 | Keyed Custom-Operator Expression Evaluation | 27/2 | 2 | 2 | 1 | 1 | 1 | 1 | 10 | ▼ round-1 backup → #10 |
| 11 | c23 | Hidden-Rule String-Transform Induction Ladder | 26/2 | 1 | 2 | 1 | 1 | 2 | 2 | 10 | ▲ 26-tier → #11 |
| 12 | c02 | Fictional Mixed-Radix Unit Conversion | 26/1 | 1 | 2 | 1 | 1 | 2 | 2 | 10 | ◆ folded in from TB-filter exclusion → #12 |
| 13 | c06 | Artificial-Grammar Violation Classification | 28/2 | 2 | 2 | 1 | 0 | 1 | 1 | 9 | ▼ 28-tier finalist → #13 |
| 14 | c09 | Composable Cipher-Chain Decode | 26/2 | 1 | 2 | 1 | 1 | 1 | 2 | 9 | ▲ slight |
| 15 | c08 | Rosetta-to-Match-Up MC Linguistic Puzzles | 26/2 | 1 | 2 | 1 | 1 | 1 | 2 | 9 | ▲ slight |
| 16 | c20 | Invented Micro-Game Turn Resolution | 27/2 | 2 | 1 | 0 | 1 | 1 | 1 | 8 | ▼ 27-tier → #16 |
| 17 | c13 | Precedence-Ordered Decision-List Classification | 27/2 | 0 | 2 | 1 | 1 | 1 | 2 | 7 | ▼▼ no reusable repo exists |
| 18 | c16 | Wason Implication-Compliance Classifier | 27/2 | 0 | 2 | 1 | 1 | 1 | 1 | 6 | ▼▼ no reusable repo exists |
| 19 | c14 | Instruction-Hierarchy Priority Ledger | 28/2 | 0 | 2 | 1 | 0 | 1 | 1 | 5 | ▼▼▼ 28-tier finalist → near-bottom |
| 20 | c12 | Sigil-Schema Field Extraction & Dialect Routing | 28/2 | 0 | 1 | 0 | 1 | 1 | 1 | 4 | ▼▼▼ 28-tier finalist → bottom |
| — | c01 | Invented-Glyph Positional Base Arithmetic | 27/2 | 2 | 2 | 1 | 1 | 1 | 0 | (9) | **GATED** — refuted, no concrete rescue |
| — | c05 | Deterministic Execution & State Tracking | 28/2 | 2 | 2 | 1 | 0 | 1 | 0 | (8) | **GATED** — refuted, no concrete rescue |
| — | c07 | Case-Marked Micro-Conlang Translation | 27/2 | 1 | 2 | 1 | 1 | 1 | 0 | (7) | **GATED** — refuted, no concrete rescue |
| — | c15 | Opaque-Codebook Remapped-Label Classification | 28/2 | 1 | 2 | 1 | 1 | 1 | 0 | (7) | **GATED** — refuted, no concrete rescue |

## Top-5 detail

*(Written for the pre-amendment top five; headings carry the amended ranks. The
folded-in c10 (#5) and c04 (#7) share the run-verified reasoning-gym base detailed
under c24 — see the amendment entries in the movers section below.)*

### 1 · c19 — Grid-World State Prediction (14/16) — ADOPT

- **Codebase:** [Farama-Foundation/Minigrid](https://github.com/Farama-Foundation/Minigrid),
  Apache-2.0, v3.1.0, actively maintained. Deep read + run verification:
  [`repos/farama-foundation-minigrid.md`](repos/farama-foundation-minigrid.md).
  Seed is properly threaded (`reset(seed)` → `self.np_random`, all generation
  randomness routed through it, no module-global seeding). Run-verified: seed 42
  byte-identical SHA-256 across runs on five envs; seed 42 vs 43 differ; 3/3
  ground-truth spot checks via an independent object-model walk. I1 = 3.
- **Adaptation diff:** **zero edits inside `minigrid/`**. ~120–200 lines of new glue in
  our package: instance generation (`reset(seed)` + `pprint_grid`/`grid.encode`), an
  independent derived-fact oracle, strata over env id/size, optional nonce remap of
  the symbol vocabulary, single LLM call + strict 0/1 scorer.
- **Remaining risks:** (a) the invented-command / wrap-vs-clamp convention strata that
  defeat one-shot guessing are **our net-new simulation-layer logic** — Minigrid
  supplies generation, serialization, and determinism, not the nonstandard dynamics;
  (b) use stochastic-layout envs (Fetch, Crossing, FourRooms, Empty-Random) — fixed
  `Empty-8x8` does not vary by seed; (c) avoid the off-path unseeded spots
  (`wrappers.py:800`, the WFC subpackage); (d) ~2–3 pre-freeze gates remain (measured
  ceiling, per-stratum floors, single-fixed-convention red-team); (e) minor
  license-metadata mismatch (LICENSE=Apache-2.0, pyproject says MIT) — cite the
  LICENSE file.

### 2 · c18 — Depth-Controlled Synthetic Deduction (13/16) — BACKUP

- **Codebase:** [asaparov/prontoqa](https://github.com/asaparov/prontoqa), Apache-2.0,
  maintained, actively cited. Deep read + run verification:
  [`repos/asaparov-prontoqa.md`](repos/asaparov-prontoqa.md). `--seed` threads to both
  `random` and `np.random` once; run-verified byte-identical at seed 12345, varying at
  999; 3/3 hand-traced labels correct. Emits `question/query/answer/chain_of_thought`
  JSON with no model call (`--model-name json`). I1 = 3.
- **Adaptation diff:** ~100 lines of external glue, no required repo edits: subprocess
  per stratum (`--min-hops/--max-hops`, `--ontology fictional` for nonce vocab,
  `--distractors`, deduction rule), parse JSON, single LLM call, exact-match vs
  `answer`.
- **Remaining risks:** avoid `--deduction-rule Composed` with postorder (verified
  crash; use `--ordering random`); the T/F label is definitional rather than an
  independent prover verdict (mitigate with our own ~30–50-line forward-chaining
  fixpoint check — cheap and worth doing); must run from repo root
  (`bad_patterns.txt` relative path); CWA-vs-Unknown semantics needs a precise
  reference prompt (2–3 gates).

### 3 · c17 — Invented-Gate Circuit Evaluation (13/16)

- **Codebase:** [BeyondBench](https://github.com/ctrl-gaurav/BeyondBench), Apache-2.0,
  ICLR 2026, active. Deep read + run verification:
  [`repos/beyondbench.md`](repos/beyondbench.md). Run-verified deterministic; 50/50
  instances validated by an independent recursive-descent evaluator. I1 = 2.
- **Adaptation diff:** ~100-line external `c17_gates.py`: vendor `_make_expr`/
  `_eval_expr` (~25 lines), fix the **verified XOR-dropping bug** (OR-substring
  replace ordering; 0/200 XOR instances generated as shipped), add invented-gate
  random-truth-table evaluation (~40 lines), thread a per-instance `random.Random`.
- **Remaining risks:** the invented-gate/truth-table core is a small rewrite, not
  config (the repo hardcodes standard ops via `eval`); module-global RNG discipline
  needed; the best desk profile in the pool (I3=2, I5=2, I6=2) makes this the
  strongest third option if grid-world familiarity worries persist.

### 4 · c24 — Composite-Key Sorting & Canonical Ordering (12/16)

- **Codebase:** [open-thought/reasoning-gym](https://github.com/open-thought/reasoning-gym),
  Apache-2.0, actively used 2025–2026. Deep read + run verification:
  [`repos/open-thought-reasoning-gym.md`](repos/open-thought-reasoning-gym.md).
  Clean per-item `Random(seed+idx)` plumbing; 21 tests pass in a fresh venv;
  run-verified deterministic with 6/6 independent ground-truth checks. I1 = 2.
- **Adaptation diff:** consume as a dependency, 0 repo lines changed; new composite-key
  sorting generator subclass (~100–150 LOC) + strict 0/1 `score_answer` override (the
  base class gives substring partial credit) + LLM glue (~30–50 LOC).
- **Remaining risks:** randomized composite tie-breaker keys are net-new task logic;
  swap the bundled real-text corpus for nonce vocab; shared reference sorter couples
  strata to re-certification (I5=1).

### 6 · c21 — Relational Micro-World with Invented Vocabulary (11/16) — round-1 winner

- **Codebase:** same run-verified prontoqa base as c18
  ([`repos/asaparov-prontoqa.md`](repos/asaparov-prontoqa.md)); I1 = 3 — the codebase
  is not the problem.
- **Why it fell:** round 1 itself calls the adapted design "the most intricate
  generator in the set" — graph/nonce sampler + relation-property profiles +
  unique-answer rejection sampler + MC/foil balance + stratified minibatches (I3=0),
  and 4+ pre-freeze gates (empirical ceiling tuning, naive floor, literal-logician
  baseline ≤50%, per-rule discoverability check, adversarial induce-conventions
  prompt) (I4=0). It carries the cleanest skeptic verdict (I6=2) but the heaviest
  build.
- **Position:** still the best *theoretical* task. The adoption path below keeps it
  alive at marginal cost: the c18 harness is ~80% of the c21 harness.

## What changed vs round 1 and why

**Biggest movers up:**

- **c19 (27-tier → #1).** Load-bearing fact: Minigrid is the pool's only run-verified
  I1=3 codebase with **verification-only** skeptic debt — properly threaded seeding
  confirmed byte-identical, zero repo edits needed, and round 1 flagged nothing beyond
  "grid-nav familiarity is the mild risk."
- **c18 (27-tier → #2).** Load-bearing fact: prontoqa's `--seed → deterministic
  gold-labeled JSON` path was run-verified end-to-end; the entire adaptation is ~100
  lines of subprocess + scoring glue.
- **c22 (26-tier → #9) and c23 (26-tier → #11).** Load-bearing facts: IFBench's 59-test
  oracle library passes verbatim (c22); InductionBench's algorithm is sound and its
  desk profile clean even though its code needs vendoring + crash fixes (c23).

**Biggest movers down:**

- **c21 (#1 → #6).** Load-bearing fact: round 1's own adaptation table demands the
  most intricate generator in the set plus 4+ pre-freeze calibration gates — I3=0 and
  I4=0 are unique among eligible top-tier candidates.
- **c12 (28-tier finalist → #20) and c14 (28-tier finalist → #19).** Load-bearing
  fact for both: **discovery found no plausibly reusable codebase** (I1=0) — every
  candidate repo was a frozen dataset, LLM-driven, or the wrong task shape — on top of
  multi-component adapted designs.
- **c05 and c15 (28-tier finalists → gated).** Load-bearing fact: round 1 refuted both
  at major severity with only structural-redesign rescues; the gate removes them from
  contention regardless of their (decent) implementation scores.
- **c03 (round-1 backup → #10).** Load-bearing fact: BIG-bench multistep_arithmetic's
  oracle is `eval()` of the generator's own string — tautological and invalid for
  invented operators — so the load-bearing oracle is new code, and c03's foil-bank +
  per-rung effect-size machinery keeps its total at 10.

**Amendment fold-ins (2026-07-21):**

- **c10 (TB-excluded → #5).** Load-bearing fact: the same run-verified reasoning-gym
  base as c24 (repo-level verification carries over) with a clean fold-in profile
  (I5=2, I6=2); the interlocking-cycle semantics are authored on top of the reused
  infra rather than shipped in-repo.
- **c04 (TB-excluded → #7).** Load-bearing fact: the landscape's "strongest
  anti-one-shot design" in the pool now carries I1=3 — reasoning-gym's `ab.py` is a
  genuine term-rewriting-to-normal-form generator on the framework run-verified in
  round 2 under c24.
- **c02 (TB-excluded → #12).** Lands mid-table: the design survives, but its headline
  repo NUMCoT is unusable (unseeded dead-code generator, buggy oracle), so I1=1 and
  it was never run-verified.

## Adoption path (this week)

1. **Day 1–2: c19 harness.** Vendor nothing; add Minigrid as a dependency. Write the
   ~120–200-line glue package: seeded instance generator over
   Fetch/Crossing/FourRooms/Empty-Random, text render via `pprint_grid`, the
   invented-command/convention layer (wrap vs clamp, indexing, 2–3 invented commands)
   as a tiny stepper over `grid.encode` state, an independent derived-fact oracle,
   strict 0/1 scorer.
2. **Day 2–3: freeze gates for c19.** Reference ("state-all-rules") ceiling prompt at
   temp 0; per-stratum floors; red-team check that no single fixed convention
   statement covers the pool. Pin seeds + manifest.
3. **Day 3–4: c18 smoke harness in parallel** (~100 lines): prontoqa subprocess per
   stratum, JSON parse, exact-match scorer. This validates the optimizer plumbing on a
   second family and is ~80% of a future c21 harness — the cheap option on round 1's
   theoretical winner.
4. **Decision point end of week:** if c19's convention layer or red-team gate
   struggles, promote c18; if grid familiarity floors look high AND deduction load
   looks heavy, c17 (BeyondBench, ~100-line adapter with the XOR fix) is the
   ready third option.
5. **Do not invest** in c12/c13/c14/c16 (no codebase exists — every hour is
   from-scratch), or in the gated c01/c05/c07/c15.

**Amendment note (2026-07-21):** the adoption call above is unchanged (c19 adopt, c18
backup), but c10 (#5) and c04 (#7) are newly competitive options sharing the
already-run-verified reasoning-gym base with c24 — one harness (reasoning-gym as a
dependency, a generator subclass, a strict 0/1 `score_answer` override) could serve
c24, c10, and c04.

## Change log

- **2026-07-21 — initial publication.** Round-2 rescore over the 21 candidates
  surviving the round-1 tie-breaker filter (17 ranked + 4 gated).
- **2026-07-21 — amendment (fold-in).** Tie-breaker=1 exclusion dropped by user
  decision; c10, c04, and c02 folded into the ranking on equal terms at ranks 5, 7,
  and 12 (20 ranked + 4 gated = 24). No previously published relative order changed;
  adoption call (c19 adopt, c18 backup) unchanged.
