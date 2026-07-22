# c19 Baseline Experiment Spec — measuring headroom on the EXISTING task shape

> **STATUS: DRAFT.** Every number below is a proposal for the project owner to accept, adjust, or reject.
> Nothing here is frozen. Sections 4 (model set), 7 (open decisions), and all numeric thresholds are
> explicitly owner-gated. Floor/ceiling expectations are grounded in `synthesis-capability.md`; where the
> evidence is thin or absent, this document says so inline.

## 0. Purpose and scope

**Question this experiment answers:** How much headroom exists between a naive prompt and a best-effort
ceiling prompt on the *vanilla* Minigrid task shape — BEFORE we spend any build effort on the
invented-command / wrap-vs-clamp complexity layer?

This is a **go/no-go instrument**, not the benchmark itself. It exists because the single biggest early-kill
risk for c19 is a naive floor that already sits near the ceiling (`synthesis-positioning.md` §3(e); P2's
Predict split shows 85-97.5% on Easy standard-semantics, `synthesis-capability.md` Q1 Table). If that risk
is realized on the vanilla task, we learn it here, cheaply, before building the net-new simulation logic that
is the real cost of c19 (candidate file, Implementation §1; I3=1).

**Hard constraints fixed by prior decisions (do not revisit in this experiment):**

- **Vanilla Minigrid dynamics ONLY.** Standard actions and standard movement semantics. No invented commands
  (no jump-two, no wrap command), no wrap-vs-clamp convention variants. Those are the *added complexity* whose
  necessity this experiment is meant to test — they cannot be present in the baseline.
- **Fresh seeds, never any published instance.** All instances are generated from seeds we pick; none may
  coincide with a seed published by any of the five related-work papers. Contamination resistance is rubric
  criterion 8.
- **Stochastic-layout envs only.** `Fetch`, `SimpleCrossing`, `FourRooms`, `Empty-Random`. Fixed-layout
  `Empty` does not vary by seed (repo notes §2: `Empty-8x8` identical for seeds 7 and 8) and is therefore
  excluded — it cannot supply distinct instances.
- **Single derived-fact question**, scored **exact 0/1 match** against the independent oracle (the Minigrid
  object-model walk, repo notes "Oracle independence — GOOD").
- **Temperature 0.** Repeats = 3 (`repeat_id` in {0,1,2}) to exercise the aggregation plumbing; with temp 0
  they should largely agree (rubric criterion 5).

---

## 1. Strata design

### 1.1 Axes

Three crossed axes define a stratum:

- **Env id** (4 levels): `Fetch`, `SimpleCrossing`, `FourRooms`, `Empty-Random`. All four are run-verified as
  seed-threaded and byte-reproducible (repo notes §1-2; candidate file "Run verification: PASS").
- **Grid size** (2 levels): **small ~5x5** and **medium ~8x8** (env-native size variants where available, e.g.
  `Fetch-5x5` vs a larger Fetch; `SimpleCrossingS9` for the medium tier). Rationale: `2604.10690` shows
  monotone decline 5x5→9x9 and 7x7 already halves visual-grid accuracy; a size axis lets us see whether
  headroom is size-driven (`synthesis-capability.md` Q1, Q4).
- **Fact type** (4 levels): **final coordinate**, **heading**, **carrying-flag**, **what-is-in-front**. These
  are read directly from the serialized grid + agent state by independent oracle glue (repo notes
  "Fact/oracle function", using `DIR_TO_VEC` for what-is-in-front). Fact type is a genuine latent-rule axis:
  `synthesis-capability.md` Q5 notes coordinate-after-a-sequence is hardest (most spatial), heading/carry are
  easier (fewer classes, less spatial) — so it spreads difficulty and guards against a single fact being
  near-guessable (positioning §3(c): reviewers check heading≈4 classes, carry=2 aren't near-guessable).

Not every (env × fact) pair is meaningful — carrying-flag and what-is-in-front are only informative on envs
with fetchable/placed objects. **Proposed applicability matrix:**

| Fact type          | Fetch | SimpleCrossing | FourRooms | Empty-Random |
|--------------------|:-----:|:--------------:|:---------:|:------------:|
| final coordinate   |  yes  |      yes       |    yes    |     yes      |
| heading            |  yes  |      yes       |    yes    |     yes      |
| carrying-flag      |  yes  |       no       |    no     |      no      |
| what-is-in-front   |  yes  |      yes       |    yes    |     yes      |

(Carrying-flag requires pickup-able objects present + a command sequence that may pick one up; only Fetch
reliably supplies that in vanilla dynamics. **OPEN:** owner may extend carrying to other envs if the glue
places objects — flagged in §7.)

### 1.2 N per stratum and total N

**Design driver: rubric criterion 5 resolvability.** Optimizers rank proposals on tiny internal evals (3-20
tasks, 1 repeat). Criterion 5 requires that a meaningful prompt change (**≥10 points**) exceed residual noise
on a **10-20 task internal eval**. That fixes the *floor* on per-stratum N: each stratum must contain enough
instances that a 10-point effect is resolvable, and the aggregate must be large enough that strata can be
averaged (repeat-within-task, then across tasks and strata; rubric "Aggregation must cross strata").

A 10-point effect on a 10-task eval = 1 task flipping is 10 points, so the *quantization floor* is met at
N=10 per internal slice. To keep effect sizes above quantization noise and give the aggregate room, the
proposal sets per-stratum N above that floor.

**Proposed N per stratum = 15.** With the applicability matrix:

- Env × size × {coordinate, heading, what-is-in-front} = 4 × 2 × 3 = 24 strata.
- Env × size × {carrying-flag}, Fetch only = 1 × 2 × 1 = 2 strata.
- **26 strata × 15 = 390 instances total** in the pinned pool.

Justification: 15/stratum keeps each stratum at or above the 10-20 task internal-eval band criterion 5 names,
so a single stratum can serve as an internal-eval slice and still resolve a 10-point effect (1.5 tasks). The
390-instance pool is comfortably in the "hundreds of tasks" band criterion 7 wants for disjoint
minibatch/subset/official splits. Seeds are the instance index within a stratum (repo notes §3: `seed =
instance index`), drawn from a fresh reserved range that no paper published.

**Proposed seed split (for later phases, pinned now so the baseline pool is a subset):** reserve seed ranges
so train/minibatch/official splits are disjoint; the baseline reads the official split only. **OPEN:** exact
range boundaries — §7.

> **Evidence note (thin):** No related-work paper measured per-stratum variance on *exactly* this
> (single-fact, ASCII, vanilla, temp-0) cell (`synthesis-capability.md` Q5, "Uncertainty: HIGH"). N=15 is
> chosen from the rubric's resolvability arithmetic, not from a measured variance estimate. If a pilot shows
> temp-0 disagreement across repeats is non-trivial (it should be near-zero), N must rise — flagged in §7.

---

## 2. The two probe prompts (DRAFTED VERBATIM)

These operationalize rubric criterion 4 (default prompts score low; better prompts close the gap) and
criterion 9 (floor, ceiling, and a reference prompt known in advance). Both are shown here for the
**final-coordinate** fact; the harness substitutes the fact-specific question line for the other three fact
types (question variants listed after each prompt).

The `{GRID}` placeholder is filled with `pprint_grid()` output (repo notes §5). The `{COMMAND}` placeholder is
the vanilla command string (only standard Minigrid actions: turn-left, turn-right, forward, pickup, drop,
toggle — rendered as single letters `L R F P D T`).

### 2.1 Probe (a) — deliberately naive prompt

> ```
> Here is a grid and a sequence of moves for the robot.
>
> {GRID}
>
> Moves: {COMMAND}
>
> Where does the robot end up? Answer with just the final coordinate.
> ```

Fact-line variants:
- heading: `Which direction is the robot facing at the end? Answer with one letter.`
- carrying-flag: `Is the robot carrying an object at the end? Answer yes or no.`
- what-is-in-front: `What is directly in front of the robot at the end? Answer with one word.`

Design intent: gives the grid, the command, and the question — and *nothing else*. No coordinate convention,
no glyph legend, no statement of what each move letter does, no origin. It leans on the model's memorized
"standard grid semantics." Per `synthesis-capability.md` Q5, this is where the too-high-floor risk lives: a
capable model may infer the standard conventions unaided. That is exactly what we want to measure.

### 2.2 Probe (b) — best-effort ceiling prompt (all standard Minigrid conventions stated)

> ```
> You are simulating a robot on a 2D grid. Follow these rules EXACTLY.
>
> COORDINATES: cells are written (row, col). Row 0 is the TOP row; col 0 is the LEFTMOST
> column. Rows increase downward, columns increase rightward.
>
> GLYPHS in the grid below (two characters per cell):
>   - a period "." is an empty floor cell the robot may enter.
>   - "#" or "WG"/"WB" etc. (a wall glyph) is a wall; the robot CANNOT enter or pass through it.
>   - a two-letter object glyph (e.g. "KY" = yellow key, "BR" = red ball, "GG" = green goal)
>     is an object occupying that cell.
>   - the robot is shown by a direction arrow: ">" faces right (east), "<" faces left (west),
>     "^" faces up (north / toward row 0), "V" faces down (south).
>
> HEADINGS: the robot has a facing direction, one of E (east/right), W (west/left),
> N (north/up), S (south/down).
>
> MOVES (apply in order, left to right):
>   - L = turn left 90 degrees in place (does not move): E->N->W->S->E.
>   - R = turn right 90 degrees in place: E->S->W->N->E.
>   - F = step ONE cell forward in the current facing direction. If the cell directly ahead
>     is a wall or is off the edge of the grid, the robot does NOT move (it stays put); it does
>     not wrap around and does not pass through walls.
>   - P = pick up the object in the cell directly ahead, if any, and only if not already
>     carrying something; the robot then carries it.
>   - D = drop the carried object into the cell directly ahead, if that cell is empty.
>   - T = toggle (open/close) the object directly ahead; does not change position or heading.
>
> Work step by step: track the robot's (row, col) position and its heading after EACH move,
> then report only the final answer.
>
> {GRID}
>
> Moves: {COMMAND}
>
> QUESTION: What is the robot's final coordinate? Answer on the last line as: row,col
> ```

Fact-line variants (replace the QUESTION line):
- heading: `QUESTION: What is the robot's final heading? Answer on the last line as one of: E W N S`
- carrying-flag: `QUESTION: Is the robot carrying an object at the end? Answer on the last line as: yes or no`
- what-is-in-front: `QUESTION: What object or terrain is in the cell directly in front of the robot at the end? Answer on the last line with one word (e.g. wall, empty, key, ball, goal).`

Design intent: states every standard Minigrid convention the naive prompt withholds — origin/indexing, glyph
legend, exact move semantics including the no-wrap/no-pass-through edge rule, and an explicit step-by-step
instruction. This is the "state-all-rules" ceiling prompt the candidate file's freeze-gate step names
(Implementation §2). It includes a lightweight CoT nudge ("work step by step") because `synthesis-capability.md`
Q4 shows CoT is the single load-bearing lever for the cheap tier (0%→74-86%); a ceiling prompt that omits it
would understate the true ceiling and inflate apparent headroom.

> **Evidence note:** `synthesis-capability.md` Q4 warns the ASCII rendered-grid representation is the *weak*
> format (visual-grid 34% vs adjacency 86% at 5x5 for Gemini-CoT). The ceiling prompt here stays on ASCII by
> constraint (task shape is fixed). So this ceiling is the **best-prompt-on-ASCII** ceiling, which the
> synthesis estimates at ~55-75%, NOT the ~80-90% best-prompt-if-restructured ceiling. The decision rule in
> §3 is calibrated to the ASCII ceiling.

---

## 3. Three-outcome decision rule

Let **naive** = mean exact-match over the pool under probe (a), **ceiling** = mean under probe (b), each
averaged repeat-within-task then across strata, per model, then aggregated across the model set (owner
picks aggregate: mean or min — §7).

Thresholds are proposals grounded in `synthesis-capability.md` Q5 (central estimates: naive ~30-45%, ceiling
~65-80% on ASCII; too-high-floor is the dominant risk) and the positioning early-warning gate
(`synthesis-positioning.md` §3, "if naive floor is high (>~80%) OR ceiling barely beats naive, STOP").

**Proposed numeric anchors (all owner-adjustable):**
- `HIGH` band: score ≥ **75%**.
- `MID/LOW` band: score < **75%**.
- "≈" (approximately equal) means |naive − ceiling| < **10 points** (the criterion-5 resolvability unit — a
  gap below this is not reliably distinguishable from noise on the internal-eval slices).
- "≪" (much less than) means ceiling − naive ≥ **20 points** (two resolvability units; a gap this size is
  robustly prompt-closable and above per-model variance).

### Outcome (a): naive ≈ ceiling ≈ HIGH → NO headroom → proceed to invented-rule build

Condition: |naive − ceiling| < 10 **and** both ≥ 75%.
Reading: the vanilla task is already near-solved regardless of prompt; there is no gap for an optimizer to
close. This is the P2-Easy scenario (85-97.5% standard semantics) the positioning doc flags as the biggest
early-kill risk.
Decision: **the vanilla task is inadequate as-is. Proceed to build the invented-command / wrap-vs-clamp
complexity layer** — that net-new difficulty is now *justified* by measurement, not assumed.

### Outcome (b): naive ≈ ceiling ≈ MID/LOW → headroom exists but is NOT prompt-closable → build AND keep execution depth shallow

Condition: |naive − ceiling| < 10 **and** both < 75%.
Reading: the task is hard, but stating all the standard conventions does *not* help — the deficit is raw
execution/state-tracking, not missing convention knowledge. `synthesis-capability.md` Q2 supports that this is
possible: models can have intact local reasoning yet fail to integrate a sequence within one forward pass.
The gap an optimizer would need to close is a *capability* gap, which prompt optimization cannot reliably
close.
Decision: **build the invented-rule layer** (there is difficulty to exploit) **AND keep execution depth
shallow** — short command sequences, small grids — so the difficulty that discriminates optimizers comes from
the *convention rules* (which prompts CAN surface) rather than from deep multi-step execution (which they
cannot). This keeps the eventual optimizer gain attributable to convention-learning, satisfying the
positioning publishability requirement (§3(a) precision requirement iv).

### Outcome (c): naive ≪ ceiling → prompt-closable headroom → direct prompt optimization is viable on the vanilla task

Condition: ceiling − naive ≥ 20 points.
Reading: naive under-performs, and simply *stating the standard conventions* recovers a large, real gap. This
is prompt-closable headroom — exactly the substrate an optimizer benchmark needs. It is also the
`synthesis-capability.md` Q4 signature (CoT/convention-statement swings the cheap tier by tens of points).
Decision: **direct prompt optimization is viable on the vanilla task.** The invented-rule build may be
*deferred or skipped* for the quick-test purpose — the vanilla task already exhibits a gradual,
prompt-closable path (rubric criterion 4). (Caveat: positioning §2/§4 note the *publishability* case still
needs the convention layer to differentiate from P2/P5; "viable for the quick-test instrument" ≠ "sufficient
for the paper." Flagged in §7.)

> **Boundary cases the owner must rule on (§7):** mixed bands (naive MID, ceiling HIGH, gap 10-20 pts — falls
> between (b) and (c)); large per-model divergence (one model shows (c), another (a)) given
> `synthesis-capability.md`'s documented GPT-5-mini non-conformance. Proposed default: use the **min across
> models** for the HIGH test (conservative against a single lucky model) and the **max gap across models** for
> the ≪ test (any model showing prompt-closable headroom counts).

---

## 4. Model selection — OPEN DECISION (owner decides; not finalized here)

### 4.1 Models tested by the five papers, with results (from the dossiers)

| Model | Tier | Paper(s) | Result on the c19-relevant cell | Source |
|-------|------|----------|----------------------------------|--------|
| Claude 3.7 Sonnet | frontier | P2 | Predict 97.5% Easy → 82.5% Very Hard; only model holding command-sequences (80%) | cap Q1/Q3 |
| GPT-4o | mid | P2, P1 | Predict 97.5% Easy → 61.2% Very Hard; ~random-walk on GRASP planning (p=0.63) | cap Q1/Q3 |
| Qwen3-32B | reasoning-mid | P2 | Predict 95% Easy → 48.8% Very Hard | cap Q1 |
| DeepSeek-R1-Distill-70B | reasoning-mid | P2 | Predict 97.5% Easy → 37.2% Very Hard; 89% room → 44% maze | cap Q1 |
| Llama-3.1-405B | frontier-open | P2 | Predict 85% Easy → 41.3% Very Hard (non-monotonic) | cap Q1 |
| Llama-3.1-70B | mid-open | P2 | Predict 63.8% Easy (outlier low) → 34% Very Hard | cap Q1 |
| Llama-3.1-8B | small-open | P2 | Predict 12.9% Easy → ~2.8% (near-floor); command-seq 0% | cap Q1 |
| O3 | frontier-reasoning | P4 | Perfect maze nav to 30x30 (clean ceiling) | cap Q3 |
| GPT-4 Turbo | mid | P3 | Vanilla FR 76 @10x10; AoP-DFS collapses to 1-3/100 | cap Q3 |
| ChatGPT-4 | mid | P3 | Vanilla FR up to 76 @10x10 → 11-13 @30x30 | cap Q1 |
| Qwen2.5-7B | small-open | P3 | Vanilla FR 48@10x10 → 13@30x30; Algo-Reasoning 0/0/0 | cap Q3 |
| **Gemini-2.5-Flash** | **cheap** | P4, P5 | base 5x5 adj 8% → CoT 86%; visual-grid CoT 34%; weak in MazeEval >7x7 | cap Q3/Q4 |
| **GPT-5-mini** | **cheap** | P5 | no adjacency advantage (30 adj/32 visual); compositional 100%→4% (5x5→9x9); **non-conforming outlier** | cap Q3/Q4 |
| **Claude-Haiku-4.5** | **cheap** | P5 | base ~0% → CoT 78% (5x5 adjacency) | cap Q3/Q4 |
| **DeepSeek-Chat** | **cheap** | P5 | base 0-16% → CoT 74% (5x5 adjacency) | cap Q3/Q4 |
| GPT-4o-mini | cheap | P4 | "weaker performer," fails beyond 7x7 | cap Q3 |

### 4.2 Proposed candidate set for our baseline (NOT finalized)

The quick test runs at the **cheap tier** (candidate file; `synthesis-capability.md` scope line). Proposed
candidate set = the exact `2604.10690` (P5) roster, because it is the only paper whose *entire* model set is
the cheap tier and it tested the closest task shape:

- **Gemini-2.5-Flash**
- **Claude-Haiku-4.5**
- **DeepSeek-Chat**
- **GPT-5-mini** (include specifically because it is the documented non-conformer — it stress-tests the
  cross-model divergence path in §3's decision rule)

### 4.3 Tradeoffs (for the owner)

- **Pro (P5 roster):** directly comparable to the most on-point published numbers; spans providers; cheap
  enough for many repeats; includes a known outlier to exercise the divergence logic.
- **Con:** all four are cheap-tier, so if the owner also wants a *frontier* anchor (to see whether the
  too-high-floor risk is worse at frontier, per P2's 85-97.5%), add one frontier model (Claude 3.7 Sonnet or
  an O-series) — but that raises cost and the quick-test is meant to run cheap (rubric criteria 3, 14).
- **Con:** GPT-5-mini's non-conformance can dominate a min-aggregate; the owner must pair the model choice
  with the aggregate choice in §3 (min vs mean).
- **Con:** provider concurrency/quota differences (rubric criterion 14, Provider Concurrency Control) mean
  wall-clock is gated by the slowest provider; a 4-provider set has 4 independent rate limits to respect.
- **Note:** exact model *versions*/snapshots and provider routing are output-affecting (rubric: Provider Call
  Config contributes to graph identity) and must be pinned once the owner selects — flagged in §7.

**Decision required from owner:** final model set, whether to add a frontier anchor, and the aggregate
function. **This spec does not finalize it.**

---

## 5. Cost & wall-clock estimate per full baseline run

**Formula:** instances × probes × models × repeats = LLM calls; calls × tokens/call = tokens; tokens × price
= cost.

**Fixed factors:** N = 390 instances; 2 probes (naive, ceiling); repeats = 3.

**Per-instance token estimate (proposed, ~6-8x8 ASCII grid, single-token-ish answer):**

| Component | Naive probe | Ceiling probe |
|-----------|------------:|--------------:|
| Input tokens (grid ~64 cells ×2 chars + command + prompt scaffold) | ~250 | ~750 (long rule block) |
| Output tokens (naive: short answer; ceiling: CoT step-by-step + answer) | ~30 | ~400 (CoT trace) |
| **Total tokens/call** | **~280** | **~1,150** |

**Calls per model per full run:** 390 × 2 probes × 3 repeats = **2,340 calls/model**.
Split: 1,170 naive + 1,170 ceiling.

**Tokens per model per full run:** (1,170 × 280) + (1,170 × 1,150) = 327,600 + 1,345,500 ≈ **1.67M tokens/model**.

**For the proposed 4-model set:** 4 × 2,340 = **9,360 calls**; 4 × 1.67M ≈ **6.7M tokens** per full baseline
run.

**Cost (order-of-magnitude, cheap-tier blended ~$0.50–$1.50 per 1M tokens depending on model and
input/output split):** 6.7M tokens ⇒ roughly **$3–$10 per full baseline run**. This is squarely in the
"cents-to-single-dollars, cheap" band the rubric goal requires (criteria 3, 14). **OPEN:** plug exact
per-model per-token prices once the model set is fixed — §7. (Pricing is model-specific; do not quote a hard
number until the owner selects models and versions.)

**Wall-clock (proposed):** at modest concurrency (say 20 concurrent calls/provider, ~2-4s/call for cheap-tier
CoT outputs), 9,360 calls / (20 × 4 providers) ≈ 117 waves × ~3s ≈ **6-10 minutes** wall-clock, dominated by
the slowest provider and its rate limit (rubric criterion 14 Provider Concurrency Control). Comfortably
"minutes, not hours."

> **Evidence note (thin):** the ceiling-probe output-token estimate (~400 CoT tokens) is a guess; CoT length
> is model-dependent and `synthesis-capability.md` does not report token counts. A 10-instance pilot per model
> should replace these estimates before committing the full run — flagged in §7.

---

## 6. Rubric-mapping table

| Design choice (this spec) | Rubric criteria served |
|---------------------------|------------------------|
| One LLM call + one exact-match oracle check per instance | 1 (single node graph), 2 (exact check by simple fn) |
| Single derived fact, single-token-ish answer, short ASCII grid input | 3 (bounded token cost) |
| Two probes (naive vs ceiling) measuring the gap | **4** (default low, better closes gap), **9** (floor/ceiling/reference known) |
| Naive probe withholds all conventions; ceiling states them + CoT | **4**, **9** |
| Temp 0, repeats = 3, N=15/stratum sized to resolve ≥10-pt effect on 10-20 task eval | **5** (determinism + resolvability) |
| 390-instance pinned pool, seeded, disjoint splits reserved | 7 (large generated pool, splits), 8 |
| Fresh reserved seeds, no published instance | **8** (contamination resistance) |
| Independent oracle = Minigrid object-model walk, not sampler re-derivation | 2, 8, 11 |
| Strata = env × size × fact type (latent-difficulty axes) | 5 (aggregation crosses strata), 10 (independent rules), 7 |
| Exact-match "expected X got Y" diagnostics per fact | 11 (diagnostic failures) |
| Decision rule keys off measured naive/ceiling gap vs known thresholds | **9** (falsifiable success criterion), 4 |
| Cheap-tier model set, short outputs, minutes wall-clock | 3, 14 |
| Repeats validate aggregation plumbing though scores agree | 5, and noted non-coverage (repeat-averaging under variance is NOT stressed) |

**Criteria the baseline deliberately does NOT cover** (per rubric "What the quick test does not cover"):
few-shot (12), reflection-trace diagnosticity beyond exact-match (11 partial), the second output-bytes
objective (GEPA two-objective), and multi-node routing — all out of scope for a *headroom baseline*.

---

## 7. Open decisions (awaiting the owner)

1. **Model set** (§4) — final roster; whether to add a frontier anchor; exact model versions/snapshots and
   provider routing (all output-affecting, must be pinned).
2. **Aggregate function** for the decision rule (§3) — mean vs min across models; pairing with the GPT-5-mini
   non-conformer.
3. **Decision thresholds** (§3) — the 75% HIGH band, the 10-point "≈" unit, the 20-point "≪" unit. All are
   proposals derived from `synthesis-capability.md` central estimates, NOT from measured c19 data.
4. **Boundary-case handling** (§3) — mixed bands (gap 10-20 pts) and per-model divergence resolution.
5. **N per stratum / stratum weights** (§1) — proposed uniform N=15; owner may reweight (e.g. more N on the
   coordinate-after-sequence stratum that the synthesis flags as most discriminating, less on near-guessable
   carry/heading). Whether to expand carrying-flag beyond Fetch.
6. **Grid sizes** (§1) — the small/medium levels (~5x5 / ~8x8) and env-native size variants to use.
7. **Seed ranges / split boundaries** (§1) — exact reserved ranges guaranteeing disjoint train/minibatch/
   official splits and zero overlap with any published instance.
8. **Pilot before full run** (§5) — run 10 instances/model to replace the token/CoT-length estimates and
   confirm temp-0 repeat agreement before committing the ~$3-10 full run; if repeats disagree, raise N.
9. **Answer normalization policy** — how the exact-match scorer canonicalizes model output (whitespace,
   `row,col` vs `(row,col)`, `E` vs `east`, `yes` vs `y`) before the 0/1 compare. Loose normalization inflates
   scores; strict normalization may penalize format-only misses. Owner sets the canonicalization rule.
10. **Quick-test vs paper scope** (§3 outcome c) — even if outcome (c) says the vanilla task is a viable
    *optimizer-benchmark instrument*, `synthesis-positioning.md` says the *paper* still needs the
    invented-command layer to differentiate from P2/P5. Owner decides whether "viable for the quick test" is
    sufficient or the build proceeds regardless for the publication case.
```
