# Capability Synthesis: LLM Grid/Maze State-Tracking & Navigation (2024-2026)

Scope: 5 verified dossiers on grid-world / maze reasoning, synthesized to inform the
**c19** baseline design (VANILLA Minigrid dynamics, full-observation ASCII grid, single
derived fact, temp 0, cheap-tier models). Throughout, **SHOWS** = directly reported in a
paper; **INFER** = my extrapolation. Papers are cited by dossier key + section refs.

Dossier keys:
- `2407.01892-grasp` — GRASP (11x11 energy-collection planning, single-call full plan).
- `2505.12135-llm-babybench` — LLM-BabyBench (BabyAI textual; **Predict split = the c19 template**).
- `2505.24306-gridroute` — GridRoute (10/20/30 grids, coordinate-list route planning, single-shot).
- `2507.20395-mazeeval` — MazeEval (5x5-15x15 mazes, partial-obs INTERACTIVE function-calling loop).
- `2604.10690` — "Do LLMs Build Spatial World Models?" (5x5-9x9 mazes, single-shot, representation swing) — **central for Q4**.

A structural caveat that colors everything below: only **two** of the five papers actually
sit in c19's quadrant (full-observation × single-shot × single/derived-fact-ish output):
`2505.12135-llm-babybench` (Predict split) and `2604.10690`. GRASP and GridRoute are
single-shot but ask for a **full multi-step plan/path** (many-valid-answer scoring), and
MazeEval is **interactive + partial-observation**. So GRASP/GridRoute/MazeEval numbers are
loose upper-bounds on *difficulty* (they conflate planning/optimization/memory with state
tracking), not direct predictors of c19's single-derived-fact accuracy.

---

## Q1. Single forward prediction — what LLMs CAN and CANNOT do

### What the evidence SHOWS

**One-shot state prediction is tractable at small sizes with good models, and collapses with size.**
The cleanest evidence is `2505.12135-llm-babybench` Predict split (single-shot, full-obs,
predict final agent position+direction after a given action sequence; Sec 4.4.1). Predict
Success Rate by difficulty (Table 3, ToT prompting):

| Model | Easy | Moderate | Hard | Very Hard |
|---|---|---|---|---|
| Claude 3.7 Sonnet | 97.5% | 94.3% | 86.7% | 82.5% |
| GPT-4o | 97.5% | 84.7% | 76.8% | 61.2% |
| Qwen3-32B | 95.0% | 65.5% | 61.9% | 48.8% |
| DeepSeek-R1-Distill-70B | 97.5% | 68.6% | 50.8% | 37.2% |
| Llama-3.1-405B | 85.0% | 53.3% | 62.2% | 41.3% |
| Llama-3.1-70B | 63.8% | 39.4% | 32.5% | 34.0% |
| Llama-3.1-8B | 12.9% | 4.5% | 2.1% | 2.8% |

- CAN: top models predict final state at Easy near ceiling (85-97.5% for 5 of 7 models),
  and Claude holds 82.5% even at Very Hard (`2505.12135`, Sec 5.2.1, Table 3).
- CANNOT (uniformly): "Easy = ceiling" is **not** universal — Llama-3.1-70B is a
  non-monotonic outlier at 63.8% Easy (below the larger 405B), and 8B is near-floor
  (12.9%) (`2505.12135`, Table 3, verified correction). Degradation is monotone with
  difficulty for GPT-4o / DeepSeek / Qwen3 (Sec 5.2.1).
- Competency breakdown (`2505.12135`, Table 4): Room Navigation is easy (93-98% for strong
  models); **Maze/multi-room navigation and Sequences-of-Commands are disproportionately
  hard** — DeepSeek-R1 drops from 89% room to 44% maze; every non-Claude model degrades
  sharply on command-sequences (Llama-8B = 0%). Only Claude holds command-sequences (80%).

**Representation dominates one-shot accuracy** (see Q4). `2604.10690` (single-shot maze
path, 5x5-9x9): adjacency-list format hits 86% at 5x5 while the *same maze* as a visual
ASCII grid gets 34% (Gemini-2.5-Flash CoT; Sec 4.1, Table 4). Base (no-CoT) prompting is
near-0% for several models even at 5x5.

**GridRoute** (single-shot, coordinate-list, output = full path): Vanilla Feasibility Ratio
up to 76 at 10x10 (ChatGPT-4) but as low as 11-13 at 30x30 (`2505.24306`, Tables 9-11,
Sec 5 RQ4). Map size (10→20→30) is the dominant driver of decline, independent of model size.

**Grid size / step-count collapse thresholds (SHOWS, but task-dependent):**
- `2604.10690`: monotone degradation 5x5→9x9 across all tasks/models; 7x7 already halves
  Gemini's visual-grid accuracy (Sec 4, Table 4).
- `2505.12135` Plan task (different task): reasoning models ~80% on 8x8, none >50% on
  16x16/24x24; on 32x32 only reasoning models score >0, max ~20% (Sec 5.2.2). Failure is
  driven by **grid size, not obstacle count** (Appendix E.2).
- GRASP: even at fixed 11x11 with a 20-step budget and full upfront knowledge, no LLM beats
  a greedy one-step-lookahead baseline (`2407.01892`, Table 1) — but this is plan-quality,
  not state-tracking.

### What I INFER
- For **pure single-derived-fact state tracking** on a **~6x6** grid with a **short**
  command sequence, the evidence points to this being the *easiest* cell in this literature:
  it is smaller than every collapse threshold above, asks for one fact (not a full path,
  removing many-valid-answer penalties that depress GridRoute/GRASP/Task-1 numbers), and
  6x6 room-navigation is ~93-98% for strong models in `2505.12135` (Table 4). INFER: strong
  models should be at/above 90% here; the risk is a **too-high floor** on the naive prompt
  (flagged in c19's own notes and echoed by `2505.12135` baseline_relevant and `2505.24306`
  overlap notes: "vanilla is not floor-level").

---

## Q2. Multistep / interactive settings — what breaks; does interaction help or hurt

### What the evidence SHOWS

**Interactive, partial-observation navigation collapses via LOOPING.** `2507.20395-mazeeval`
is the only interactive/partial-obs paper. Findings (Sec 4.2-4.5):
- **100% of failures (all 7 non-O3 models, both languages) are excessive looping** —
  revisiting a cell 10+ times (Sec 4.5, l.204). **Zero** failures from exhausting the 3n²
  move budget (l.206) — models loop long before running out of moves.
- Wall-collision rate stays low (~0.8 avg even in failed runs, l.208): per-step *local*
  reasoning (reading distance-to-wall) is intact. The failure is **integrating history into
  a persistent spatial memory** — despite full travel history being re-supplied every turn.
  Mechanistically the paper says models "treat each navigation decision as relatively
  independent" (Sec 5.3).
- Backtracking escalates steeply with size (Claude Opus 4: 6.8 backtracks at 5x5 → 68.6 at
  12x12; Sec 4.5). Failures are **systematic, not stochastic** (Sec 4.2, l.161).

**Cross-query state drift within a single prompt.** `2604.10690` Tasks 2/3 (batched related
questions in one prompt):
- On an exact-repeat query (Q3=Q0), Gemini and GPT-5-mini **fully re-derive** the answer
  from scratch (high ROUGE-L) rather than reusing prior computation (Sec 4.2.1); DeepSeek /
  Claude-Haiku under-reason (ROUGE-L down to 0.08) without gaining accuracy.
- Compositional task: GPT-5-mini and DeepSeek-Chat **decline** at the composable Q2 vs the
  avg of its parts (Δ = −2.2, −2.4; Table 9). Models "treat each question independently
  rather than building cumulative spatial knowledge" (Abstract) despite 96-99% semantic
  coverage in their reasoning traces — a dissociation between "understanding" and reuse.

**Multi-step plan GENERATION is hard across the board.** GRASP: full-plan generation on
11x11 loses to greedy and (for GPT-3.5) to random walk (`2407.01892`, Table 1). LLM-BabyBench
Plan (simulator-executed): tops out <50% on medium/large grids (`2505.12135`, Sec 5.2.2).
Decompose: PR=0 for all models at 10+ subgoals (Sec 5.2.3). GridRoute: FR/OR decline
systematically with map size (`2505.24306`, RQ4).

**Failure taxonomies converge** on: looping/oscillation, out-of-bounds/boundary violation,
path-through-obstacle, and off-by-one edge errors (GRASP Fig 4 categories a-f incl.
oscillation `2407.01892`; GridRoute 5-way taxonomy incl. Out-of-Bounds "lack of spatial
awareness" `2505.24306` Sec 3.3; MazeEval looping `2507.20395`).

### Does interaction help or hurt? (SHOWS + INFER)
- **SHOWS (comparative, within `2505.24306`):** more within-call step-by-step reasoning
  (CoT/AoP; still single-shot) helps vs direct generation, especially as size grows
  (`2505.24306` RQ1; `2604.10690` Sec 4.1: CoT rescues Claude-Haiku from ~0% to 78%).
- **SHOWS (no head-to-head, but structural):** No paper runs the same task both interactively
  and one-shot. But the modal *interactive* failure (looping/lost memory) is **structurally
  impossible in a single-shot single-fact task** — there are no repeated turns to lose track
  of (`2507.20395` relation_to_candidate; `2604.10690` cross-query non-reuse also vanishes
  if you ask exactly one fact per call).
- **INFER:** For c19's single-shot single-fact design, dropping interaction *removes* the
  dominant failure modes documented here (looping, cross-query drift). This makes c19
  **easier** than the interactive/batched settings — reinforcing the too-high-floor risk,
  not a collapse risk. CONFLICT to surface: MazeEval attributes failure to *long-horizon
  memory across turns*, which its own Discussion (5.4) admits is untested for the
  full-obs/single-shot case — so we cannot cite MazeEval's collapse thresholds as evidence
  that c19 will be hard.

---

## Q3. Model-tier breakdown (cheap tier is what c19 runs at)

### What the evidence SHOWS

**Frontier / reasoning tier:**
- O3: perfect maze navigation up to 30x30 interactive (fails 40x40); near-perfect efficiency
  (`2507.20395`, Sec 4.1-4.4). A clean ceiling case.
- Claude 3.7 Sonnet: best single-shot Predictor (82.5% Very Hard, 80% on hardest levels);
  only model holding command-sequences (`2505.12135`, Sec 5.2.1).
- Reasoning models (Qwen3-32B, DeepSeek-R1-Distill) reach ~80% on Plan small grids but still
  <50% medium/large (`2505.12135`, Sec 5.2.2).
- GPT-4o / GPT-4 Turbo: mid — GPT-4o statistically indistinguishable from random walk on
  GRASP planning (`2407.01892` t=-0.48, p=0.63); GPT-4 Turbo Vanilla FR 76 at 10x10 but
  AoP-DFS collapses to 1-3/100 (`2505.24306`, Table 10) — prompt framing can catastrophically
  backfire.

**Cheap tier (directly relevant — c19 runs here). The most on-point evidence is
`2604.10690`, whose entire model set IS the cheap tier: Gemini-2.5-Flash, GPT-5-mini,
Claude-Haiku-4.5, DeepSeek-Chat** (5x5-9x9 single-shot mazes; Table 4):
- **Base (no-CoT) prompting: near-total failure.** Claude-Haiku-4.5 = 0.00 on most
  5x5/7x7/8x8/9x9 settings; DeepSeek-Chat 0.00-0.16; Gemini-2.5-Flash base 5x5 adjacency = 8%
  (Sec 4.1, Table 4).
- **CoT rescues them dramatically at 5x5:** Claude-Haiku 78%, DeepSeek-Chat 74%,
  Gemini-2.5-Flash 86% (adjacency-list; Sec 4.1).
- **GPT-5-mini is the odd one:** no adjacency-list advantage (30% adj vs 32% visual at 5x5,
  CoT); its compositional Q0 accuracy craters 100%→4% across 5x5→9x9 (Table 8); lowest
  overall compositional accuracy 51.5% (Table 9).
- MazeEval cheap-tier confirms weakness: Gemini-2.5-Flash and GPT-4o-mini are "weaker
  performers," failing beyond 7x7 (`2507.20395`, Sec 4.2). Gemini-2.5-Flash wall-collision
  spikes to 17.2 avg at 9x9.
- GridRoute small models: Qwen2.5-7B Vanilla FR 48 at 10x10 → 13 at 30x30; Algo-Reasoning
  (complex worked trace) produces 0/0/0 — "limited capabilities of Qwen2.5-7B"
  (`2505.24306`, Sec 5 RQ2, Table 11).

### INFER for the cheap tier
The cheap tier is **highly prompt-sensitive**: base prompting can floor them at ~0% even on
5x5 (`2604.10690`); the *right* representation + CoT swings them to 74-86%. This is the
single most important design fact for c19: **the naive-vs-ceiling prompt gap at the cheap
tier is enormous** (potentially ~0% → ~80%), and format choice can be as large a lever as
prompting. GPT-5-mini's non-conformance is a caution that per-model variance is high.

---

## Q4. Prompting strategies & input representations — how much they moved accuracy

### What the evidence SHOWS (magnitudes)

**Representation swing — the central `2604.10690` result. Exact conditions (Sec 4.1, Table 4,
Abstract):**
- Model **Gemini-2.5-Flash**, **CoT prompting**, single-shot maze **path-finding** (Task 1),
  50 test mazes/size, DFS+percolation p=0.2, exact-match path accuracy.
- **5x5: 86% adjacency-list vs 34% visual-grid.** **7x7: 80% adjacency vs 16% visual.**
- The "16-86%" corpus headline = the 7x7-visual floor (16%) to the 5x5-adjacency ceiling
  (86%); the paper calls it a **2-5x gap** and says it holds **only for 3 of 4 models**.
- **GPT-5-mini is the documented exception**: NO adjacency advantage (30% adj vs 32% visual,
  5x5 CoT). So the representation effect is real but not universal.
- CRITICAL for c19: **the winning format (adjacency-list with special tokens) is the
  STRUCTURED/SYMBOLIC one; the ASCII "visual grid" (`Row 0: [".", "S", ".", "#"]` + legend)
  is the LOSING format.** c19's planned representation (full-observation ASCII grid) is on
  the *weak* side of this swing.
- Both formats are text-based; no image/pixel input tested (Sec 5). Base (no-CoT): near-0%
  for Claude-Haiku/DeepSeek regardless of format.

**Prompting-strategy magnitudes:**
- `2604.10690`: **CoT is load-bearing** — Claude-Haiku ~0% → 78%, DeepSeek ~0% → 74% at 5x5
  (Sec 4.1). Biggest single lever for cheap models.
- `2505.12135` Predict (Table 7, avg over 7 models): **ToT 62.62% > CoT 61.07% > Zero-Shot
  59.29% ≈ Few-Shot 59.05%.** ToT best but only ~3 pts over zero-shot on this task; few-shot
  gave essentially no lift. (Note: this averages strong+weak models, muting the cheap-tier
  swing seen in `2604.10690`.)
- `2505.12135` **format** effect (Predict, Table 6, avg all models, ToT fixed): **Structured
  bullets 61.08% > JSON 59.28% > Narrative 56.78%** — a ~4-5 pt swing from formatting alone.
  Structured (key:value bullets) beat prose and JSON. (All three are coordinate-list text; no
  rendered-ASCII arm.)
- `2505.24306` GridRoute: embedding algorithm mechanics in-prompt (AoP) beats Vanilla,
  especially at larger sizes (RQ1); FewShot-Base ≈ AoP-Dijkstra. BUT worked-trace few-shot is
  a **trade-off, not a win** (verified): Algo-Reasoning improves GM/MSE but *hurts* FR/OR and
  produces 0/0/0 on Qwen2.5-7B (Sec 5 RQ2). AoP-DFS collapses GPT-4 Turbo to ~1-3/100.
- `2407.01892` GRASP: zero-shot only; authors flag this as a limitation possibly
  *underestimating* capability — no CoT/few-shot tested.

### CONFLICTS / caveats to surface
1. **Few-shot direction conflicts:** `2505.12135` few-shot ≈ zero-shot (no help);
   `2505.24306` few-shot can *hurt* (worked-trace trade-off, breaks small models);
   `2604.10690` k-shot "can hurt, not help, especially on larger mazes" (Sec 5.1). Consensus:
   **few-shot is NOT a reliable lever; CoT is.**
2. **Structured-vs-visual is consistent across the two most c19-relevant papers:** both
   `2604.10690` (adjacency ≫ ASCII visual) and `2505.12135` (structured bullets > prose)
   favor structured/symbolic over rendered/prose. This is the strongest cross-paper
   convergence and it warns that **c19's ASCII-grid representation may itself depress
   accuracy** independent of task difficulty.
3. **No paper directly ablates rendered-ASCII-grid vs coordinate/adjacency list at fixed
   CoT** in a way that isolates c19's exact format — `2604.10690`'s visual-grid is the
   closest proxy and it loses badly.

---

## Q5. Implications for c19's baseline — predicted naive-prompt floor and ceiling-prompt score

**c19 config recap:** VANILLA Minigrid dynamics (standard actions, NO invented commands in
the baseline), full-observation **ASCII grid**, single derived fact (final coord / heading /
carrying-flag), exact-match, temp 0, **cheap-tier** models (Gemini-2.5-Flash, GPT-5-mini,
Claude-Haiku-4.5, DeepSeek-Chat, and similar).

This is the intersection of: the *easiest task shape* in the literature (single fact, ~6x6,
full-obs, no planning/optimization/memory-across-turns) but the *weakest representation*
(rendered ASCII grid) at the *most prompt-sensitive tier*.

### Predicted NAIVE-PROMPT floor (base prompting, ASCII grid, no CoT), cheap tier

**Range: ~15% - 55% mean accuracy across cheap-tier models, wide per-model spread (single
models could be ~0% or ~60%+).**

Reasoning (mostly INFER, anchored to SHOWS):
- SHOWS: base/no-CoT prompting floors Claude-Haiku and DeepSeek at ~0% even at 5x5, and
  Gemini-Flash base at 8%, on the *visual-grid* format (`2604.10690`, Table 4). ASCII grid is
  that losing format. This pulls the floor DOWN.
- SHOWS (opposing): the task there is full *path generation* (harder, path-ambiguous). c19
  asks ONE fact — far easier. `2505.12135` room-navigation single-shot is 93-98% for strong
  models; even cheap models should do better on one-fact-6x6 than on full-path-mazes. This
  pulls the floor UP.
- INFER net: the single-fact simplification substantially offsets the ASCII-format penalty,
  but base prompting + ASCII is genuinely a weak combination at the cheap tier. Wide band
  because per-model variance is large (GPT-5-mini non-conformance; Gemini base = 8%). If the
  fact asked is "carrying-flag" or "heading," floor could be higher (less spatial); if it's
  "final coordinate after a multi-step sequence," lower.
- **Uncertainty: HIGH.** No paper tests exactly this (single fact + rendered ASCII + base
  prompt + cheap tier). The ~0% base-prompt data is for a harder output target; whether the
  single-fact simplification lifts a 0% floor to 30% or to 60% is genuinely unknown.

### Predicted CEILING-PROMPT score (best format + CoT/ToT, cheap tier)

**Range: ~70% - 90% mean accuracy at ~6x6, degrading toward ~50-70% if grid pushed to 8x8-9x9
or command sequences lengthened.**

Reasoning (mostly INFER, anchored to SHOWS):
- SHOWS: cheap-tier CoT at 5x5 = 74-86% on *adjacency-list* maze paths (`2604.10690`).
  Single-fact target is easier than full path → pushes ceiling UP toward/above 86%.
- SHOWS: BUT c19 uses ASCII grid, the *weak* format; if c19 keeps ASCII rather than switching
  to adjacency/structured, subtract a large representation penalty. `2604.10690` visual-grid
  CoT is only 34% (5x5) / 16% (7x7) for Gemini. So **format choice alone could move c19's
  ceiling by 30-50 points.** If "ceiling prompt" means best-prompt-on-ASCII, ceiling is lower
  (~50-70%); if it means best-prompt-and-allowed-to-restructure-toward-structured-text,
  ceiling is ~80-90%.
- SHOWS: GPT-5-mini may not benefit from structured formats and craters at larger sizes
  (100%→4% compositional, 5x5→9x9) — one cheap model may stay low regardless.
- INFER net: at strictly ~6x6 with a short command sequence and the single-fact target,
  best-prompt cheap-tier should reach ~80-90% *if a structured representation is permitted*,
  or ~55-75% *if locked to rendered ASCII*.
- **Uncertainty: MEDIUM-HIGH**, driven mainly by (a) whether c19's ASCII grid stays the
  representation and (b) command-sequence length / which derived fact.

### Go/No-Go read (INFER)
- The **dominant risk c19 must design against is a too-HIGH floor, not a collapse.** Multiple
  dossiers explicitly warn vanilla grid navigation is "not floor-level" (`2505.12135`
  baseline_relevant; `2505.24306` overlap; `2604.10690` shows CoT rescue to 74-86%). With
  VANILLA Minigrid dynamics (standard, memorizable action semantics), a capable cheap model
  given CoT + a structured view could plausibly hit ~80-90% — leaving thin headroom and a
  weak discriminating signal.
- **Levers the evidence says will genuinely add difficulty / lower a too-high ceiling** (if
  go/no-go says "needs complexity"): (a) c19's own planned invented-commands + wrap-vs-clamp
  strata (defeat memorized semantics — no paper here tested this, so it is net-new
  difficulty); (b) forcing the *rendered ASCII* representation (the documented weak format,
  −30-50 pts vs structured); (c) longer command sequences and multi-room/maze topology
  (`2505.12135` Table 4: maze ≪ room; command-sequences ≪ single moves); (d) nudging grid
  size to 8x8-9x9 (`2604.10690` monotone decline; MazeEval 9x9 tier cutoff).
- **Conversely, if the baseline as specified (VANILLA + ASCII + single fact + 6x6) already
  lands mid-range** (my central estimate: naive ~30-45%, ceiling ~65-80% on ASCII), it has
  usable headroom and may NOT need added complexity — the ASCII penalty is doing difficulty
  work that the vanilla dynamics alone would not. The decision hinges on measured naive-floor:
  if naive > ~60%, add complexity; if naive lands ~20-45% with ceiling ~70-85%, the task
  discriminates well as-is.

---

## Executive Summary (10 lines)

1. Only 2 of 5 papers (`2505.12135` Predict, `2604.10690`) truly match c19's full-obs ×
   single-shot × single/derived-fact shape; GRASP/GridRoute score full plans, MazeEval is
   interactive+partial-obs — treat their numbers as difficulty upper-bounds, not predictors.
2. One-shot state prediction is tractable and near-ceiling at small grids for strong models
   (Claude 3.7: 97.5% Easy, 82.5% Very Hard) but degrades monotonically with grid size and
   is much worse for maze/multi-room and long command-sequences (`2505.12135` Tables 3-4).
3. In interactive/partial-obs settings the universal failure is LOOPING/lost spatial memory
   (100% of MazeEval failures; collapse ~7x7-9x9 for non-O3 models) — a failure mode that is
   structurally IMPOSSIBLE in c19's single-shot single-fact design.
4. Cross-query state drift is real even within one prompt: models re-derive or fail to reuse
   repeated/compositional spatial facts (`2604.10690` Tasks 2-3) — also removed if c19 asks
   exactly one fact per call. So interaction/batching HURTS; c19's design avoids both.
5. Cheap tier (Gemini-2.5-Flash, GPT-5-mini, Claude-Haiku-4.5, DeepSeek-Chat) is the exact
   `2604.10690` roster and is EXTREMELY prompt-sensitive: base prompting floors several at
   ~0% on 5x5, CoT rescues to 74-86%; GPT-5-mini is a non-conforming outlier.
6. Central representation result (`2604.10690`, Gemini-2.5-Flash, CoT): 5x5 = 86% adjacency
   vs 34% visual-grid; 7x7 = 80% vs 16% (2-5x, holds for 3/4 models; GPT-5-mini exempt).
7. That swing warns c19's chosen RENDERED ASCII grid is the WEAK format; structured/symbolic
   text wins in both matching papers (`2505.12135`: structured bullets 61% > JSON > narrative).
8. Prompting: CoT is the reliable lever (0%→78% cheap-tier); ToT slightly beats CoT/zero-shot
   (`2505.12135` 62.6 vs 59.3); few-shot is unreliable and can HURT/break small models
   (`2505.24306` worked-trace 0/0/0 on Qwen-7B; `2604.10690` k-shot hurts on larger mazes).
9. Predicted c19 cheap-tier: NAIVE floor ~15-55% (HIGH uncertainty — base+ASCII is weak but
   single-fact is easy; no paper tests this exact cell); CEILING ~55-75% if locked to ASCII,
   ~80-90% if a structured representation is permitted (MEDIUM-HIGH uncertainty).
10. Go/No-Go: the dominant risk is a TOO-HIGH floor, not collapse; VANILLA+ASCII+single-fact
    likely lands mid-range with usable headroom, but if measured naive > ~60% add complexity
    via c19's invented-commands/wrap-clamp strata, longer sequences, or 8-9x9 sizing.
