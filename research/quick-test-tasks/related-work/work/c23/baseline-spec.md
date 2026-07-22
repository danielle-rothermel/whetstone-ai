# DRAFT — c23 Baseline Experiment Spec: Hidden-Rule String-Transform Induction Ladder

> **STATUS: DRAFT.** Every number below is a proposal for the project owner to accept, tune,
> or reject. Floor/ceiling expectations are grounded in `synthesis-capability.md`; where the
> synthesis flags an evidence gap (serialization format, exact cheap-model behavior, held-out
> vs. IB's own protocol), this spec says so inline. Nothing here is frozen until the pilot
> (Step 5 of the repo adaptation plan) calibrates reachability on real generations.

## 0. Purpose and what this baseline is NOT

**Purpose.** Measure the **headroom on the EXISTING task shape** — before building any of the
candidate's added complexity (the composition ladder, invented multi-rule stacking, the MC
variant, the multi-family strata). The single question this experiment answers:

> On fresh-seed, single-rule, InductionBench-style ISL/OSL string-transform instances in the
> existing I/O-demos-plus-query format with short-string exact-match, **how large is the gap
> between a naive prompt and a best-effort ceiling prompt**, and is that gap prompt-closable?

The answer routes the project down one of three branches (Section 3).

**Fixed by prior decisions** (not open here):
- InductionBench-style **subregular (ISL/OSL)** string-transform instances.
- Regenerated with **FRESH seeds** using the existing InductionBench codebase (vendored +
  patched per the repo note: `config.py` stub, drop the `translate_fewshot_input_output_pairs`
  import, thread a real `seed` param, and kill `list(set(...))` hash-order nondeterminism by
  replacing with `sorted(...)` at lines 53/61/107/132/208/244 — the preferred fix over pinning
  `PYTHONHASHSEED`). **Never** use published InductionBench instances (rubric criterion 8).
- **Single-rule instances only.** No composition ladder, no invented multi-rule stacking.
- Existing **I/O-demos-plus-query** format; **short-string exact-match** output.
- **Exact 0/1** scoring vs. the independent oracle (`apply_ISL_rule` / `apply_L_OSL_rule` /
  `apply_R_OSL_rule` re-applied to the held-out query, compared for string equality).
- **Temperature 0**; **repeats = 3**.

**Deferred (NOT in this baseline):** rule composition, MC distractors, casing/swap/command→action
strata, the many-shot demo-count axis, the second (output-bytes) objective.

---

## 1. Strata design

### 1.1 What varies (axes chosen from the candidate + repo notes)

The candidate and repo expose four generator knobs on the ISL/OSL path: `--type`
(ISL / L-OSL / R-OSL), `--k` (context window), `--vocab_size` (|Σ|), and `--number_of_rules`.
Under the single-rule constraint, `--number_of_rules = 1` is **fixed**. That leaves three
live axes. The synthesis is unambiguous about their difficulty ordering:

- **`k` is the single largest difficulty driver** (IB Sec 5.2): k=2→k=4 collapses most models
  from partial success to ~0 compatibility. This is our **primary stratum axis**.
- **`--type` (ISL / L-OSL / R-OSL)** is the secondary axis: it changes the transducer family
  and is exactly the kind of independent, one-generation-diff stratum the candidate praises
  (I5=2). All three share one oracle codepath, so covering them is cheap.
- **`--vocab_size` (|Σ|)** matters less than `k` (synthesis Q1). We hold it **small and fixed**
  for the baseline (|Σ| = 4) to keep the axis count down and land in the "partial-success"
  regime; it becomes a curation knob if the pilot shows the band collapsing.

### 1.2 Proposed strata (single-rule, |Σ|=4 fixed)

| Stratum | type | k | |Σ| | rules | Expected regime (from synthesis) |
|---|---|---|---|---|---|
| S1 | ISL | 2 | 4 | 1 | Easiest; the "holds" regime — GPT-4o ~40 compat, Llama-3.3-70B 60→80 |
| S2 | L-OSL | 2 | 4 | 1 | Easy; output-strictly-local left variant |
| S3 | R-OSL | 2 | 4 | 1 | Easy; right variant |
| S4 | ISL | 3 | 4 | 1 | Moderate; k=3 is the interpolation point between "holds" (k=2) and "collapse" (k=4) |

**Rationale for the k ladder stopping at 3, not 4.** The synthesis is explicit that k=4 sends
everything except o3-mini to ~0 (IB Table 1). A baseline whose hardest stratum is unreachable
violates rubric criterion 9 (a known-reachable ceiling) and criterion 4 (room to improve, not a
floor for everyone). k∈{2,3} keeps all strata inside the partial-success band. **The pilot must
confirm S4 (k=3) is reachable by the ceiling prompt; if S4 sits near floor for all models, drop
it and hold k=2 across the board.** k=4 is reserved as a curation lever only if k=2 turns out to
be one-shot-guessable (criterion 4 failure in the other direction).

### 1.3 N per stratum and total N — justified by criterion 5

**Requirement (criterion 5):** the effect size of a meaningful prompt change (≥10 points) must
exceed residual noise on a **10–20 task internal eval**, so optimizers can rank proposals on
tiny subsets. Exact-match is per-instance binary, so we size N so that a 10-point (0.10) shift
is resolvable and the per-stratum averages are stable.

- **Internal-eval unit:** 10–20 tasks (criterion 5). We set the **per-stratum baseline N so it
  contains at least one full internal-eval unit and enough to bound the binary-proportion noise.**
- **Proposed N per stratum = 50.** At p≈0.5 (worst-case variance for a binomial), the standard
  error of a single-stratum exact-match rate is √(0.25/50) ≈ 0.071; a 10-point true difference
  (0.10) is ~1.4 SE — marginal alone, but the go/no-go decision (Section 3) reads the
  **pooled** naive-vs-ceiling contrast across all strata, not one stratum. Pooling amplifies
  resolution (below).
- **Total N = 4 strata × 50 = 200 instances.** Pooled across strata, the SE of the overall
  exact-match rate at p≈0.5 is √(0.25/200) ≈ 0.035; the naive-vs-ceiling **difference** SE
  (two independent 200-instance measurements) is √(2)×0.035 ≈ 0.050. A 10-point pooled gap is
  ~2 SE (resolvable at ~95%); the synthesis-predicted 20–40-point gap is 4–8 SE (unambiguous).
  This satisfies criterion 5's ">= 10-point effects resolve above noise" at the decision level.

> **DRAFT note.** N=50/stratum is a proposal balancing criterion-5 resolution against Section 5's
> cost. If the owner wants each *individual stratum* (not just the pool) to resolve a 10-point
> effect at ~95%, raise to N≈100/stratum (SE of difference ≈0.071 per stratum → 10 pts ≈ 1.4 SE,
> still marginal; N≈200/stratum gets there). The pooled-decision framing is why 50 is proposed as
> the floor. Repeats=3 at temp 0 are for plumbing validation (criterion 5), not variance
> reduction — they should largely agree.

---

## 2. The two probe prompts (DRAFTED VERBATIM)

These operationalize **criterion 4** (default prompts score low; better prompts close the gap)
and **criterion 9** (floor, ceiling, and a reference prompt known in advance). Both are run
**verbatim, temperature 0, repeats=3**, against the same pinned instance pool. The only
difference between them is the instruction text — same demos, same query, same extraction.

Output extraction for both (per synthesis Q4 item 4 / MIR App B.6): take the text after the last
`Output:` line, strip surrounding whitespace and markdown fences, compare for exact string
equality to the oracle output. No CoT is forced in either prompt (synthesis: no-CoT beats
forced-CoT unanimously for the cheap tier; forcing it would confound the ceiling).

### 2.1 Prompt A — deliberately naive (floor probe)

```
Here are some examples:

vez fam qor -> VEZ vez fam qor
qor luz tam -> QOR tam luz luz
fam vez luz -> FAM vez luz luz

luz qor vez -> 
```

(Verbatim template — the demos and query are filled per instance; nothing else is added. The
model is given no task framing, no statement that a hidden rule exists, no output-format
instruction, no delimiter guidance. This is the "Repeat the transformation shown" level of
scaffolding the candidate example calls out as the naive-prompt miss.)

Verbatim instruction string used to build A:

> `<demos as "IN -> OUT" lines>\n<query IN> -> `

That is the entire prompt. No system prompt beyond the provider default.

### 2.2 Prompt B — best-effort ceiling (states all standard/default conventions)

Verbatim system + user template:

```
SYSTEM:
You are solving a hidden-rule string-transformation puzzle. Each puzzle has one fixed,
deterministic transformation rule that maps an input string to an output string. The rule
depends only on the tokens in the input (their identity, their length, their position, and
the characters at the ends of each token) — it never uses outside knowledge, randomness, or
context beyond the examples. Tokens are the whitespace-separated words. The same rule was
applied to every example below. Your job: infer that single rule from the examples, then
apply exactly the same rule to the final query.

Rules of the format:
- Read all the demonstration pairs. Each is written "INPUT -> OUTPUT".
- The transformation is a length- and position- and suffix/prefix-sensitive edit over the
  tokens: tokens may be duplicated, reordered, re-cased, or rewritten based on their local
  context (the characters immediately around each token position, up to a small window).
- Determine the exact rule that makes ALL demonstrations correct simultaneously. If more than
  one rule fits the demonstrations, choose the simplest one that fits every pair.
- Do not explain your reasoning. Output only the transformed string for the query.
- Preserve spacing and casing exactly as the rule dictates. Do not add quotes, punctuation,
  or commentary.

USER:
Demonstrations:
vez fam qor -> VEZ vez fam qor
qor luz tam -> QOR tam luz luz
fam vez luz -> FAM vez luz luz

Query:
luz qor vez ->

Output only the transformed string, on a single line prefixed with "Output:".
```

(Verbatim template — demos/query filled per instance. B states every standard convention of the
existing task: single deterministic rule, tokens = whitespace-separated words, local/suffix/
positional sensitivity matching the ISL/OSL substrate, "fits all demos / simplest rule" induction
principle, direct-answer / no-CoT, and a strict `Output:` format for robust extraction.)

**Why B is a legitimate ceiling, not cheating:** it states the *conventions and inductive bias*
of the task family (what kind of rule, how to read the format, how to answer) but **does not
reveal the specific latent rule** of any instance — the model still has to induce it from demos.
This is the "designer can write a prompt that scores ≈100% because they know the conventions"
sense of criterion 9, adapted to the reality (synthesis) that ≈100% is not reachable at the cheap
tier even with perfect conventions.

---

## 3. Three-outcome decision rule

Read the **pooled naive rate `A`** and **pooled ceiling rate `C`** (across all 200 instances ×
3 repeats × chosen models, exact-match). Thresholds below are **proposals justified from
`synthesis-capability.md`**, whose headline predictions are: naive floor ~5–25%, ceiling ~35–65%,
gap ~20–40 points, and *no cheap model saturates* (ceiling well below 100%).

Define bands (proposed):
- **HIGH** = pooled rate ≥ 70% (near the top of the synthesis ceiling range or above).
- **MID/LOW** = pooled rate ≤ 45% (at or below the synthesis-predicted ceiling midpoint).
- **Gap significant** = `C − A ≥ 15` points (≈3 SE of the pooled difference; comfortably above
  criterion 5's 10-point resolvability floor).

| Outcome | Condition | Interpretation | Action |
|---|---|---|---|
| **(a) No headroom** | `A ≈ C` (gap < 15 pts) **and** both HIGH (≥70%) | Existing shape is already near-ceiling; nothing for an optimizer to close. | Proceed to the candidate's **added-complexity design** (composition ladder / multi-family strata) — the baseline shape is too easy to validate optimizers. |
| **(b) Headroom, not prompt-closable** | `A ≈ C` (gap < 15 pts) **and** both MID/LOW (≤45%) | Raw capability deficit: even best-effort conventions don't lift the cheap tier. This is IB's "even the simplest class fails" thesis showing up. | Build the **added-complexity design AND keep base difficulty shallow** (hold k=2, small vocab; the ladder must add *decomposable* difficulty on top of a reachable floor, not push deeper into the collapse regime). |
| **(c) Prompt-closable headroom** | `C − A ≥ 15` pts (gap significant), `A` in MID/LOW, `C` meaningfully higher | The naive prompt leaves recoverable points on the table that stating conventions closes. | **Direct prompt optimization is viable on the existing shape** — run COPRO/MIPROv2/GEPA on this baseline substrate; no added complexity needed to get a usable optimizer signal. |

**Justification of the numeric thresholds from the synthesis:**
- The **15-point gap threshold** sits below the synthesis-predicted 20–40-point gap (so a real
  effect clears it) but above criterion 5's 10-point noise floor and above the ~5-point pooled
  difference-SE band (so noise doesn't trip it).
- The **70% HIGH band** is deliberately *above* the synthesis ceiling top (~65%): the synthesis
  says no cheap model saturates, so hitting HIGH would be a genuine surprise indicating the strata
  are too easy (one-shot-guessable, criterion 4 failure) — exactly the "no headroom" signal.
- The **45% MID/LOW band** is the synthesis ceiling midpoint; a ceiling stuck at or below it means
  conventions alone can't move the task, which is the capability-deficit (b) reading.
- **Format-refusal caveat (synthesis Q3):** Claude-Haiku-class and Gemini-Flash-class models can
  crater to single digits purely from misreading the I/O block (MIR refusal; WILT Haiku 1/50,
  Flash-8b 0/50). If a *specific model* shows A≈C≈near-zero while others show a healthy gap, that
  is **not** outcome (b) — it is a per-model representation problem. Report per-model rates, not
  only the pool, and treat a single model's floor-crater as a format bug to fix before concluding.

---

## 4. Model selection — OPEN DECISION (owner decides)

The synthesis is blunt: **no dossier tests c23's exact cheap tier**; all cheap-tier numbers are
same-family proxies. The table below is what the papers actually measured, mapped to the cheap
models we would plausibly run. **This is not a recommendation — it is the evidence the owner needs
to choose.**

### 4.1 What the papers tested (from the dossiers)

| Model (as tested) | Paper / setting | Result | Proxy for (our tier) |
|---|---|---|---|
| o3-mini | IB moderate (k4/|Σ|4/r3) | 10/10/30 compat (only nonzero model); 5.69% weighted leaderboard | frontier reasoning (not our tier) |
| o1-mini | IB moderate | 0.00 compat; MIR best ~0.696 no-CoT | frontier reasoning |
| GPT-4o | IB moderate: 0/0/0 compat; IB easy ISL k2/v2/r1: **40 compat**; MIR-Core 0.540 no-CoT | **GPT-5-mini** (non-reasoning) |
| DeepSeek-V3 / Chat | IB moderate 0/0/0; recall 3–23; WILT v2.5-chat 6/50 | **DeepSeek-Chat** |
| DeepSeek-R1 | MIR 0.757 no-CoT (but reasoning model — overstates plain Chat) | over-states DeepSeek-Chat |
| Claude-3.5-Haiku | MIR: "surprisingly low," refuses / misreads I/O block | **Claude Haiku 4.5** (format-refusal risk) |
| Claude-3-Haiku | WILT 1/50 (fails to use turns) | Claude Haiku 4.5 (refusal risk) |
| Claude-3.5-Sonnet | MIR 0.775 no-CoT / 0.585 CoT | mid/frontier (optional upper anchor) |
| Gemini-1.5-Flash / Flash-8b | WILT 7/50 / **0/50**; MIR-2.0-Flash anomalous | **Gemini-2.5-Flash** (refusal/anomaly risk) |
| Llama-3.3-70B | IB easy ISL k2/v2/r1: **60→80 with 0→3 shot**; ~0–10 at k4 | open cheap-ish; most encouraging on-target datapoint |

### 4.2 Proposed candidate set (DRAFT — for owner to finalize)

A **3-model set** balancing coverage, cost, and the synthesis's warnings:
- **GPT-5-mini** — cleanest GPT-4o proxy; GPT-4o is the best-behaved cheap datapoint on-target.
- **DeepSeek-Chat** — cheapest, and the synthesis's DeepSeek proxies are the least refusal-prone
  of the cheap tier.
- **Claude Haiku 4.5** — deliberately included *because* it is the format-refusal risk; if the
  baseline is going to break on I/O-block comprehension, we want to see it here, not in the
  expensive run. (Serves criterion 13 — failure paths exercised on purpose.)

Optional 4th: **Gemini-2.5-Flash** (second refusal-risk data point) or **Claude-3.5-Sonnet-class**
(an upper anchor to see how much of the gap is tier-limited vs. task-limited).

### 4.3 Tradeoffs (stated, not resolved)
- **More models** = better separation of "task headroom" from "this one model," and satisfies the
  positioning reviewers' "≥2–3 models" demand — but multiplies Section 5 cost linearly.
- **Including refusal-risk models (Haiku/Flash)** surfaces the representation failure early (good
  for the debug loop) but risks a per-model floor-crater that muddies the pooled decision rule —
  mitigated by reading per-model rates (Section 3 caveat).
- **A reasoning model** (o1/o3-mini class) would show the ceiling of the substrate but is off-tier
  and expensive; probably out of scope for a *baseline headroom* measurement.

**→ OWNER DECISION: finalize the model set and whether to include a 4th/anchor model.**

---

## 5. Cost & wall-clock estimate per full baseline run

**Per-instance token estimate** (short strings, small vocab, few demos):
- Prompt A (naive): ~120 input tokens, ~20 output tokens.
- Prompt B (ceiling): ~400 input tokens (system conventions + demos), ~20 output tokens.
- **Blended average: ~300 input + ~20 output ≈ 320 tokens/instance/prompt.**

**Call count per full baseline run:**

```
calls = N_instances (200) × prompts (2: A and B) × models (3) × repeats (3)
      = 200 × 2 × 3 × 3 = 3,600 LLM calls
```

**Token volume:**
```
input  ≈ 3,600 calls × ~260 avg input tok  ≈ 0.94 M input tokens
output ≈ 3,600 calls × ~20 output tok       ≈ 0.07 M output tokens
```
(Using blended per-prompt input; A and B averaged.)

**Cost (order-of-magnitude, cheap-tier pricing ~$0.15–$0.60 / M input, ~$0.60–$2.40 / M output;
3 models averaged):**
```
input  ≈ 0.94 M × ~$0.40/M ≈ $0.38
output ≈ 0.07 M × ~$1.50/M ≈ $0.11
per full baseline run ≈ $0.50 (order $1, well under "cents-to-a-dollar" — criterion 3)
```

**Wall-clock:** 3,600 calls at short output length. At a modest 10 concurrent requests and ~1.5 s
latency/call, ≈ 3,600 × 1.5 / 10 ≈ **~9 minutes**; at 20 concurrent, **~4.5 minutes**. Comfortably
inside criterion 14's "minutes, not hours," assuming no rate-limit bottleneck (the criterion's
explicit caveat).

> **DRAFT note.** Costs scale linearly with model count and N. Adding a 4th model → ~$0.67/run,
> ~12 min. Raising N to 100/stratum (400 total) doubles both. All figures are order-of-magnitude;
> pin real per-model pricing before the owner signs off.

---

## 6. Rubric-mapping table

| Design choice | Serves criteria | How |
|---|---|---|
| Single LLM-call driver + single 0/1 exact-match scorer | 1, 2 | One LLM Call Node + one Eval Node; deterministic string equality, no sandbox/partial credit |
| Short strings, ~20-token outputs, ~320 tok/instance | 3, 14 | Bounded per-rollout cost (~$0.50/run) and minutes wall-clock |
| **Naive prompt A vs. ceiling prompt B, run verbatim** | **4, 9** | Operationalizes "default scores low, better prompt closes gap" and "known floor/ceiling/reference prompt" |
| **4 strata × 50 = 200 instances; pooled-difference SE ≈0.05** | **5** | 10-point effect resolves above noise at the decision level (≥10-pt effects on 10–20-task internal evals) |
| Temperature 0, repeats=3 | 5 | Determinism so prompt differences (not sampling) drive score; 3 repeats validate plumbing, should agree |
| **Fresh seeds via patched generator; never published IB instances; nonce-vocab option** | **8** | Contamination-proof; models cannot already know answers |
| Latent single rule per instance, oracle re-applies rule to held-out query | 7, 10, 11 | Generated pool, known ground truth; rule is the latent structure; "expected X got Y" is diagnostic |
| I/O demos are the only channel for the rule | 12 | Few-shot demos are load-bearing by construction |
| Haiku/Flash included as refusal-risk models; per-model rates reported | 13 | Format-violation / refusal failure paths exercised on purpose, not hidden by the pool |
| `sorted(...)` seed-determinism fix (not just `PYTHONHASHSEED`) | 5, 7, 8 | Same-seed re-runs reproduce; stable task identity; held-out-seed generalization defensible |
| ISL/L-OSL/R-OSL as independent strata, one generation diff each | 7, 10 | Independent latent-rule strata, aggregation crosses strata (avg by repeat within task, then across strata) |

---

## 7. Open decisions (awaiting owner)

1. **Model set (Section 4).** Finalize the 3-model set (proposed: GPT-5-mini, DeepSeek-Chat,
   Claude Haiku 4.5); decide whether to add a 4th refusal-risk model (Gemini-2.5-Flash) and/or an
   upper anchor (Sonnet-class). This is the flagged OPEN DECISION.
2. **N per stratum.** Proposed 50 (pooled-decision resolution). Owner may raise to 100–200 if
   *per-stratum* 10-point resolution is required, at linear cost.
3. **Keep or drop the k=3 stratum (S4).** Contingent on the pilot: if S4 sits near floor for all
   models, drop it and hold k=2. Owner sets the reachability bar for keeping it.
4. **Vocab size.** Fixed at |Σ|=4 in this draft; owner confirms, or authorizes it as a curation
   lever if k=2 proves one-shot-guessable.
5. **Decision-rule thresholds (Section 3).** Confirm the 15-point gap / 70% HIGH / 45% MID-LOW
   bands, or adjust to the owner's risk tolerance.
6. **Prompt B wording.** The verbatim ceiling prompt states conventions but not the specific rule;
   owner should confirm it doesn't over-hint (which would inflate the ceiling and mask headroom).
7. **CoT axis.** This draft fixes both prompts to direct-answer / no-CoT per the synthesis
   (no-CoT beats forced-CoT unanimously at the cheap tier). Owner confirms we don't sweep CoT in
   the baseline (it is a phase-2 optimizer confound to control, not a baseline variable).
8. **Pilot gate.** Section 1.2 / repo Step 5 require a small pilot to confirm reachability before
   freezing the pool. Owner approves the pilot-first sequencing and the go/no-go it feeds.

---

## Summary (10 lines)

1. Goal: measure headroom on the EXISTING single-rule ISL/OSL shape (fresh seeds, exact-match,
   temp 0, repeats=3) BEFORE building any added complexity.
2. Strata: 4 strata on the primary axis k (ISL/L-OSL/R-OSL at k=2, plus ISL at k=3), |Σ|=4 fixed,
   1 rule fixed; k=4 excluded as unreachable per synthesis.
3. N = 50/stratum → 200 total; pooled naive-vs-ceiling difference SE ≈0.05, so a 10-point effect
   resolves at the decision level (criterion 5).
4. Two verbatim probes: Prompt A (bare demos + query, no framing) and Prompt B (full conventions:
   single deterministic token-rule, direct-answer, strict `Output:` format) — criteria 4 and 9.
5. Decision rule: gap<15pts & both ≥70% → no headroom → build added complexity; gap<15pts & both
   ≤45% → capability deficit → added complexity + keep base shallow; gap≥15pts → prompt-closable →
   optimize on existing shape.
6. Thresholds justified from synthesis: predicted floor ~5–25%, ceiling ~35–65%, gap ~20–40pts,
   no cheap model saturates — so 70% HIGH would be a "too easy" surprise.
7. Model selection is OPEN: proposed GPT-5-mini / DeepSeek-Chat / Claude Haiku 4.5 (Haiku/Flash
   carry synthesis-documented format-refusal risk, deliberately included); owner finalizes.
8. Cost: ~3,600 calls/run (200×2×3×3), ~0.94M in / 0.07M out tokens, ≈$0.50/run, ~5–9 min wall-clock.
9. Evidence caveats surfaced: no dossier tests our exact cheap tier (all proxies); serialization-
   format effects are an untested gap; IB's rule-reconstruction metric under-predicts our easier
   held-out-query exact-match, biasing the ceiling upward.
10. Open decisions: model set, N, keep/drop k=3, vocab size, decision thresholds, Prompt B wording,
    CoT-fixed vs. swept, and a pilot reachability gate before freezing the pool.
```
