# c18 Baseline Experiment Spec — Depth-Controlled Synthetic Deduction (True/False)

> **STATUS: DRAFT.** Awaiting project-owner sign-off. Numbers marked with a
> confidence tag; every extrapolated floor/ceiling is grounded in
> `synthesis-capability.md` and flagged where the evidence is thin. Sections 4
> (model selection) and 7 (open decisions) are deliberately left open for the
> owner to resolve.

## Purpose and scope

Measure **headroom on the existing PrOntoQA task shape** before building any of
c18's added-complexity axes (the Unknown/open-world label, the second
constraint-puzzle stratum, soft-rule strata, or the independent forward-chaining
oracle as a *design* element). The question this experiment answers:

> On freshly reseeded, contamination-free PrOntoQA True/False instances at the
> codebase's native hop depths, how far below a ceiling prompt does a naive
> prompt sit, and is that gap **prompt-closable** or a **raw-capability
> deficit**?

The answer routes the next decision (Section 3): proceed to added complexity,
proceed to added complexity *and* keep base difficulty shallow, or run direct
prompt optimization on the existing shape.

### Fixed by prior decisions (NOT open here)

- Instances regenerated **as-is** from `asaparov/prontoqa`
  (`run_experiment.py --model-name json`), fresh `--seed` per split, fresh
  `--ontology fictional` nonce ontologies. Never reuse published instances
  (rubric criterion 8; contamination).
- **Existing hop-depth settings and native question format.** No new axes: no
  added Unknown-label handling beyond what the codebase natively emits, no
  distractor or soft-rule strata **added by us** (the codebase's own
  `--distractors` setting is a separate matter — see Open Decision O1).
- Output is **True/False, exact match** against the generator's stored `answer`
  gold, scored 0/1 by a simple function against an independent oracle
  (forward-chaining fixpoint re-derivation, ~30–50 lines; risk item in
  `c18.md`). No partial credit.
- **Temperature 0**; **repeats = 3** (`repeat_id` distinct, to validate
  plumbing; determinism means they should largely agree — rubric criterion 5).

---

## 1. Strata design

### Axes chosen (candidate-appropriate, from `c18.md` + repo note)

The candidate doc and repo note expose several native difficulty dials. For a
**headroom-only** baseline on the existing shape we take the two that (a) are
native to the reseed-only path, (b) the literature shows actually move accuracy,
and (c) do **not** introduce a new label or stratum:

| Axis | Levels | Native knob | Why included |
|---|---|---|---|
| **Hop depth** | D1, D2, D3, D5 | `--min-hops/--max-hops` loop | The single most robust accuracy dial in the literature — monotonic decline with depth across every paper (`synthesis-capability.md` §1). Gives the incremental ladder and a designer-known ceiling. |
| **Ontology type** | fictional (primary) | `--ontology fictional` | Fixed to fictional for contamination-resistance (criterion 8). `true`/`false` ontologies are **excluded** because real-world priors create a ceiling effect (PrOntoQA: `true` ontology does not degrade D3→D5; `2210.01240` baseline_relevant) — that would confound the headroom read. |

**Ontology type is therefore held constant (fictional), not a stratum axis** in
this baseline. The stratification axis is **hop depth alone**. This is a
deliberate simplification consistent with "measure headroom on the existing
shape": a single clean depth ladder, one contamination-safe surface.

Depth levels **D1, D2, D3, D5** chosen to span the informative band: D1 as the
near-solved anchor (expected ceiling-adjacent), D2/D3 as the predicted mid-band,
D5 as the hard tail where cheap models approach the two-way floor
(`synthesis-capability.md` §1, §5). D4 dropped to keep N bounded; D0 dropped as
trivially saturated. (Owner may re-add — Open Decision O2.)

### N per stratum and total N

**Resolvability is the binding constraint (rubric criterion 5):** a meaningful
prompt change must produce a ≥10-point effect that exceeds residual noise on a
**10–20 task internal eval**. Because decoding is temperature 0, per-task
behavior is near-deterministic given a prompt, so the residual noise on a 10–20
task internal eval is small and a 10-point effect (≥1–2 of 10–20 tasks flipping)
resolves cleanly. The baseline pool must be large enough that:

1. Each depth stratum can supply **disjoint** internal-eval subsets (≥10–20
   tasks) *and* a held-out official split (criterion 7), without reusing tasks.
2. Per-stratum aggregate accuracy has a tight enough CI to read a ≥10-point
   floor-vs-ceiling gap as real. At N=100/stratum a Wilson 95% CI half-width is
   ≈±8–10 pts near p=0.7; PrOntoQA itself used **400/condition** with 95% Wilson
   intervals (`2210.01240` baseline_relevant) — the reviewer-expected standard
   (`synthesis-positioning.md` §3c, item 7).

**Proposed:** **N = 150 instances per depth stratum × 4 strata = 600 instances**
total for the baseline pool.

- 150/stratum gives a Wilson 95% half-width ≈ ±7–8 pts near p=0.7 — enough to
  resolve a 10-point floor→ceiling gap per stratum, and comfortably enough to
  carve disjoint 10–20-task internal-eval subsets plus a ≥100-task official
  split.
- 600 total keeps a single full baseline run cheap (Section 5) while exceeding
  the "hundreds of tasks" pool requirement (criterion 7).
- **Owner may prefer 400/stratum** to exactly match PrOntoQA's CI regime for a
  publishable number (Open Decision O3); 150 is the cheap-iteration floor, 400
  is the publication-grade target.

Aggregation crosses strata by averaging exact 0/1 scores first by repeat within
task, then across tasks within stratum, then reporting **per-depth** and a
depth-weighted overall (rubric "Aggregation must cross strata" callout;
per-depth breakdown is a reviewer demand, `synthesis-positioning.md` §3c item 3).

---

## 2. The two probe prompts (DRAFTED VERBATIM)

Both prompts take the generator's native `question` (facts + rules) and `query`
concatenated, and must elicit a single `True`/`False` token for exact match.
These operationalize rubric criterion 4 (default scores low, better prompt
closes the gap) and criterion 9 (floor, ceiling, reference prompt known in
advance).

### (a) Naive prompt — deliberately minimal

```
{question}

{query}

Answer True or False.
```

Rationale: no chain-of-thought, no statement of the closed-world convention, no
statement that only the listed rules may be used, no output-format discipline
beyond "True or False." This is the intended **floor** — it invites
surface-plausibility answering, which the `c18.md` example instance shows misses
the two-hop chain. Per `synthesis-capability.md` §5 this should land materially
above the 50% two-way floor (clean fictional surface, easy A≈2 rule regime) but
well below the ceiling prompt.

### (b) Ceiling prompt — best-effort, states all standard conventions

```
You are a careful deductive reasoner. You are given a set of facts and if-then
rules, followed by a single query statement. Every predicate is fictional and
carries no real-world meaning — rely ONLY on the facts and rules given, never on
outside knowledge or surface plausibility.

Determine whether the query statement is entailed by the facts and rules under
the closed-world assumption: a statement is True if it can be derived by
chaining the given rules from the given facts, and False otherwise. Apply the
rules step by step, following each "every X is a Y" / "X are (not) Z" rule in
order, until you reach the queried property.

Facts and rules:
{question}

Query:
{query}

Reason step by step through the chain of rules, then end your reply with exactly
one word on its own final line: either
True
or
False
```

Rationale — each clause maps to a lever the literature shows works:

- "rely ONLY on the facts and rules … never on outside knowledge or surface
  plausibility" → blocks the property-string-match shortcut and the
  real-world-prior shortcut (`synthesis-capability.md` §4 item 4; `2210.01240`
  App A.2).
- "closed-world assumption … True if derivable … False otherwise" → states the
  native 2-way convention precisely (the CWA/Unknown ambiguity risk in `c18.md`
  is sidestepped here because the native format is 2-way — no Unknown to
  mishandle).
- "Reason step by step" (CoT) → the single largest reliable lever for
  non-reasoning cheap models: +8 to +12 pts across JustLogic
  (`synthesis-capability.md` §4 item 1).
- "following each rule in order" → naming the reasoning form / ordering, a
  measured lever (`synthesis-capability.md` §4 items 2–3).
- "exactly one word on its own final line" → makes exact-match extraction clean;
  guards the format-violation failure path (rubric criterion 13).

**Note (thin evidence):** no dossier ran a CoT-vs-naive ablation *on PrOntoQA
label-only for a modern cheap model* — the +8–12 CoT lift is extrapolated from
JustLogic's harder NL surface (`synthesis-capability.md` §3, §5, uncertainty
"moderate"). Treat the predicted gap magnitude as a hypothesis this run tests,
not a known quantity.

---

## 3. Three-outcome decision rule

Let **N** = naive-prompt aggregate accuracy, **C** = ceiling-prompt aggregate
accuracy, both depth-weighted over the four strata, cheap tier, temp 0.
Thresholds are grounded in `synthesis-capability.md` §5, which predicts (cheap
tier, 2-way native, distractors ON): **naive floor ~60–78%, ceiling ~75–90%**,
with the strong caveat that if distractors are OFF the naive prompt may already
score **85–95%** by shortcut-matching.

Define **HIGH ≥ 85%**, **MID = 65–85%**, **LOW < 65%** for the *ceiling* score;
and **gap G = C − N**, with **G "small" < 10 pts** (below the criterion-5
resolvability threshold) and **G "large" ≥ 15 pts**.

| Outcome | Signature | Interpretation | Next action |
|---|---|---|---|
| **(a) No headroom** | N ≈ C, both HIGH (C ≥ 85%, G < 10) | Task is near-saturated even naively; likely the distractor-shortcut regime (§5). | **Proceed to the candidate's added-complexity design** (distractors on/deeper, Unknown axis) — the existing shape is too easy to discriminate. |
| **(b) Headroom, not prompt-closable** | N ≈ C, both MID/LOW (C < 85%, G < 10) | Both prompts stuck below ceiling → raw capability deficit at depth, not a prompt gap (matches the depth-collapse finding, §1; CoT does not remove depth degradation, §2). | **Proceed to added-complexity design AND keep base difficulty shallow** — deep hops are unsolvable by prompting, so the discriminating band is shallow-to-mid depth. |
| **(c) Prompt-closable headroom** | N ≪ C (G ≥ 15), C in MID/HIGH | The ceiling prompt's CoT + convention-stating closes a real gap → the gap is prompt-shaped. | **Direct prompt optimization is viable on the existing shape** — run COPRO/MIPROv2/GEPA on the reseed-only task; this is the publishable Area-9 path (`synthesis-positioning.md` §3). |

**Threshold justification (from `synthesis-capability.md`):**

- The **85% HIGH** cut sits at the top of the predicted ceiling band (75–90%)
  and inside the distractor-off shortcut band (85–95%); a naive score at/above
  it is the §5 saturation signature.
- The **65% LOW** cut is just below the predicted naive floor (60–78%); a
  ceiling score below it means even best-effort prompting cannot lift the
  aggregate, i.e. deep-hop capability collapse dominates the mix.
- The **G ≥ 15 pts** "large gap" cut is comfortably above the criterion-5
  resolvability floor (10 pts) so outcome (c) is not a noise artifact; it is
  consistent with the CoT lift magnitude (+8–12 at the JustLogic surface,
  expected somewhat larger at PrOntoQA's cleaner surface).
- **Per-depth override:** even when the aggregate reads (a), inspect the D3/D5
  strata — if the gap concentrates there while D1/D2 saturate, that is
  effectively outcome (c) restricted to the hard tail and argues for prompt
  optimization scoped to deep instances. Report the per-depth gap regardless.

**Evidence-thinness caveat:** the exact band boundaries are extrapolations — no
dossier tests this precise (cheap-2026-tier × 2-way-native × PrOntoQA-label-only)
cell (`synthesis-capability.md` §5, uncertainty "moderate-to-high"). The rule is
robust to being *off* on absolute levels because it keys on the **shape** (N≈C
vs N≪C, and where C lands), not on hitting a precise number.

---

## 4. Model selection — OPEN DECISION (owner decides)

### Evidence table: models tested by the seven dossier papers

Only label/answer-accuracy rows shown (c18 is label-only); proof-accuracy
collapses do **not** translate 1:1 to label accuracy (`synthesis-capability.md`
preamble). Tier tags are mine.

| Model | Tier | Task (label space) | Result | Source |
|---|---|---|---|---|
| DeepSeek R1 | frontier reasoning | JustLogic 3-way | **80.9%** (best; only model > human avg 73%) | `2501.14851` |
| o1 (2024-12) | frontier reasoning | JustLogic 3-way | 72.9% (depth-7 "Uncertain" degeneracy) | `2501.14851` |
| o1-mini | frontier reasoning | JustLogic 3-way | 62.0% (sharp deep-depth decline) | `2501.14851` |
| DeepSeek R1 Distill Qwen 14B | mid reasoning | JustLogic 3-way | 61.7% | `2501.14851` |
| o4-mini | frontier reasoning | SATBench (search) | 89.3% overall / **65.0% hard-UNSAT** | `2505.14615` |
| DeepSeek R1 | frontier reasoning | SATBench | 87.8% overall | `2505.14615` |
| DeepSeek-V3 | frontier chat | SATBench | 84.0% overall | `2505.14615` |
| Claude-3.7-Sonnet | frontier chat | SATBench | 74.8% overall | `2505.14615` |
| GPT-4o | mid chat | JustLogic 3-way | 0-shot 53.8% / CoT 65.6% / ToT 71.4% | `2501.14851` |
| **GPT-4o-mini** | **cheap** | JustLogic 3-way | **0-shot 53.0% / few-shot 54.7% / CoT 51.8%** | `2501.14851` |
| GPT-4o-mini | cheap | SATBench | 53.9% (near-random) | `2505.14615` |
| **Llama3-8B** | **cheap/open** | JustLogic 3-way | **0-shot 49.8% / CoT 57.8%** | `2501.14851` |
| Llama3-70B | mid/open | JustLogic 3-way | 0-shot 53.1% / CoT 64.6% | `2501.14851` |
| GPT-3.5-Turbo | (dated) cheap | FLD 3-way (hard axiom set) | answer-acc 35.8% / 37.6% (≈random) | `2308.07336` |
| LongAlpaca-13B | cheap/open | FLD 3-way | 21.2% / 19.6% (below floor) | `2308.07336` |
| GPT-4 | frontier (2023) | FLD 3-way | answer-acc 52.4% / 49.4% | `2308.07336` |
| Mistral-7B-Instruct | cheap/open | Multi-LogiEval Yes/No | PL d1 80.8% → d5 44.4%; FOL d5 20.0% | `2406.17169` |
| Yi-34B-Chat | mid/open | Multi-LogiEval | PL d1 85% → d5 26.7%; FOL d5 13.3% | `2406.17169` |
| Orca-2-13B | cheap/open | Multi-LogiEval | PL d5 15.6%; FOL d5 6.7% (lowest) | `2406.17169` |
| Gemini-Pro | mid chat | Multi-LogiEval | PL d1 90% → d5 60%; FOL 76.9% → 53.3% | `2406.17169` |
| ChatGPT (3.5) | cheap chat | Multi-LogiEval | PL d1 91.7% → d5 44.4%; FOL d5 37.8% | `2406.17169` |
| GPT-4 | frontier (2023) | Multi-LogiEval | PL d1 89.2% → d5 66.7%; FOL d5 66.7% | `2406.17169` |
| text-davinci-002 | dated | PrOntoQA (proof-acc) | 1&3-hop handled; 5-hop top-down → chance | `2210.01240` |
| PaLM-540B / LLaMA-65B / FLAN-T5-11B | dated | PrOntoQA-OOD | size ≉ perf; charts only, no exact % | `2305.15269` |

**Key reads for our tier:** the closest task-shape proxy (JustLogic 3-way,
exact-match, single-shot) puts *cheap-tier* models at ~50–57% zero-shot, ~52–65%
with CoT. Our native format is 2-way (50% floor, not 33%), cleaner surface,
easier rules, and the 2026 cheap tier is stronger than these 2024 proxies — all
four factors push c18 **above** these numbers (`synthesis-capability.md` §3, §5).

### Proposed candidate set for our baseline (NOT finalized)

The 2026 cheap tier named as the quick-test target in `synthesis-capability.md`
§3 (no dossier tests these directly):

- **Gemini-2.5-Flash**
- **GPT-5-mini**
- **Claude Haiku 4.5**
- **DeepSeek-Chat**

Proposed baseline: **run 2 of these 4** for the first headroom pass (one to
anchor cost estimates, a second to check cross-model agreement of the go/no-go
read), expand to all 4 only if the two disagree on the outcome bucket.

**Tradeoffs (for the owner):**

- **More models** → stronger evidence the headroom read is not model-specific,
  and cross-model transfer becomes checkable (a reviewer demand,
  `synthesis-positioning.md` §3c item 5); but linear cost/wall-clock growth.
- **A single cheap model** → cheapest fastest iteration, but a single-model read
  is explicitly *not publishable* (`synthesis-positioning.md` §3e) and risks
  reading model-idiosyncrasy as task property.
- **Include one mid/frontier anchor** (e.g. add Claude Sonnet or DeepSeek-V3) →
  bounds the ceiling and shows whether the gap is cheap-tier-specific; adds
  cost. The quick-test's whole point is *cheap* iteration, so a frontier anchor
  is optional and probably a later ablation, not the first pass.
- **Reasoning vs non-reasoning cheap models** → GPT-5-mini / DeepSeek-Chat
  behave differently on depth than pure chat models; mixing them tests whether
  the go/no-go read is robust to that, but muddies a clean single-tier number.

**Owner decides:** which of the 4, how many, and whether to add a
mid/frontier anchor. (Open Decision O4.)

---

## 5. Cost & wall-clock estimate (per full baseline run)

### Token model per instance

Native PrOntoQA fictional instance surface is short (facts+rules paragraph +
one-line query). From the run-verified output (`asaparov-prontoqa.md`: 5-example
JSON ≈ 32 KB including 8 in-context demos per example), a **single** theory+query
without demos is small. Estimates (state as assumptions — Open Decision O5):

| Component | Naive prompt | Ceiling prompt (CoT) |
|---|---|---|
| Input tokens (instance + prompt scaffold) | ~250 | ~400 |
| Output tokens | ~3 (`True`/`False`) | ~250 (CoT chain + label) |
| **Total tokens/instance** | **~253** | **~650** |

Deeper hops (D5) grow the facts/rules block; PrOntoQA hit a 2049-token limit only
at 3-hop *with 8 demos* (`2210.01240` baseline_relevant). Demo-free, even D5 stays
well under ~600 input tokens. Use **~500 input avg** as a safe upper bound.

### Full-run multiplier

```
instances (N)        = 600  (150 × 4 depth strata)
prompts              = 2    (naive + ceiling)
models               = M    (proposed 2; up to 4)
repeats              = 3
→ LLM calls/run      = 600 × 2 × M × 3 = 3,600 × M
```

Token volume per run (using ceiling avg ~650 tok as the conservative blended
per-call figure, since half the calls are the cheap naive prompt ~253 → blended
~450 tok/call):

```
tokens/run ≈ 3,600 × M × 450  ≈ 1.62M × M  total tokens
             (≈ 0.9M input + 0.7M output, per model)
```

### Cost (order-of-magnitude, cheap tier)

At a representative cheap-tier blended rate of **~$0.30–0.60 per 1M tokens**
(cheap 2026 tier; exact provider prices are an Open Decision, O6):

| M models | Tokens/run | Est. cost/run |
|---|---|---|
| 1 | ~1.6M | ~**$0.5–1** |
| 2 (proposed) | ~3.2M | ~**$1–2** |
| 4 | ~6.5M | ~**$2–4** |

**Comfortably "cents-to-a-few-dollars per run"** — satisfies rubric criteria 3
and the goal ("cents per run, in minutes"). This is the *baseline headroom* run,
not an optimizer sweep (optimizer runs multiply by iterations/candidates).

### Wall-clock

3,600×M short calls at temp 0. At modest concurrency (e.g. 20 in flight) and
~1–3 s/call, one model's 3,600 calls ≈ **3–9 minutes**; 2 models ≈ **6–18 min**;
4 models ≈ **12–36 min** — assuming no rate-limit bottleneck (rubric criterion
14). Ceiling-prompt CoT calls dominate wall-clock (longer outputs). **Finishes
in minutes, not hours** — satisfies criterion 14. (Concurrency/rate-limit
headroom per provider is unverified locally — Open Decision O7.)

---

## 6. Rubric-mapping table

| Design choice (this spec) | Rubric criteria served |
|---|---|
| One LLM call → one exact-match eval; label-only `True`/`False` | **1**, **2** |
| Short inputs, bounded output (naive ~3 tok; CoT capped) | **3**, **14** |
| Naive vs ceiling probe prompts; three-outcome gap rule | **4**, **9** |
| Temp 0; repeats=3; N sized so ≥10-pt effect > noise on 10–20-task subset | **5** |
| Per-depth + depth-weighted aggregation across strata; disjoint internal/official splits | **5**, **6**, **7**, "aggregation crosses strata" callout |
| Fresh `--seed` per split; `--ontology fictional` nonce symbols; no published reuse | **8** |
| Floor (naive) & ceiling (best-effort) known/estimated in advance; falsifiable go/no-go | **9** |
| Hop-depth ladder = latent difficulty dial; per-depth gap inspectable | **10** |
| Exact-match "expected X got Y"; independent oracle re-derives label | **2**, **11** |
| Ceiling prompt's explicit rule-chaining convention = a discoverable latent rule | **10**, **12** |
| Ceiling prompt's "exactly one word final line" format discipline; format-violation path exercised | **13** |
| ~500-instance pool from a pinned generator version + seed manifest | **7**, **8** |
| Minutes-scale wall-clock at cheap tier, concurrency-bounded | **14** |

Criteria **4, 5, 8, 9** (the emphasis set) are each served by ≥2 independent
design choices above.

---

## 7. Open decisions (awaiting owner)

| # | Decision | Why it matters | Default if unspecified |
|---|---|---|---|
| **O1** | **Distractors ON or OFF** in the reseed-only baseline (codebase `--distractors {none,relevant,irrelevant}`). | **Single biggest swing factor** (`synthesis-capability.md` §5, §4 item 4): distractors OFF risks naive-prompt saturation at 85–95% via property-string-matching, collapsing all headroom. Must verify the generator's *default* in the codebase before running. Note: "no distractor strata **added by us**" does not decide whether to leave the native distractor knob on. | **ON (`relevant`)** — required to avoid the trivial shortcut and keep the headroom read meaningful. Flagged as the top thing to confirm. |
| **O2** | Depth levels: proposed {D1,D2,D3,D5}. Add D4? Add D0? Extend beyond D5? | Determines where the informative mid-band sits and whether the hard tail is represented. | {D1,D2,D3,D5}. |
| **O3** | N per stratum: 150 (cheap-iteration) vs 400 (PrOntoQA CI-grade). | Publication-grade CIs (Wilson, 400/condition) vs fast iteration. | 150 for first pass; 400 for any publishable run. |
| **O4** | Model set (Section 4): which of {Gemini-2.5-Flash, GPT-5-mini, Haiku 4.5, DeepSeek-Chat}, how many, and whether to add a mid/frontier anchor. | Cost/wall-clock vs evidence strength; single-model is not publishable. | 2 cheap models first pass. |
| **O5** | Confirm per-instance token estimates against a real reseeded batch (esp. D5 input size demo-free). | Cost/wall-clock accuracy. | ~500 input / ~250 CoT output upper bound. |
| **O6** | Provider price sheet for the chosen cheap tier. | Turns the token estimate into a dollar figure. | ~$0.3–0.6 / 1M tokens blended. |
| **O7** | Concurrency / rate-limit headroom per provider at intended parallelism. | Wall-clock claim (criterion 14) assumes no rate-limit bottleneck. | 20 in flight, assumed unthrottled. |
| **O8** | Independent-oracle scope: full forward-chaining fixpoint (30–50 LOC) vs spot-check subset. | The native label is *definitional* (negation flag), not a prover verdict (`c18.md` risk; `asaparov-prontoqa.md` red-flag 5) — a generation bug is invisible in the label alone. Reviewers demand a verifier-soundness check (`synthesis-positioning.md` §3c item 8). | Full fixpoint check on 100% of instances (cheap for A≈2 chaining). |
| **O9** | Ceiling-prompt exact wording — freeze after 2–3 pre-freeze gate reads? | `c18.md` names 2–3 pre-freeze gates for CWA/Unknown wording; even at 2-way, the "closed-world True/False" phrasing should be sanity-checked so it doesn't itself leak the answer or bias toward one label. | Freeze §2(b) after one gate read. |

---

## Appendix: grounding notes on floor/ceiling expectations

- **Floor (naive), predicted ~60–78% aggregate** — extrapolated from JustLogic
  3-way zero-shot ~53% for cheap tier, adjusted up for 2-way format, cleaner
  fictional surface, easier A≈2 rules, stronger 2026 tier
  (`synthesis-capability.md` §5). **Uncertainty moderate-to-high**; distractors
  OFF could push it to 85–95% (O1).
- **Ceiling (best-effort CoT), predicted ~75–90% aggregate** — floor + CoT
  (+8–12) + convention-naming, but the depth ceiling is real and prompting does
  not remove deep-hop degradation, so aggregate stays short of ~95%
  (`synthesis-capability.md` §5). **Uncertainty moderate.**
- **Thin-evidence flag:** no dossier measures the exact (2026-cheap × 2-way
  native × PrOntoQA label-only) cell; the closest PrOntoQA numbers are *proof*
  accuracy on a 2022 model. The decision rule (Section 3) is deliberately
  shape-based to survive absolute-level error.
