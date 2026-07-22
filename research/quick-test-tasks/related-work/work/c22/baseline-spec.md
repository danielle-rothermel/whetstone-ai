# DRAFT — Baseline Experiment Spec · c22 Stacked Verifiable-Constraint Micro-Generation

**STATUS: DRAFT.** Every number below is a proposal for the project owner to accept, adjust, or reject. Floor/ceiling expectations are grounded in `synthesis-capability.md`; where evidence is thin it is flagged inline as `[thin]`.

## Purpose

Measure **headroom on the EXISTING task shape** — the reseed-only baseline — before building any of c22's added-complexity layers (conditional Selection/discovery atoms, IFBench OOD atoms, seeded-generator hardening). The single question this experiment answers: *is there a naive→ceiling gap on plain stacked IFEval atoms, and if so, is it prompt-closable or a raw-capability deficit?* The answer routes the project down one of three branches (§3).

## Fixed constraints (from prior decisions — not open)

- **Oracle:** Standard IFEval checker atoms ONLY. Reuse the `google-research` IFEval checker library as-is (`instructions.py` / `instructions_util.py` / `instructions_registry.py`, `INSTRUCTION_DICT`), each class's `check_following` as the 0/1 atom oracle. Repo note: [../repos/google-research-instruction-following-eval-ifeval.md](../repos/google-research-instruction-following-eval-ifeval.md). Scoring glue reuses `evaluation_lib.test_instruction_following_strict`.
- **No OOD atoms, no conditional Selection constraints** (those are the added-complexity design this experiment gates).
- **Composition:** 3–5 atoms sampled fresh per instance over trivial micro-tasks; **all-checkers-pass strict 0/1** (`all(check_following(...))`), no partial credit.
- **Fresh seeds only** — never published instances. Instances synthesized per `random.seed(instance_seed)` before each `build_description`, seed recorded as metadata (rubric criterion 8, contamination resistance). The shipped `config.seed` is a decoy wired only to the LLM payload; do NOT rely on it.
- **Independent oracle**, exact 0/1 vs. the deterministic checker (never an LLM judge).
- **Temperature 0**; **repeats = 3** (validate plumbing; determinism means they should largely agree).

---

## 1. Strata design

### Axes (candidate- and repo-appropriate)

Two axes the candidate doc and repo notes explicitly identify as difficulty drivers:

- **Axis 1 — Constraint count** (`n ∈ {3, 4, 5}`). The candidate's stated difficulty dial (`c22.md`: "the constraint count and level are the difficulty dial"). Strict-AND compounds failure per added atom (synthesis Q1/Q5).
- **Axis 2 — Atom-type mix** (`easy-skewed` vs. `mixed`). The single largest score driver in the synthesis (Q4, Q5). Categories per synthesis Q1:
  - **Easy atoms** (per-atom pass ~0.80–0.90): casing, format-wrapper, keyword-presence, start/end-token. IFEval ids e.g. `change_case:*`, `startend:*`, `keywords:existence`, `detectable_format:*`.
  - **Hard atoms** (per-atom pass ~0.55–0.75): exact-word-count-equals-N, forbidden-letter across the whole output. IFEval ids e.g. `length_constraints:number_words` (exact), `keywords:forbidden_words`, `letters` variants.
  - `easy-skewed` = all atoms drawn from the easy pool. `mixed` = ≥1 hard atom included.

This yields a **3×2 = 6-stratum grid**. Honor `INSTRUCTION_CONFLICTS` (`instructions_registry.py:79-176`) when sampling a stack so atoms cannot contradict. Avoid `langdetect`/nltk-`punkt`-dependent atoms in the easy pool where determinism is fragile (repo red flag 3); prefer regex/count-only checkers.

**Deliberately excluded from the baseline grid:** language-detection atoms (nondeterministic edge), and any atom requiring nltk `punkt` unless the runtime download is pinned. These are noise sources that would undercut criterion 5.

### N per stratum and total N

Resolvability requirement (rubric criterion 5): a meaningful prompt change (≥10 points) must exceed residual noise on a 10–20 task internal eval. At temperature 0 with pinned seeds the per-task score is near-deterministic, so residual noise across the 3 repeats should be small; the binding constraint is instead the **binomial resolution of a per-stratum rate estimate**.

- **N = 20 instances per stratum** (× 3 repeats = 60 rollouts/stratum). At an all-pass rate near 0.3–0.7, the standard error of a 20-instance rate estimate is ≈ 0.10–0.11, so a **≥10-point (0.10) effect is at the ~1 SE margin per single stratum** and comfortably resolvable when pooled across the aggregate. 20 satisfies criterion 5's "10–20 task internal eval" band directly.
- **Total N = 120 instances** (6 strata × 20) × 3 repeats = **360 rollouts per (model, prompt)** condition.

**Aggregation** (per the rubric's "aggregation must cross strata" callout): average exact 0/1 first by repeat within task, then across tasks within stratum, then across strata. Report per-stratum rates AND the pooled all-pass rate. Improvements must come from solving more complete instances, never partial credit inside an instance.

> `[thin]` The N=20/stratum choice trades resolvability against cost. If the owner needs a ≥10-pt effect resolvable *within a single stratum* at high confidence, bump to N=40/stratum (see Open decisions). The pooled 120-instance aggregate resolves a 10-pt shift robustly regardless.

---

## 2. The two probe prompts (DRAFTED VERBATIM)

Both prompts receive the same generated instance: a base micro-task line plus the concatenated per-atom constraint descriptions emitted by `build_description`. These operationalize rubric criterion 4 (default scores low / better prompts close the gap) and criterion 9 (floor, ceiling, and reference prompt known in advance).

### (a) Naive prompt — deliberately minimal

```
{BASE_TASK_AND_CONCATENATED_CONSTRAINTS}

Answer:
```

That is the entire prompt: the raw generated instance text, a blank line, and `Answer:`. No restatement of the constraints, no enumeration, no format hygiene, no emphasis, no worked reasoning. This is the "prompt lists only the stated atoms" miss case from `c22.md`'s own example — it operationalizes the criterion-4 expectation that default behavior sits well below ceiling.

### (b) Ceiling prompt — best-effort, states all standard/default conventions of the existing task

```
You must produce a response that satisfies EVERY constraint below. The response
is scored 1 only if all constraints pass a deterministic checker; missing even
one constraint scores 0. There is no partial credit.

Constraints (all must hold simultaneously):
{BASE_TASK_AND_CONCATENATED_CONSTRAINTS}

Follow these conventions of this task exactly:
- Every constraint is stated explicitly above. Nothing is hidden; you do not need
  to infer any unstated rule. Enumerate each constraint and satisfy it directly.
- Output ONLY the answer text itself. Do not add a preamble, explanation,
  restatement, label, quotation marks, or trailing commentary — extra text can
  itself violate a length, word-count, casing, or forbidden-token constraint.
- Do NOT use markdown, bold, bullets, or headers unless a constraint explicitly
  requires them; stray formatting characters count against exact-match checks.
- For any word-count or length constraint, count exactly and match the stated
  number precisely (exact means exact, not "about").
- For any forbidden-letter or forbidden-word constraint, scan your whole answer
  and confirm the letter/word appears nowhere.
- For any casing, start-token, or end-token constraint, verify the first/last
  characters or tokens literally match what is required.
- Before finalizing, silently check each constraint one more time; if any fails,
  revise until all pass. Then output only the final answer.

Answer:
```

This states every standard/default convention of the reseed-only task: strict all-pass semantics, output-only hygiene (the strict-vs-loose false-negative mitigation from synthesis Q4, since c22 inherits IFEval checkers without the loose metric), per-atom-type checking hints, and a self-verify pass. Because **nothing is hidden in the reseed-only baseline** (synthesis Q5), this prompt can in principle enumerate and satisfy all stated atoms — it is the true ceiling probe.

---

## 3. Three-outcome decision rule

Thresholds are anchored to `synthesis-capability.md` Q5 point estimates for the expected default (3–5 mixed atoms, cheap tier, temp 0, strict all-pass): **naive floor central guess ≈ 0.30 (range 0.25–0.40)**, **ceiling central guess ≈ 0.70 (range 0.60–0.80)**. "High" is defined relative to the ceiling-prompt expectation, not 1.0.

Let `naive` and `ceiling` be the pooled all-pass rates under prompts (a) and (b), and let `gap = ceiling − naive`.

| Outcome | Numeric rule (proposed) | Interpretation | Action |
|---|---|---|---|
| **(a) No headroom** | `naive ≈ ceiling` (`gap < 0.10`) AND both `≥ 0.75` | Both prompts near-ceiling; task is trivially solved either way | Proceed to the candidate's **added-complexity design** (Selection/OOD atoms). The existing shape is exhausted. |
| **(b) Headroom, NOT prompt-closable** | `naive ≈ ceiling` (`gap < 0.10`) AND both in the mid/low band (`< 0.55`, esp. `< 0.40`) | Even the ceiling prompt cannot close the gap to high → raw capability deficit under strict stacking, not a prompting problem | Proceed to added-complexity design AND **keep base difficulty shallow** (fewer atoms / easy-skewed strata) so the optimizer signal is not floored. |
| **(c) Prompt-closable headroom** | `naive << ceiling` (`gap ≥ 0.20`) | A single engineered prompt recovers most of the gap — the anti-enumeration mechanisms are absent (synthesis Q5), so the gap is wide-but-shallow | **Direct prompt optimization is viable on the existing shape** for the quick-test purpose. Note the synthesis warning: this gap is likely one-shot-closable, so sustained optimization difficulty may still require added complexity. |

**Threshold justification.** `gap ≥ 0.20` marks "prompt-closable" because the synthesis predicts a naive→ceiling spread of ~0.30–0.45 (0.30→~0.70) that is "real and exploitable" and "most of that gap is closable by a single well-drafted enumerate-the-atoms prompt." The `0.10` band for "≈" matches criterion 5's ≥10-point resolvable-effect floor: a gap below the resolvable-effect size is indistinguishable from noise and should be treated as "no gap." The `0.75` / `0.55` / `0.40` cut points sit at the boundaries of the synthesis Q5 ceiling range (0.60–0.80) and floor range (0.25–0.40): "high" ≈ top of the ceiling band, "mid/low" ≈ at or below the floor band.

**Ambiguous middle** (`0.10 ≤ gap < 0.20`, or both prompts land ~0.55–0.75): treat as a soft (c)-lean but flag for owner review — the synthesis uncertainty is HIGH and asymmetric here (floor could be 0.10–0.50), so a borderline gap should trigger a targeted re-run (e.g. split easy-skewed vs. mixed strata) before committing a branch.

**Read per-stratum too.** The decision may differ by stratum: easy-skewed/3-atom strata may show outcome (a) while mixed/5-atom strata show (b) or (c). If so, the shallow-base recommendation in (b) is directly actionable — pick the strata that hold a residual gap.

---

## 4. Model selection — OPEN DECISION (owner decides)

### What the dossiers measured (strict all-pass / prompt-level, the c22-comparable metric)

| Model | Metric (source) | Result | Tier | Notes |
|---|---|---|---|---|
| GPT-4 | IFEval prompt-strict, 1–3 atoms (`2311.07911`) | **76.89%** | frontier | inst-level 83.57% — shows the all-pass compression |
| PaLM-2-Small | IFEval prompt-strict, 1–3 atoms (`2311.07911`) | **43.07%** | small/cheap | single most transferable cheap-tier datapoint |
| GPT-4-turbo | VFF strict, L1/L2/L3 (`2502.04498`) | **76.3 / 53.3 / 35.3** | frontier | cleanest c22 analog (deterministic, strict AND) |
| GPT-3.5-turbo | VFF strict, L1/L2/L3 (`2502.04498`) | **62.9 / 34.1 / 16.4** | cheap proxy | closest measured proxy for cheap-tier band |
| GPT-4 | COLLIE zero-shot avg (`2307.08689`) | 50.9% | frontier | pass@20 >63% |
| GPT-3.5 | COLLIE pass@20 (`2307.08689`) | 32% | cheap proxy | |
| 7B–8B open (Mistral-7B, LLaMA-2-7B, LLaMA-3-8B) | VFF L1/L3 (`2502.04498`); IFBench pre-RLVR (`2507.02833`) | L1 ≈ 50–65, L3 ≈ 9–16; IFBench 16–31 | small/open | consistent low floor under stacking |
| GPT-4.1 / Claude 3.7·4 Sonnet / Qwen3-32B | IFBench OOD (`2507.02833`) | **<50%** | frontier | OOD atoms — NOT in baseline scope, context only |
| Gemini 2.5 Flash | DeonticBench (different task) (`2604.04443`) | weak | cheap | NOT on stacked-IFEval-atom strict test |
| GPT-5-mini | IF-RewardBench judge τ_b 0.211 (`2603.04738`) | judge-only | cheap | NOT measured as a generator on c22-like tasks |
| DeepSeek-V3 / Claude Haiku 4.5 | — | **not measured anywhere in the 11 dossiers** | cheap | zero direct evidence |

### Proposed candidate set for our baseline (NOT finalized)

The quick-test contract wants a **small, cheap model** (rubric goal: cents per run, minutes). Proposed:

- **Primary:** one cheap-tier hosted model — candidates: `Gemini-2.5-Flash`, `GPT-5-mini`, `Claude Haiku 4.5`, or `DeepSeek-Chat`. This is the decisive band for c22 and, critically, the band with **essentially zero direct evidence** (synthesis Q3/Q5 flag this as the corpus's biggest gap) — so measuring it is the point.
- **Optional second (frontier anchor):** one frontier model (e.g. a GPT-4-class or Claude Sonnet-class model) to bracket the ceiling and confirm the strict-AND compression direction, matching the reviewer demand for multiple base models (positioning §3c item 6).

### Tradeoffs (for the owner)

- **Cheap-tier is where the answer matters** but where we are flying blind — the floor/ceiling estimates (§3) are interpolations, not measurements. Expect the actual numbers to genuinely inform, and possibly surprise.
- **Single cheap model** = cheapest, fastest, matches the quick-test spirit; but a single-model result "won't generalize" (positioning §3c). **Adding a frontier anchor** roughly doubles cost but brackets the range and pre-empts the reviewer objection.
- Model choice interacts with §3 thresholds: a stronger cheap model shifts both `naive` and `ceiling` upward, making outcome (a) more likely; a weaker one pushes toward (b). The thresholds are set for the cheap-tier band and may need per-model adjustment.

**DECISION DEFERRED to project owner.** Do not finalize the model set here.

---

## 5. Cost & wall-clock estimate (per full baseline run)

### Token model per instance

- **Naive prompt input:** base-task + 3–5 concatenated constraint descriptions ≈ **80–200 tokens**; call it **~150 input tokens** average.
- **Ceiling prompt input:** the same instance text + the fixed convention block (~250 tokens) ≈ **~400 input tokens** average.
- **Output:** micro-task, a few words → **~10–30 tokens**; call it **~20 output tokens** (the self-verify step in the ceiling prompt is instructed to be silent, so output stays tiny; if a model emits visible reasoning, cap via `max_tokens ≈ 64`).

### Rollout count per full run

`N (120) × repeats (3) × prompts (2) = 720 rollouts per model.`

- **1 model:** 720 rollouts. **2 models:** 1,440 rollouts.

### Token totals per model (per full run, both prompts)

Per model: 120 × 3 = 360 rollouts per prompt.
- Naive: 360 × (150 in + 20 out) = **54k in + 7.2k out**.
- Ceiling: 360 × (400 in + 20 out) = **144k in + 7.2k out**.
- **Per-model total ≈ 198k input + 14.4k output ≈ 213k tokens.**
- **2 models ≈ 426k tokens total.**

### Dollar estimate `[thin — model prices are illustrative placeholders; owner confirms]`

At a representative cheap-tier rate of ~\$0.15/M input and ~\$0.60/M output:
- Per model: (0.198M × \$0.15) + (0.0144M × \$0.60) ≈ **\$0.03 + \$0.009 ≈ \$0.04 per full baseline run**.
- 2 models (cheap + a costlier frontier at, say, ~\$3/M in, \$15/M out for the frontier): cheap ≈ \$0.04; frontier ≈ (0.198M×\$3)+(0.0144M×\$15) ≈ **\$0.59 + \$0.22 ≈ \$0.81**. **Combined ≈ \$0.85 per full run.**

This satisfies the rubric's "cents per run" goal for the cheap model; a frontier anchor pushes it toward ~\$1.

### Wall-clock

720 rollouts, ~20 output tokens each, no sandbox, no chained calls. At modest concurrency (e.g. 20 in flight) and ~1–2 s/call, **≈ 1–2 minutes per model**, rate-limits permitting (rubric criterion 14). Generator + scoring are local and negligible.

---

## 6. Rubric-mapping table

| Design choice | Serves criteria | How |
|---|---|---|
| Single LLM call + deterministic checker, no chain | 1, 2 | one LLM Call Node → one Eval Node; exact 0/1 by a simple function, no sandbox |
| Micro-task, few-word output, `max_tokens ~64` | 3, 14 | short bounded input/output → cents per run, minutes |
| **Naive vs. ceiling prompts (§2)** | **4, 9** | default scores low; ceiling prompt is the known ~100% reference; the pair operationalizes floor/ceiling/reference-prompt |
| **Two axes × N=20/stratum, 120 total, temp 0, 3 repeats (§1)** | **5** | ≥10-pt effect resolvable on a 10–20 task internal eval; determinism → repeats agree, effect exceeds noise |
| **Fresh seeds per instance, seed as metadata, no published rows (§Fixed)** | **8** | synthetic randomized instances → contamination-resistant; models cannot pre-know answers |
| Reuse IFEval checkers verbatim as independent oracle | 2, 8, 11 | non-tautological ground truth; per-atom "expected X, got Y" is diagnostic |
| Per-atom verdict logged as diagnostic fact | 10, 11 | difficulty decomposes into independent rules; failures reveal which rule was violated |
| Constraint-count + atom-mix strata | 7, 10 | large synthetic pool with known ground truth; latent-rule strata are inspectable |
| Aggregate 0/1 within-repeat → within-task → within-stratum → across strata | (aggregation callout) | averages cross strata; no partial credit inside an instance |
| Strict all-pass 0/1, no partial credit | 2, 4 | exact per-instance rule; incremental gains come from solving more complete instances |
| Three-outcome decision rule with synthesis-anchored thresholds (§3) | 4, 9 | falsifiable go/no-go tied to known floor/ceiling |

Baseline-scope non-coverage (state explicitly, per rubric §"What the quick test does not cover"): this baseline does NOT exercise multi-node routing, sandbox execution, dr-code preprocessing, the second (output-bytes) Objective, or repeat-averaging under real variance (determinism means the 3 repeats mostly agree). Those need their own checks before the full experiment.

---

## 7. Open decisions (awaiting owner)

1. **Model set (§4).** Which cheap-tier model as primary? Include a frontier anchor (doubles cost, brackets ceiling, pre-empts single-model objection) or not?
2. **N per stratum.** Proposed 20 (satisfies criterion 5, cheap). Bump to 40 if a ≥10-pt effect must resolve *within a single stratum* at high confidence?
3. **Atom pool membership.** Confirm the exact IFEval `instruction_id` set assigned to the easy vs. hard pools, and confirm exclusion of language-detection / nltk-`punkt`-dependent atoms (determinism risk). Owner may want a specific curated list.
4. **Selection-condition ambiguity band (§3).** Accept `gap ≥ 0.20` = prompt-closable and `gap < 0.10` = no gap, or tighten/loosen given the HIGH synthesis uncertainty on the floor (0.10–0.50)?
5. **Ceiling-prompt wording (§2b).** Approve the drafted convention block, or trim/extend the hygiene hints? (It inherits IFEval-strict false-negative risk; the hygiene block is the mitigation.)
6. **Loose-metric side report.** Synthesis Q4 notes strict-only costs a few points to formatting false-negatives. Optionally also log per-atom (loose-style) rates alongside strict all-pass, as reviewers will later demand (positioning §3c item 4) — cheap to add now.
7. **Dollar rates.** The §5 prices are illustrative placeholders; confirm actual provider pricing for the chosen models.
8. **Seed-plumbing implementation.** Confirm the glue seeds the module-global `random` (or holds a `random.Random(seed)`) before each `build_description`, and passes explicit kwargs for nonce vocab — the repo's `config.seed` is a decoy (repo red flags).

---

*DRAFT — c22 reseed-only baseline experiment spec. Prepared for project-owner review; no branch committed and no model finalized.*
