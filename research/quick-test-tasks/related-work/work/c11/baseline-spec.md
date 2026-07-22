# DRAFT — Baseline Experiment Spec: c11 House-Convention Canonicalizer

**STATUS: DRAFT. Awaiting project-owner decisions (see §7). Do not run without sign-off.**

## Purpose

Measure **headroom on the existing task shape before building any added complexity.** The
experiment answers one go/no-go question: on plain RFC 8785 (JCS) canonical-JSON — with **no
invented house-rule deviations yet** — is there a below-ceiling gap for a prompt optimizer to
close, and if so, is that gap prompt-closable or a raw capability deficit?

This is deliberately the *plain-JCS* variant. Per the positioning synthesis (§3, §5 below),
plain JCS sits in the crowded "public-standard-conformance" cell, not c11's differentiating
"invented-adversarial" cell, so a large gain here is a weaker research result. But it is the
**cheapest, lowest-risk first measurement** — it tells us whether the *task shape* has
headroom at all before we spend effort inventing and validating house rules. The three-outcome
decision rule in §3 routes us from this measurement to the next design step.

### Fixed by prior decisions (not open here)

- Instances: plain RFC 8785 (JCS) canonical JSON, **no invented deviations yet**.
- Inputs: sampled from pinned json-schema-faker schemas with **fresh seeds** (never published
  instances — contamination guard, rubric criterion 8). json-schema-faker is run-verified:
  519/519 tests green, byte-deterministic per seed (repo note).
- Oracle: **trailofbits rfc8785-py** reference implementation, used unmodified (`rfc8785.dumps`).
  It is a pure function of the input Python object; determinism total (repo note). Independent
  of the generator: our generator emits ordinary Python dicts/lists/scalars; the oracle computes
  the canonical form from RFC rules, not from how the input was built (repo note, "Oracle
  independence — Strong").
- Output: the canonical JSON string. Scoring: **0/1 whole-string exact match** vs the oracle.
- Decode: **temperature 0**. Repeats: **3** distinct `repeat_id`s (plumbing validation; with
  determinism they should largely agree — rubric criterion 5).
- One trivial input normalization is permitted before the exact match: strip surrounding
  whitespace / code fences (mirrors the candidate's pre-freeze step 5). Nothing else.

> **Pin before running.** json-schema-faker version (repo note reviewed v0.6.2), rfc8785-py
> version, the schema set (§1), the seed list per stratum, and the split identities
> (calibration / internal-eval / official / held-out). Avoid `format: "date"`/`"date-time"`
> or pin `minDateTime`+`maxDateTime` — the one wall-clock nondeterminism source in the
> generator (repo note).

---

## 1. Strata design

### Axis choice (justified from candidate doc + synthesis)

The candidate design makes each latent rule own a stratum so pool accuracy climbs rule by rule
(candidate "What it proposes"). For the *plain-JCS* baseline the "rules" are the RFC 8785
sub-rules themselves, and the synthesis identifies exactly which ones are the break points.
The capability synthesis (Q1, Q5) is unambiguous: **structural JSON validity is near-saturated;
the failures live in the canonical *details*** — key sort order (UTF-16 code-unit), number
canonicalization (ECMAScript shortest-round-trip — flagged as "the single hardest sub-rule"),
exact escaping (\u), and zero-insignificant-whitespace. Schema *content* swings the score more
than model choice (Q5, repeated twice).

So the stratum axis is **which JCS sub-rule the instance stresses**, realized as schema-content
shape. Five strata:

| # | Stratum | What the schema forces | Synthesis basis for difficulty |
|---|---------|------------------------|--------------------------------|
| S1 | Flat / shallow | Few keys, all-ASCII strings, ints only, no sort tension | Q5: "floor rises toward 30%+" for shallow; near-saturated structure |
| S2 | Key-sort stress | Many keys whose insertion order != UTF-16 sorted order; nested objects | Q1/Q5: key sort by UTF-16 code unit rarely reproduced exactly |
| S3 | Number-canonicalization | Floats needing ECMA shortest round-trip; large ints; -0; exponent forms | Q5: "the single hardest sub-rule"; number formatting is the break point |
| S4 | Unicode / escaping | Strings needing \u escapes, control chars, chars where escape policy is exact | Q5: unicode requiring \u escapes drops the floor toward 5% |
| S5 | Mixed / deep | Nested combination of S2+S3+S4 in one instance | Q1: whole-string exact match compounds every sub-rule slip |

Every instance is generated adversarially so it actually exercises its stratum's sub-rule
(candidate step 3: reject inputs where the messy and canonical forms coincide — e.g. reject S2
instances whose keys are already sorted). S1 is the explicit first-rung stratum.

### N per stratum and total N — justified by rubric criterion 5

Criterion 5 requires that a meaningful prompt change (**≥10 points**) resolves above residual
noise on a **10–20 task internal eval**. That is the binding constraint on the *internal-eval*
slice, not the whole pool. Design:

- **Internal-eval slice: ≥2 tasks per stratum** (candidate step 3 requires ≥2/stratum in every
  internal subset) × 5 strata = **10 tasks minimum, 20 tasks target** per internal eval. This
  is exactly the 10–20 band criterion 5 names. At temperature 0 with a pinned split, a 10-point
  effect = ≥1 flipped task out of 10 (≥2 out of 20), which is resolvable given the near-
  determinism the rubric assumes (criterion 5: "repeats should largely agree").
- **Official / calibration pool: 40 per stratum × 5 = 200 instances.** Rationale: the strata
  must be averageable first by repeat within task, then across tasks and strata (rubric
  "Aggregation must cross strata" callout), and a per-stratum accuracy needs enough instances
  that a per-stratum rate is stable to a few points — 40/stratum gives a binomial standard
  error of ≈±8pp at p=0.5, tightening as strata saturate or floor out. 200 total keeps a full
  baseline run in the cents-per-run / minutes envelope (§5) while being a "hundreds of tasks"
  pool (rubric criterion 7).
- **Held-out fresh-seed split: a further 200** (disjoint seeds), reserved so any later
  optimizer claim is verified on seeds never seen — contamination/leakage guard for the
  eventual comparison (positioning §3c.4). Not scored in the baseline itself; generated and
  frozen now so the split identity is pinned before any optimization touches the task.

**Totals:** 200 official + 200 held-out = 400 pinned instances; a 10–20 task internal-eval
slice is drawn disjointly (or as a labeled subset) from the official 200. Seeds recorded as
metadata per instance (criterion 5).

> Note on the degenerate second objective: output-bytes is **provably degenerate** here (correct
> canonical strings have fixed length), so this baseline validates single-objective optimization
> plus two-objective plumbing only. It must say so (rubric GEPA callout; candidate risk 3). Not
> measured as a real objective in this spec.

---

## 2. The two probe prompts (DRAFTED VERBATIM)

These operationalize rubric criterion 4 (default prompts score low; better prompts close the
gap) and criterion 9 (ceiling, floor, reference prompt known in advance). Both are prompt-only
(no constrained decoding — that is a decoding-engine intervention unavailable in the baseline,
synthesis Q4.2). The model receives the messy input JSON and must return only the canonical
string.

### (a) Naive probe — deliberately underspecified

```
Canonicalize this JSON.

{input}
```

Rationale: this is the "single instruction, no worked examples" floor the synthesis Q5 predicts
at ~5–30% (best guess ~15%). It names the operation but states none of the JCS sub-rules. It is
the operationalization of criterion 4's "models' default behavior sits well below the ceiling."

### (b) Ceiling probe — best-effort, states all standard/default conventions

```
Convert the JSON below into its RFC 8785 (JCS) canonical form and return ONLY the
resulting string, with no code fences, no commentary, and no trailing newline.

Apply every one of these rules exactly:

1. Object keys are sorted by their UTF-16 code units (compare the raw UTF-16 code-unit
   sequences of the key strings, ascending). Sort keys at every nesting level.
2. No insignificant whitespace: no spaces or newlines anywhere except inside string
   values. `{"a":1,"b":[2,3]}`, never `{ "a": 1 }`.
3. Strings use the shortest JSON escaping: escape only " \ and the control characters
   U+0000..U+001F. Use the two-character forms \" \\ \b \f \n \r \t where they exist;
   otherwise \u00XX (lowercase hex). Do not escape forward slash or any other character.
4. Numbers use the ECMAScript Number-to-string (shortest round-trip) form: integers with
   no decimal point or exponent; no leading zeros; no "+" on exponents; "-0" becomes "0";
   the minimal digit sequence that round-trips to the same IEEE-754 double.
5. Literals are exactly true, false, null. Arrays preserve their element order.

Worked examples:

Input:  {"b": 2, "a": 1}
Output: {"a":1,"b":2}

Input:  {"x": 1.0, "y": 1e2, "z": -0}
Output: {"x":1,"y":100,"z":0}

Input:  {"s": "line1\nline2", "t": "π"}
Output: {"s":"line1\nline2","t":"π"}

Input:  {"nested": {"d": 4, "c": 3}, "arr": [true, null]}
Output: {"arr":[true,null],"nested":{"c":3,"d":4}}

Now canonicalize:

{input}
```

Rationale: this states every standard/default convention of the *existing* task and shows worked
examples spanning sort / number / escape / whitespace — the "several worked examples" the
synthesis Q5 ceiling prediction (~40–75%, best ~55%) is conditioned on. It is the "best feasible
prompt" and gives criterion 9 its known ceiling. The four worked examples are illustrative
covering, not full-table coverage (no table exists in plain JCS), so no leakage concern applies
to the baseline; the leakage guard becomes live only for the eventual invented-rule task.

> **Verify the ceiling-probe examples against the real oracle before freezing.** The Output
> strings above are hand-written and MUST be regenerated by `rfc8785.dumps` on the exact inputs
> and pasted verbatim — do not trust the hand-typed forms. (The π/escape example in particular:
> confirm whether the oracle emits the literal UTF-8 char or a \u escape, and match it.)

---

## 3. Three-outcome decision rule

Thresholds are inferences from the capability synthesis (Q5), which gives a predicted naive
floor of **~5–30% (best ~15%)** and a predicted ceiling-probe score of **~40–75% (best ~55%)**,
with an explicit upside risk that plain JCS is *memorizable public knowledge* and the naive
floor could exceed 40%. Bands below are chosen relative to those predictions; all are DRAFT and
should be re-pinned once the calibration numbers land. "≈" means within ~10 points (the
criterion-5 resolution grain). Scores are official-pool exact-match, averaged per repeat then
across strata.

Let **F** = naive-probe score, **C** = ceiling-probe score. Define **HIGH ≥ ~80%**,
**MID ≈ ~40–75%**, **LOW ≤ ~30%**.

| Outcome | Condition | Interpretation | Next step |
|---------|-----------|----------------|-----------|
| **(a) No headroom** | F ≈ C ≈ HIGH | Plain JCS is too easy / memorized (the Q5 "JSON is saturated" upside risk realized). Any reasonable prompt hits the ceiling; every optimizer would look identical (rubric criterion 4 fails). | **Proceed to the candidate's added-complexity design** (invented house-rule deviations, post-diet ≤6 single-pass rules). The task shape has no discriminative gap without them. |
| **(b) Headroom, not prompt-closable** | F ≈ C ≈ MID or LOW | The ceiling prompt states every rule and still can't lift the score — a **raw capability deficit** (the synthesis' persistent Wrong-Value / number-canonicalization slip that "does not go away with prompting", Q3/Q5). | **Added-complexity design AND keep base difficulty shallow.** If the model can't execute plain JCS even when told every rule, invented deviations layered on top will floor the score. Bias strata toward S1/S2, drop or cap S3 number-heavy content. |
| **(c) Prompt-closable headroom** | F ≪ C (gap ≥ ~20 points), C not near ceiling-saturated | Naive is low, stating the rules lifts it substantially — the gap is **prompt-shaped**. This is the criterion-4 ideal: room to improve, and prompts close it. | **Direct prompt optimization is viable on the existing shape.** Run COPRO/MIPROv2/GEPA on plain JCS first; added complexity is optional, not required (synthesis Q5 go/no-go: "added complexity would be optional, not required"). |

Notes:
- The F≪C gap threshold of **~20 points** is set at 2× the criterion-5 resolution grain (10
  points), so a "prompt-closable" verdict clears noise with margin.
- Outcome (a) is the synthesis' *primary predicted risk* for plain JCS (Q5, positioning §5): run
  the naive floor specifically on the **number-heavy (S3) and unicode-heavy (S4) strata**, since
  the corpus says that is where breakage — and therefore discriminative signal — lives. If even
  S3/S4 come back HIGH, outcome (a) is firm.
- These thresholds are the *falsifiable success criterion* rubric criterion 9 demands; they must
  be frozen from the *measured* calibration numbers (candidate step 5: "derive floor/gap targets
  from measured numbers"), not from the predictions, before any optimizer run.

---

## 4. Model selection — OPEN DECISION (owner decides)

The synthesis is explicit: **the cheap tier c11 actually runs at is not directly tested by any
paper** (Q3). Everything below is adjacent-proxy evidence. This table is for the owner to choose
from; it is **not finalized here.**

### Models tested by the related-work papers, with results (from the dossiers)

| Model (tier) | Paper / dossier | Metric reported | Result | c11 relevance |
|--------------|-----------------|-----------------|--------|---------------|
| GPT-4o (frontier closed) | LLMStructBench 2602.14743 | DOC_micro (all-or-nothing, closest proxy to exact match) | **0.52** (top tier) | The DOC_micro ceiling the synthesis anchors c11's ceiling on |
| Gemma3-27B (open) | LLMStructBench | DOC_micro | **0.52** (tied GPT-4o) | Frontier open matches closed on all-or-nothing |
| GPT-4o | StructEval 2505.20139 | avg over 44 tasks (graded) | **76.02%** (best) | JSON/CSV/YAML gen "saturated" >90% for it |
| GPT-4o-mini (cheap closed) | StructEval | avg | **73.19%** (~3pp behind 4o) | Closest "-mini" proxy to c11 cheap tier |
| GPT-4.1-mini (cheap closed) | StructEval | avg | **75.64%** (best on T-gen 92.57) | Cheap-tier proxy, strong on generation |
| o1-mini (reasoning) | StructEval | avg | **75.58%** | Reasoning gave no decisive edge over 4o here |
| GPT-4o-mini | PolyNorm 2511.03080 | WER (lower better) | **6.60–11.19%** (vs 4o 4.17–7.88) | Meaningfully worse than 4o but beats rule baseline |
| GPT-4.1 / Claude 3.7/4 Sonnet / Qwen3-32B (frontier) | IFBench 2507.02833 | novel-constraint accuracy | **<50%** one-shot | The most c11-relevant SHOWN result: novel rules crater accuracy even at frontier |
| o3 (frontier reasoning) | IFBench | novel-constraint accuracy | only OOTB model IF-RLVR can't beat | Reasoning helps most on novel constraints |
| Deepseek-R1-1.5B (tiny open) | LLMStructBench | DOC_micro | **0.16** (strat P) → recovers under schema-forcing | Prompting strategy beats scale at this tier |
| Gemma3-1B / Phi3-3.8B (tiny open) | LLMStructBench | DOC_micro | **0.10 / 0.01** (strat P) | Floor of the cheap/open tier |
| Qwen3-4B (open small) | StructEval | avg | **67.04%** (best open-source) | Small-open structured-output proxy |
| Llama-3.1-8B / Phi-3-mini (open) | StructEval | TOML gen | **6.76 / 0.00** | Format-collapse worst at cheap/open tier |

### Proposed candidate set for OUR baseline (NOT finalized)

A cheap tier that the quick-test economics require (rubric criteria 3, 14: cents/run, minutes),
spanning ≥2 small models so the eventual result isn't single-model under-powered (positioning
§3c.3). Candidate slate to choose ≥2 from:

- **Gemini-2.5-Flash** — cheap frontier-family, fast, high concurrency.
- **GPT-5-mini** — "-mini" closed tier the StructEval/PolyNorm proxies most resemble.
- **Claude Haiku 4.5** — cheap closed, strong instruction-following.
- **DeepSeek-Chat** — cheap, and the DeepSeek family is the one with published tiny-model
  structured-output numbers (proxy anchor).

> If any Claude/Anthropic model is on the final slate, confirm exact model IDs and pricing
> against the `claude-api` skill before finalizing §5 cost — do not rely on memory.

### Tradeoffs (for the owner)

- **One model** minimizes §5 cost and is enough to answer the headroom go/no-go, but produces a
  result reviewers flag as under-powered (positioning §3c.3 — field norm is 12–22 models).
- **≥2 small models** costs ~2× but lets the eventual optimizer claim argue generality across the
  cheap tier; the synthesis (Q3) warns cheap-tier scores swing widely, so a single model's
  number may not generalize.
- **Include one reasoning model?** The synthesis (Q3) says reasoning gave no decisive edge on
  structured-output proxies but *did* help most on the novel-constraint axis (IFBench o3). Plain
  JCS is not novel, so a reasoning model is likely unnecessary for *this* baseline but may matter
  for the eventual invented-rule task — the owner may want one data point now.
- **Model choice matters less than schema content here** (synthesis Q5, twice: "schema content
  will swing the score more than model choice"). This argues for spending the budget on strata
  coverage over model count, if forced to trade.

---

## 5. Cost & wall-clock estimate per full baseline run

**Per-instance token estimate** (order-of-magnitude; instances are short by design, rubric
criterion 3):

- Input JSON instance: ~50–300 tokens depending on stratum (S1 shallow ~50, S5 deep ~300).
- Naive-probe prompt overhead: ~10 tokens. Ceiling-probe overhead (rules + 4 worked examples):
  ~450 tokens (one-time per call, dominates the naive case).
- Output canonical string: ~50–300 tokens (roughly the input size).
- **Blended: naive ~ 350 tokens/call, ceiling ~ 800 tokens/call. Use ~600 tokens/call blended.**

**Call count per full baseline run:**

```
calls = instances(200 official) × probes(2) × models(M) × repeats(3)
```

- M = 1: 200 × 2 × 1 × 3 = **1,200 calls** ≈ 1,200 × 600 = **~720K tokens**.
- M = 2: **2,400 calls** ≈ **~1.44M tokens**.
- M = 4: **4,800 calls** ≈ **~2.88M tokens**.

(The 200 held-out instances are generated once and frozen, not called in the baseline, so they
add no per-run token cost.)

**Dollar cost** (illustrative — cheap-tier blended ~\$0.5–\$1.5 per 1M tokens combined
input+output; **confirm live pricing before running**, via `claude-api` for any Anthropic model):

- M = 2 (~1.44M tokens): **~\$1–\$2 per full baseline run.**
- M = 4 (~2.88M tokens): **~\$2–\$4 per full baseline run.**

This lands squarely in the rubric's "cents-to-a-few-dollars per run" envelope (criteria 3, 14).

**Wall-clock:** at a modest sustained ~5 calls/sec (temp 0, short outputs, no sandbox, no rate-
limit bottleneck at cheap-tier concurrency — rubric criterion 14): M=2's 2,400 calls ≈ **~8
minutes**; M=4's 4,800 calls ≈ **~16 minutes**. "Minutes, not hours" (criterion 14) holds. Add
generation time: json-schema-faker produced its full suite in ~870ms (repo note), so generating
400 instances is sub-second and negligible.

---

## 6. Rubric-mapping table

Every design choice → the criteria it serves (criteria 4, 5, 8, 9 emphasized per the task).

| Design choice (this spec) | Rubric criteria served |
|---------------------------|------------------------|
| One LLM call → one exact-match eval; no chain | 1, 2 |
| 0/1 whole-string exact match vs oracle; no partial credit | 2, "Aggregation" callout |
| Short inputs, bounded outputs (~50–300 tok) | 3, 14 |
| **Naive vs ceiling probes designed so default scores low and rule-stated prompt lifts it** | **4**, 10 |
| Temperature 0; pinned seeds recorded as metadata; 3 repeats | **5** |
| Internal-eval slice = 10–20 tasks, ≥2/stratum, ≥10-pt effect resolvable | **5** |
| Fresh seeds only, never published instances; held-out fresh-seed split frozen | **8** |
| Synthetic parameterized pool (json-schema-faker), 200 official + 200 held-out, known ground truth | 7, **8** |
| **Ceiling probe = known-in-advance ceiling; naive = known floor; §3 thresholds = falsifiable criterion** | **9** |
| Frozen thresholds derived from *measured* calibration, not predictions | **9** |
| 5 strata by JCS sub-rule (sort/number/escape/mixed), each a latent-rule stratum | 10, "Aggregation crosses strata" callout |
| Exact-match diff names which sub-rule slipped (expected X, got Y) | 11 |
| Independent oracle (rfc8785-py), pure function, generator-independent | 2, 8 (no tautology) |
| Output-bytes objective declared degenerate here | GEPA callout; candidate risk 3 |
| 3 repeats validate aggregation plumbing (determinism ⇒ largely agree) | 5, 13 |

---

## 7. Open decisions (awaiting owner)

1. **Model slate (§4).** Which models, and how many? Recommendation to *choose from*:
   Gemini-2.5-Flash / GPT-5-mini / Claude Haiku 4.5 / DeepSeek-Chat; ≥2 for generality vs 1 for
   cost. Include a reasoning model now or defer to the invented-rule task? **Owner decides; drives
   §5 cost directly.**
2. **Decision-rule thresholds (§3).** HIGH/MID/LOW bands and the ~20-point "prompt-closable" gap
   are DRAFT inferences from the synthesis. Confirm or re-pin after calibration numbers land.
3. **N per stratum (§1).** 40 official / 40 held-out per stratum (200/200 total) proposed. Owner
   may want more official-pool instances if per-stratum error bars (~±8pp) are too loose for the
   go/no-go, or fewer to cut cost.
4. **Internal-eval slice size.** 10 (min) vs 20 (target) tasks — criterion 5 permits either.
   Owner sets whether it's a disjoint split or a labeled subset of the official 200.
5. **Ceiling-probe example set.** Four worked examples proposed; owner may add/remove. Regardless,
   **all example outputs must be regenerated by the real oracle and pasted verbatim before
   freezing** (§2 note).
6. **Schema pin (§1).** Exact json-schema-faker schemas per stratum are not yet written (they are
   ~30–60 LOC new config, repo note). Owner approves the schema set and confirms date formats are
   excluded or `minDateTime`/`maxDateTime` pinned.
7. **Version pins.** json-schema-faker version (v0.6.2 reviewed), rfc8785-py version, and whether
   the oracle is used strictly unmodified for the plain-JCS baseline (it is — no house rules yet).
8. **What "no headroom" (§3 outcome a) triggers.** If plain JCS is HIGH, we jump to the invented-
   rule design. Owner confirms that branch is in scope / budgeted before we run, so the result is
   actionable.

---

## Evidence-thinness disclosures (required)

- **No paper runs c11's exact task** (byte-exact canonical-JSON vs a deterministic oracle);
  all 7 dossiers are partial-credit adjacent proxies. Every floor/ceiling number in §3 and §5's
  token blend is an **inference with wide error bars** (capability synthesis, up-front caveat).
- The **cheap tier c11 runs at is not directly tested** by any paper (§4); the model table is
  proxy evidence, not measurements of our models on our task.
- **Serialization-convention ablation — the exact axis c11 varies — has zero measured effect
  size** anywhere in the corpus (synthesis Q4). The §1 stratum-difficulty ordering is inferred
  from adjacent break-point evidence (number formatting, novel conventions), not measured.
- The **naive-floor prediction pulls two ways**: IFBench's novel-constraint evidence argues for a
  LOW floor, but plain JCS is a memorizable public standard, so the floor could exceed 40%
  (synthesis Q5). This is *the* uncertainty the baseline exists to resolve — do not treat either
  direction as settled going in.
