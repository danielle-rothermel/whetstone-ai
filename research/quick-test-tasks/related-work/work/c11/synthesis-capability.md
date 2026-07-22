# Capability Synthesis — c11 (House-Convention Canonicalizer)

Scope: 7 verified dossiers. Task family = "produce one structured / normalized output
string from a given input under a fixed convention, scored by a verifier." c11's
reseed-only baseline is specifically: RFC 8785 (JCS) canonical-JSON, inputs from pinned
json-schema-faker schemas with fresh seeds, oracle = trailofbits rfc8785-py, output = the
canonical JSON string, **exact match**.

Dossier keys used throughout:
- `2404.03868` — EDC (Extract-Define-Canonicalize, KG canonicalization)
- `2501.10868` — JSONSchemaBench (constrained decoding, schema-valid JSON)
- `2505.20139` — StructEval (structured-output generation, 18 formats)
- `2507.02833` — IFBench (verifiable instruction following, 58 constraints)
- `2507.21340` — StructText (synthetic table-to-text + extraction)
- `2511.03080` — PolyNorm (few-shot LLM text normalization for TTS)
- `2602.14743` — LLMStructBench (NL-message-to-JSON extraction, 22 open models)

IMPORTANT CAVEAT UP FRONT (evidence, not inference): **None of the 7 papers runs c11's
exact task** — canonical-JSON serialization scored by whole-string exact match against a
deterministic oracle, under invented standard-deviating rules. Every dossier's
`relation_to_candidate` says so explicitly. The closest structural cousins score with
**partial-credit** verifiers (F1/WER/BLEU/keyword-match/schema-validity), and none scores
byte-exact canonical serialization. So every number below is an **adjacent-task proxy**,
and the c11-specific predictions in Q5 are inferences with wide error bars.

---

## Q1. One forward pass: what CAN and CANNOT models do, and where does one-shot collapse?

### What the evidence SHOWS

**Structural JSON validity in one pass is nearly solved; per-field value correctness is not.**
- `2602.14743` (LLMStructBench, Sec. V-VII, Table V): single-shot, one example pair,
  schema in context. F1_micro (fuzzy value/token accuracy) is **flat and high, 0.89–0.96**
  across all 22 models 0.6B–70B. But DOC_micro (document-level *full* validity, the
  all-or-nothing metric closest to c11) is **volatile, 0.01–0.52**. Even the top tier
  (GPT-4o, Gemma3-27B) tops out at **DOC_micro 0.52** — i.e. ~half of documents are
  fully correct end-to-end (Table V). Wrong-Value (WV) errors are the dominant, persistent
  failure at *every* scale, 96–100% of remaining errors even for the best models (Sec. VII).
- `2501.10868` (JSONSchemaBench, Sec. 5–6): single greedy/temp-0 generation. On *schema
  validity* (a broad accept class, easier than exact match), even the best constrained
  frameworks hit real ceilings well below 100% on hard real-world schemas (theoretical
  coverage 0.13–0.54 on the hardest datasets, Table 12). LM-only (no constrained decoding)
  has the *lowest* compliance rate — "unreliability as a standalone solution" (Sec. 5.2).
- `2505.20139` (StructEval, Table 6/8): zero-shot single pass. Best model GPT-4o averages
  **76.02%** across 44 tasks; best open-source (Qwen3-4B) 67.04%. But format matters
  enormously: JSON/CSV/YAML generation is **"saturated" >90–100%** for strong models
  (GPT-4o T→CSV=100, T→JSON=99.36, Table 8), while rare formats (TOML) collapse to
  single digits / 0.00 for weaker open models.

**Difficulty/collapse axes that the evidence pins down:**
- *Value/content correctness* is the persistent one-pass bottleneck, not structure
  (`2602.14743` Sec. VII; `2505.20139` saturation split).
- *Format familiarity*: common formats (JSON) saturate near-ceiling one-shot; uncommon
  formats collapse (`2505.20139` TOML 0.00–16 for open models, Table 8).
- *Unseen/held-out constraints*: `2507.02833` (IFBench, Sec. 1) — frontier models
  (GPT-4.1, Claude 3.7/4 Sonnet, Qwen3-32B) score **below 50%** on 58 *novel* verifiable
  constraints in one pass, despite 80%+ on the familiar 25-constraint IFEval. This is the
  single most c11-relevant SHOWN result: **novel, standard-deviating rules crater one-shot
  accuracy relative to familiar ones**, and this holds even for frontier models.

### What I INFER
For c11's *plain JCS* reseed baseline specifically: JSON serialization is the "saturated,
familiar" format (good for one-shot structure), but the *canonicalization rules* of RFC
8785 (deterministic key sort by UTF-16 code unit, minimal number formatting per ECMAScript
`Number.prototype.toString`, no insignificant whitespace, specific escaping) function like
the "novel constraint / rare-format" axis that IFBench and StructEval show collapses
one-shot accuracy. Because c11 uses **whole-string exact match** (harder than DOC_micro's
"valid document" and far harder than F1_micro), the relevant proxy ceiling is the
DOC_micro-style all-or-nothing metric (≤0.52 top-tier) *further depressed* by exact-match
strictness and by unfamiliarity with exact JCS number/escape rules.

---

## Q2. Multistep / interactive / many-shot: what breaks, does more help?

### What the evidence SHOWS
**No paper tests a per-instance interactive repair loop with oracle feedback.** Every
dossier's `one_shot_vs_multistep` confirms this — "multi-turn" in IFBench means
constraint-isolation across a fixed 3-turn context, not scored retry. So there is **no
direct evidence** that interactive correction helps or hurts in this family.

What IS shown about "more context / more demonstrations":
- **Pipeline iteration helps but saturates fast.** `2404.03868` (EDC, Sec. 4.2.1, App.
  B.2): a single self-refinement pass (EDC+R) "consistently and significantly" improves
  canonicalization F1 for all models; a *second* pass (EDC+2xR) yields near-zero
  additional gain (WebNLG 0.794→0.797) — "diminishing returns." Refinement here is
  self-bootstrapped (own prior output + retrieved schema), **not** oracle feedback.
- **Weaker models exploit added context less.** `2404.03868` Sec. 4.2.1: Mistral-7b
  benefits less from the refinement hint than GPT-3.5/GPT-4 — "not as able to leverage the
  provided hints." Directly relevant to the cheap tier.
- **Demonstration/example *curation* is a large swing.** `2511.03080` (PolyNorm, App. D):
  offline hillclimbing of the ICL example set moved GPT-4o WER dramatically (Japanese
  12.32%→7.88%, Lithuanian 12.22%→6.99%). This is author-side prompt curation between
  runs, not model-side interaction — but it shows demo *quality/coverage* dominates.
- **Multi-constraint / regime-mixed *training* helps generalization** (`2507.02833`, Table
  7): models trained on a mix of single+multi-turn generalize to both better than
  single-regime. But this is RLVR training, out of scope for c11's prompt-only baseline.
- **Richer in-prompt guidance can trade one failure mode for another** (`2602.14743`,
  Sec. VI-C): switching Deepseek-R1 from strategy P to PJ+ eliminates schema/Missing-Key
  failures but "redirects almost every remaining uncertainty into field content" — turns
  some previously-perfect parses into Wrong-Value. More enforcement ≠ more exact matches.

### What I INFER
For c11: adding demonstrations should raise the floor substantially (the JCS rules are
learnable-by-example, as PolyNorm shows for normalization), but the c11 design's own
concern about demo-coverage-vs-table-width leakage is echoed by the evidence: with enough
examples covering all rule facets, the task risks becoming copy-from-context. There is
**no evidence** that an interactive/agentic loop is needed or tested; EDC's diminishing
returns suggest a single good pass captures most achievable accuracy. Cheap-tier models
will benefit *less* from extra context (EDC's Mistral finding), so many-shot gains at the
cheap tier will be smaller than at frontier.

---

## Q3. Model-tier breakdown (frontier reasoning vs. cheap tier)

The **cheap tier c11 runs at** (Gemini-2.5-Flash / GPT-5-mini / Claude Haiku 4.5 /
DeepSeek-Chat class) is not directly tested by any paper. The closest proxies are the
~1B–14B open-weight models and the "-mini" closed models the dossiers do report.

### Frontier / reasoning tier (SHOWN)
- `2507.02833`: o3 is the only out-of-the-box model IF-RLVR models can't beat; frontier
  non-reasoning (GPT-4.1, Claude 3.7/4 Sonnet, Qwen3-32B) still **<50%** on novel
  constraints (Sec. 1).
- `2505.20139`: reasoning model o1-mini = 75.58 avg; GPT-4o = 76.02 (the top). Reasoning
  gave no decisive edge over GPT-4o here.
- `2602.14743`: GPT-4o (closed reference) = DOC_micro 0.52, **tied** with open Gemma3-27B;
  "did not exhibit a clear advantage over the best open-source models" (Sec. VII).

### Cheap-tier proxies (SHOWN)
- **Small models fail structurally under weak prompting but recover under schema
  enforcement.** `2602.14743`: Deepseek-R1-1.5B under strategy P has 36.8% Missing-Key
  errors and DOC_micro 0.16; Gemma3-1B DOC_micro 0.10; Phi3-3.8B DOC_micro 0.01 under P
  but recovers to a usable range under PJ+ (Table V, VIII). **Prompting strategy dominates
  model scale** at this tier (Abstract).
- **"-mini" closed models:** `2505.20139` GPT-4o-mini avg 73.19 (vs GPT-4o 76.02) —
  only ~3 pts behind on graded structured-output; GPT-4.1-mini avg 75.64, actually best
  on T-gen (92.57). `2511.03080` GPT-4o-mini WER 6.60–11.19% vs GPT-4o 4.17–7.88% — mini
  is meaningfully worse but still beats the rule-based baseline on 7/8 languages.
- **Small-model value accuracy is high but full-document correctness is low.**
  `2602.14743`: even 0.6–1.7B models reach F1_micro 0.92–0.94 but DOC_micro 0.10–0.40
  (Table V) — they get most fields right, rarely get the *whole* document right.
- **Format collapse is worst at the cheap/open tier.** `2505.20139` Table 8: open models
  score 0.00–16 on TOML generation (Phi-3-mini 0.00, Llama-3.1-8B 6.76) vs commercial
  75.78% avg on StructEval-T generation.

### INFERENCE for the cheap tier
c11's exact-match metric is a DOC_micro-style all-or-nothing measure, and the cheap tier's
DOC_micro proxies (~0.01–0.44) are exactly where the evidence shows the widest, lowest
spread. The cheap tier will get individual JCS rules "mostly right" (high F1_micro analog)
but will rarely produce a byte-exact whole string (low DOC_micro analog), *and* the JCS
number/escape/sort rules are the unfamiliar-convention axis that depresses even frontier
models below 50% (IFBench). So the cheap-tier one-shot exact-match floor for c11 should be
**low**.

---

## Q4. Prompting strategies / input representations that moved accuracy (with conditions)

### SHOWN, with magnitudes:
1. **In-prompt schema + one example vs. API format-flag only** (`2602.14743`, Table III/V,
   the biggest clean swing): across all 22 models, strategy choice moved total failures
   from **1,597 (P) to 18 (PJ+)** and DOC_micro from **0.01 to 0.52** for
   comparable models. Strategy P (in-prompt schema+example, no format param) won for 11/22
   models; PJ+ (in-prompt schema+example + full JSON-Schema object as API format) won for
   8/22 and was the "safest for parseable output," especially for small/unreliable models
   — but at the cost of *more* Wrong-Value errors. **Net: schema+example in the prompt is
   the dominant lever; forcing the API format object trades structural for semantic errors.**
2. **Constrained decoding vs. free generation** (`2501.10868`, Table 8): constrained
   decoding (Guidance) beat LM-only by **~3% on every quality task** (GSM8K 80.1→83.8).
   But constraints can push the model off-distribution (blocking the comma in "89,000"
   could yield "890000"), a mechanism that could *hurt* exact-match. Constrained decoding
   is a *decoding-engine* intervention, not available in c11's plain prompt-only baseline.
3. **Curated few-shot ICL example set** (`2511.03080`, App. D): refining which examples
   are shown moved GPT-4o WER by 4–5 absolute points per language (Japanese 12.32→7.88).
   Example *coverage/quality*, not prompt phrasing, was the swing factor ("unified prompt
   format across languages, varying only the localized examples," Sec. 3.1).
4. **Self-refinement pass** (`2404.03868`, Table 1/6): one EDC+R pass adds meaningful F1
   (WebNLG 0.746→0.794); removing the schema-retriever hint drops it back (0.794→0.752).
   Second pass negligible.
5. **Output-representation constraint direction** (`2501.10868`, Sec. 6): forcing a
   canonical/simplified output (bare integer, single letter) *helped* accuracy vs free
   text here — contradicting the "format restrictions hurt" prior (Tam et al. 2024). But
   `2505.20139` and `2602.14743` show forcing structure can *introduce* new errors. **The
   evidence conflicts on whether tighter output constraints help or hurt** — it depends on
   whether the constraint aligns with or fights the model's natural output.

### NOT shown (evidence gap): No paper does a controlled ablation of *serialization
convention* (key order, quoting, whitespace, number format) — the exact axis c11 varies.
Every dossier's `representations` field states this explicitly. So c11's specific lever
(house-rule representation) has **zero** direct measured effect size in this corpus.

---

## Q5. Predicted naive-prompt floor and ceiling for c11's plain-JCS reseed baseline
(cheap tier, temp 0, exact match)

This is INFERENCE built on adjacent proxies. Reasoning chain and uncertainty are explicit.

**Anchors (SHOWN):**
- Cheap/small-tier DOC_micro (all-or-nothing, single example, schema in prompt):
  **0.01–0.52**, with cheap tier clustering **0.10–0.44** (`2602.14743` Table V).
- Novel/unfamiliar convention penalty: even frontier <50% one-shot (`2507.02833`).
- JSON is a *saturated, familiar* format for structure (`2505.20139`) — so structural
  validity ("is it parseable JSON") will be high; the failures will be in the *canonical*
  details (sort order, number minimization, escaping, no-whitespace).
- Exact-match is strictly harder than every proxy metric used above.

**Predicted NAIVE-PROMPT FLOOR (single instruction "output the RFC 8785 canonical JSON,"
no worked examples, temp 0, cheap tier):**
- Estimate: **~5%–30% exact-match**, best guess **~15%**.
- Reasoning: The model will reliably emit *valid* JSON (familiar format, high structural
  proxy), but plain JCS requires simultaneously getting **key sort order** (UTF-16 code
  unit — a rule models rarely reproduce exactly), **number canonicalization** (the ECMA
  shortest-round-trip form — the single hardest sub-rule; `2505.20139`/`2501.10868` both
  flag number-format handling as a break point, e.g. "89,000"), **exact escaping**, and
  **zero insignificant whitespace**, *all* in one string for a 0/1 score. If the schemas
  are shallow (few keys, no floats, all-ASCII strings) the floor rises toward 30%+; if they
  contain floats, unicode requiring \u escapes, or many keys needing sort, it drops toward
  5%. Wide band because c11's schema-fixing choices (which json-schema-faker schemas) will
  swing this more than model choice — this mirrors `2505.20139`'s finding that
  format/content, not model, dominates.
- Uncertainty: HIGH. No paper measures exact-match canonical JSON. The 5–30% band could be
  wrong if models happen to have strong JCS priors from training on the widely-published
  RFC 8785 spec (a contamination-*helps* risk c11's plain-JCS variant does NOT guard
  against, since JCS is a real public standard — unlike the eventual invented house rules).
  **This is a genuine risk to the floor being higher than expected: plain JCS is
  memorizable public knowledge, so a well-trained model could score much higher than the
  novel-constraint proxies suggest.** This directly bears on the go/no-go decision.

**Predicted CEILING-PROMPT SCORE (best feasible prompt: explicit rule statement + several
worked canonicalization examples covering sort/number/escape/whitespace, temp 0, cheap
tier):**
- Estimate: **~40%–75% exact-match**, best guess **~55%**.
- Reasoning: Strong prompting is the dominant lever (`2602.14743`: DOC_micro 0.01→0.52
  from strategy alone; `2511.03080`: 4–5 pt WER gains from example curation), so a good
  prompt should lift the cheap tier substantially. But the ceiling is capped by (a) the
  DOC_micro-style all-or-nothing proxy ceiling (~0.52 even for *frontier* GPT-4o/Gemma3-27B
  under best strategy) and (b) the persistent Wrong-Value bottleneck that "does not go away
  with scale or prompting" (`2602.14743` Sec. VII) — the analog here is a persistent
  number-format or sort-order slip that fails the whole string. The number-canonicalization
  sub-rule in particular is unlikely to reach 100% even with examples; it's the c11 analog
  of IFBench's "words/sentence" categories that stay hardest post-training (`2507.02833`
  Table 2). So a cheap-tier ceiling meaningfully above ~75% is unlikely on exact match.

**Conflict / caveat to surface:** The floor prediction pulls two ways. IFBench's
novel-constraint evidence argues for a LOW floor (JCS rules are convention-following the
cheap tier is bad at). But plain JCS is a *published, real* standard — `2501.10868`'s own
premise is that JSON structure is a saturated, familiar capability, and `2505.20139` shows
JSON generation is near-ceiling. If cheap models have absorbed JCS/canonical-JSON behavior
from training data (plausible — rfc8785 libraries and the spec are public), the plain-JCS
floor could be **substantially higher than 15%** — possibly 40%+. This is exactly why c11's
design flags that the plain-JCS variant has "NO invented house-rule deviations yet":
**the go/no-go hinges on whether plain JCS is already too easy (memorizable) to be
discriminative.** The evidence supports the worry: JSON is the one format that's saturated
everywhere in this corpus.

**Implication for go/no-go (INFERENCE):**
- If the empirical naive floor comes back **high (>~40%)** and the ceiling near-saturates
  (>~85%), that confirms plain JCS is too memorizable/easy → c11 *needs* the invented
  standard-deviating house rules to create discriminative headroom (the IFBench pattern:
  novel constraints are where the signal is).
- If the naive floor comes back **low (<~20%)** with a mid ceiling (~55%), plain JCS is
  already discriminative and the exact-match+number-canonicalization difficulty alone may
  suffice — added complexity would be optional, not required.
- The strongest single evidence-based recommendation: **run the plain-JCS baseline first
  and check the naive floor specifically for number-heavy / unicode-heavy schemas**, since
  the corpus consistently identifies number formatting and rare-convention details as the
  break points, and schema content will swing the score more than model choice.

---

## Executive summary (10 lines)

1. No paper runs c11's exact task (byte-exact canonical-JSON vs. a deterministic oracle);
   all 7 are partial-credit adjacent proxies, so every c11 prediction is an inference.
2. One-shot: JSON *structural validity* is near-saturated for familiar formats, but
   *full-document exact correctness* tops out at DOC_micro ~0.52 even for GPT-4o/Gemma3-27B
   (`2602.14743`); Wrong-Value is the persistent bottleneck at every scale.
3. Novel, standard-deviating constraints crash one-shot accuracy below 50% even for
   frontier models (`2507.02833` IFBench) — the most c11-relevant SHOWN result.
4. No paper tests an interactive per-instance repair loop; self-refinement helps once then
   saturates (`2404.03868` EDC+R), and weaker models exploit extra context less.
5. Cheap-tier proxies (0.6–14B, "-mini"): high per-field accuracy (F1 0.89–0.96) but low
   full-document validity (DOC_micro 0.01–0.44); prompting strategy beats model scale.
6. Biggest measured lever: in-prompt schema+example (`2602.14743` DOC_micro 0.01→0.52) and
   curated few-shot examples (`2511.03080` WER −4–5 pts); serialization-convention ablation
   is untested anywhere (the exact c11 axis has zero measured effect size).
7. Predicted cheap-tier naive floor for plain JCS: **~5–30% exact match (best ~15%)** —
   but with a real upside risk it's higher because JCS is a memorizable public standard.
8. Predicted cheap-tier ceiling-prompt score: **~40–75% (best ~55%)**, capped by the
   ~0.52 all-or-nothing proxy ceiling and the un-fixable number-canonicalization slip.
9. Conflict to flag: novel-constraint evidence argues LOW floor; JSON-saturation evidence
   argues plain JCS may be too easy/memorizable → floor could exceed 40%.
10. Go/no-go: if the empirical naive floor is high/ceiling saturates, c11 NEEDS invented
    house-rule deviations for headroom; run plain-JCS first on number-/unicode-heavy
    schemas, where the corpus says the breakage (and discriminative signal) lives.
