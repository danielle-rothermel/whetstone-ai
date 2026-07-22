# Quick-Test Task Search — Process Reflection

Date: 2026-07-21 · whetstone-ai
Scope: a retrospective on the *process* that produced the quick-test candidate
set (not a re-ranking of candidates). Grounded in `design/brainstorm.md`,
`design/candidates-merged.md`, the nine research docs, and the run statistics.

Authoritative target throughout: `design/quick-test-rubric.html`.

Run statistics referenced below:
`brainstorm=76, external=14, domains=8, literature=108, merged=24,
dropped_clusters=9, scored=24, verified=10, fatal_refutations=0`.

---

## 1. What worked well (with evidence)

### 1.1 Brainstorm/literature convergence was strong — and that is the headline signal

Of the 24 merged candidates, **22 are tagged `origin: both`** — independently
surfaced by first-principles brainstorm *and* by academic literature search.
Only two (c10 interlocking-cycle calendar, c21 relational micro-world) are
brainstorm-only, and **zero are literature-only**. This is the single most
reassuring result in the whole run: two methodologically independent generators
(intuition-driven lenses vs. citation-driven breadth search) landed on the same
task families. Convergence of this magnitude means the finalist set is unlikely
to be an artifact of either method's blind spots.

It also answers the "did brainstorm or literature contribute more finalists?"
question cleanly: **neither dominated on discovery; they corroborated.** What
each contributed was *different*, and complementary:
- Brainstorm contributed the *task shapes* and the anti-one-shot engineering
  (per-instance keys, precedence stacks, "invent the glyphs / operators /
  house rules"). The rubric-critical design moves — closed-vocab to protect
  temp-0 resolvability (c09), single-field queries to preserve strict 0/1
  scoring (c11/c12), distractor-isolates-one-rule MC construction (c08/c23) —
  originate in the brainstorm lenses.
- Literature contributed *calibration evidence*: where the default floor sits,
  named diagnostic failure modes (the "decimal round-trip" error in the
  constructed-notation doc), and generator donors (GSM-Symbolic templating,
  InductionBench, RuleTaker/ProofWriter, IFEval/COLLIE checker libraries).
  Crucially, it also supplied the *only* direct proof that our own optimizers
  (COPRO/MIPROv2/GEPA) close incremental, monotone gaps — on HotpotQA/HoVer/PUPA
  (`research-prompt-opt-literature.md` §4–6) — which no brainstorm could provide.

The merge doc used the literature primarily to *validate and de-risk* shapes the
brainstorm had already proposed, rather than to introduce new ones. That is a
healthy division of labor, but see §2.1 for the risk it hides.

### 1.2 External model lanes ran and added non-redundant ideas

All five external/model lanes are traceable in `brainstorm.md` by source tag,
and each contributed ideas that survived merging:
- `kimi-k2p7` → SSC, CSD, CAP (folded into c06, c23, c03).
- `deepseek-v4-flash` → EncodedArith, TokenCounter, MultiRuleSort, CaseOrder
  (into c01, c24; TokenCounter correctly *dropped* as D3).
- `minimax-m3` → Constrained Micro-Description, Orthogonal-Feature Labeling,
  Keyed-DSL Arithmetic (into c22, c13, c03) — c22's whole cluster is anchored
  on the minimax entry.
- `gpt-5.x-codex` → Relational Micro-World, Protocol Bitmask, Opaque Device
  State, Priority Ledger (into c21, c12, c05, c14).

So the external lanes were **not redundant** with the Claude/first-principles
pass: they seeded finalists c14, c21, c22 in particular. The 14 "external ideas"
were a good marginal investment. Worth noting the honest failure mode they also
surfaced: several external ideas (TokenCounter, Inverted-MC) were *char-level or
one-shot-guessable* and got correctly filtered — the lanes added breadth at the
cost of some noise, which is the expected trade and was handled.

### 1.3 The rubric was operationalized into hard filters, not vibes

The merge doc opens by naming the two rubric constraints that "shaped every
merge" — strict per-example 0/1 (no intra-example partial credit) and
MC/few-word bounded output — and then visibly *rewrites* candidates to comply:
every "per-field partial scoring" brainstorm variant was converted to a
single-field/single-token query with incrementality pushed to **rule strata
across the pool**. This is exactly right and is the most sophisticated move in
the whole process: it preserves criterion 10 (decomposable difficulty) *and*
criterion 2 (exact 0/1) simultaneously, which naively conflict. The nine dropped
clusters each cite a specific rubric criterion they fail (D3/D9 → criterion 5
temp-0 noise; D6/D8 → criteria 1–2; D1 → criterion 4 one-shot). The filtering
was criterion-anchored, not taste-anchored.

### 1.4 Char-level noise was caught as a design constraint, not a candidate

The StringLLM/CUTE char-level cluster was retained *as a red-flag caveat*
(D9) that then justifies design choices elsewhere (closed-vocab in c09, dropping
TokenCounter in D3, "prefer closed-vocabulary or output-length variants" in the
brainstorm cross-cutting notes). Turning a weak candidate into a reusable
constraint is a good outcome from a breadth sweep.

---

## 2. What did not work well (with evidence)

### 2.1 Skeptics killed nothing — scoring was almost certainly too credulous

`fatal_refutations=0` across 24 scored / 10 verified finalists, with two
adversarial skeptics per finalist. **Zero kills is a warning sign, not a
victory.** Either the candidate set was already perfectly filtered before the
skeptics ran (unlikely for 24 items), or the skeptic lenses were not sharp
enough / the scoring rubric had no failing band. The merge doc itself flags
real, unresolved risks that a skeptic *should* have been able to convert into a
kill-or-demote:
- c20 (invented micro-game) is self-described as "highest design effort in the
  list" with rules that "must be validated unambiguous with a reference ~100%
  prompt before use" — i.e. its ceiling is *unverified*. That is a criterion-9
  failure risk, yet it survived at full standing.
- c09 (cipher chain) and c18/c23 both carry "validate the noise floor / curate
  complexity before adoption" caveats — unverified criterion-5 and
  criterion-4 risks.
- c15 and c14 both say "confirm the chosen cheap model can override priors at
  all before adopting" — a live threat that the *floor is the ceiling* on a
  small model, which would void criterion 4 entirely.

None of these produced a refutation. The most likely process defect: skeptics
critiqued candidates *in the abstract* rather than being required to **name the
one rubric criterion most likely to fail on the actual pinned cheap model**, and
scoring had no explicit "unverified ceiling/floor ⇒ cap the score" rule. A
credulity check that produces zero kills on 24 bespoke, uncalibrated task
designs is not measuring what it should.

### 2.2 Heavy near-duplication survived into 24 finalists

The merge deduped clusters but the finalist set is still internally redundant:
c01/c02/c03/c10 are four flavors of "invented arithmetic/notation"; c11/c12/c24
are three flavors of "canonicalize/extract/order under invented conventions";
c05/c17/c19/c20 are four flavors of "run an invented machine forward." For a
*quick-test selection* whose entire purpose is to pick **one** task, carrying 24
candidates — many of which are siblings differing only in output format — is more
than the decision needs and dilutes skeptic attention across near-identical
designs. The brainstorm's own cross-cutting notes already identified the robust
clusters (rewrite-to-normal-form appeared 3–4× independently); the process could
have collapsed to ~8 cluster-representatives *before* scoring and spent the
skeptic budget on adversarially calibrating those eight on the real model.

### 2.3 No candidate was actually calibrated on the target model

Every "known ceiling / known floor" claim (criterion 9) in the candidate doc is
*asserted from literature analogy*, never measured. The constructed-notation doc
is explicit that the two purest-fit synthetic families (balanced ternary #10,
keyed novel-operator #11) have "no external validation of difficulty curve — you
must calibrate floor/ceiling yourself on 10–20 evals." The process produced 24
plausible designs but **zero data points** on the one question the rubric makes
falsifiable: does a naive prompt actually score well below a ceiling prompt on
the cheap model, with an incremental middle? A single afternoon of running a
floor prompt + a ceiling prompt on 2–3 cluster-representatives would have been
worth more than the 12th–24th candidate.

### 2.4 The vestigial second objective was under-developed

The rubric's `callout good` makes a second Objective (output bytes ↓) *mandatory*
to claim GEPA / two-objective coverage. The docs treat it as a bolt-on ("output
bytes doubles as the vestigial second objective," c11) rather than a selection
criterion. Only text-output tasks (Themes A/D/E/G) can carry it; the MC-output
finalists (c06, c08, c14, c16, c17-verdict) **cannot**, because a single-letter
answer has no meaningful length-budget axis. That materially narrows which
finalists can validate the two-objective plumbing, and the merge did not flag it
as a discriminator. This should have been a scored column.

---

## 3. What I would change in the workflow next time

1. **Insert a calibration gate between scoring and skeptics.** Before adversarial
   review, run floor-prompt vs. ceiling-prompt on the pinned cheap model over
   10–20 instances for each *cluster representative*. Criterion 4/9 stop being
   assertions. Any candidate whose measured floor≈ceiling or floor≈0 is demoted
   automatically. This directly targets the §2.1 and §2.3 failures.

2. **Collapse to cluster-representatives before scoring, not after.** Score ~8
   representatives, not 24 siblings. Reallocate the freed skeptic budget to
   depth (calibrate each survivor) instead of breadth (re-litigating format
   variants). The brainstorm already names the clusters.

3. **Give skeptics a required output shape.** Each skeptic must (a) name the
   single rubric criterion most likely to fail on the *pinned model*, and (b)
   propose the cheapest experiment that would falsify the candidate. Add an
   explicit scoring rule: "unverified ceiling, floor, or temp-0 noise floor caps
   the score below the promote line." Zero kills should be *impossible* unless a
   calibration experiment cleared the risk. Consider a third skeptic lens
   dedicated solely to "is this one-shot-guessable by a competent prompt
   engineer?" — the rubric's sharpest and most easily-violated criterion.

4. **Make the second objective a scored column, not a footnote.** Tag each
   candidate "carries a length objective: yes/no" and weight it, since it is
   *mandatory* for the GEPA/two-objective claim the quick test exists to support.
   Prefer at least one finalist that carries it natively (a short *string* output,
   not MC).

5. **Prune the domain list at the edges; it was slightly too wide.** Eight
   domains produced heavy overlap in the extraction/canonicalization and
   simulation regions (Themes E/F/G bled into each other repeatedly). Merging
   "structured extraction" and "classification under remapped labels" into one
   lens, and "simulation/state-machines" with "formal grammars/rewriting" into
   one, would cut duplicate literature work without losing coverage — the
   cross-cutting notes show these already collapse in practice.

6. **Keep the external lanes; they paid off.** They seeded three finalists (c14,
   c21, c22) that the internal lens under-weighted (instruction-hierarchy,
   relational vocab, constrained generation). Do *not* cut them to save budget.

7. **Add one lens the current set lacks: "cheapest-to-implement first."** The
   process optimized for rubric fit but never scored *build cost*. c20 is flagged
   as highest-effort; c01/c11-slug/c03 are near-trivial generators. For a
   quick-test whose point is to unblock the debug loop *soon*, implementation cost
   should be an explicit axis, biasing toward the cheapest task that clears the
   calibration gate.

---

## 4. Open questions the search surfaced about the rubric itself

1. **Does "decomposes into independent latent rules" (crit. 10) survive strict
   0/1 scoring (crit. 2)?** The merge doc's key move was pushing decomposition
   from *within* an example to *across pool strata*. This works arithmetically,
   but it means an optimizer only sees incremental credit if the **stratum mix in
   a 10–20-task minibatch is balanced**. The rubric asks for "enough examples
   from each stratum" but does not specify that minibatch *sampling* must be
   stratified. On a 10-task GEPA minibatch, an unstratified draw could hide a
   whole rule's stratum and make a good prompt look no better than a bad one —
   silently breaking crit. 5's "effect size exceeds noise" guarantee. **The
   rubric should require stratified minibatch sampling, or acknowledge the tension.**

2. **Can crit. 4 (incremental gap) and crit. 5 (temp-0 determinism) hold on the
   same cheap model?** Determinism at temp 0 is easiest on short MC outputs, but
   the richest incremental ladders (rewrite systems, cipher chains, arithmetic)
   have longer/char-level outputs where temp-0 resolvability is exactly what the
   D3/D9 caveats warn is fragile. The rubric treats these as independent criteria;
   in practice they *trade off*, and the safe zone (short string, closed vocab,
   no char arithmetic) is narrow. Is that intersection large enough to also carry
   the mandatory length objective (§2.4)? Unresolved without calibration.

3. **What if the cheap model's floor *is* its ceiling?** Multiple candidates
   (c14, c15) carry the caveat "confirm the model can override its prior at all."
   If a small model cannot be moved off a strong pretrained prior by *any* prompt,
   the task has no headroom on *that* model even if a larger model shows a clean
   gap. The rubric's criterion 9 assumes the designer "can write a ~100% prompt" —
   but says nothing about whether the ~100% prompt is reachable *on the pinned
   cheap model*. The known ceiling must be defined **per model**, and the rubric
   should say so.

4. **Is the internal-reward → official-objective rank correlation (crit. 6)
   even testable on a task this clean?** With deterministic temp-0 scoring and
   exact 0/1 rewards, internal and official *are the same measurement* up to
   split identity. The quick test may validate the *plumbing* of crit. 6 but
   cannot stress its *statistics* — the same way crit. 5's non-coverage note
   admits repeat-averaging math is never stressed under determinism. The rubric
   should state explicitly that crit. 6 here is a plumbing check, not a proxy-
   validity measurement (the real proxy question lives in the noisy full
   experiment).

5. **How many latent rules is "enough" before a task stops being one-shot
   guessable (crit. 4) yet stays learnable (crit. 12)?** The docs repeatedly
   settle on "4–5 interacting rules with precedence" as folklore, never derived.
   Too few ⇒ one-shot; too many ⇒ unlearnable (InductionBench's warning that
   models fail even the simplest class). The rubric gives no target and no way to
   estimate it a priori — which is precisely why the calibration gate in §3.1 is
   the highest-value process change: the rubric's central knob is currently
   guessed, not measured.
