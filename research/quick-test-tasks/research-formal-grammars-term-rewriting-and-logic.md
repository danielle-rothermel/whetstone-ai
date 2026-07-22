# Quick-Test Candidates — Formal Grammars, Term-Rewriting, and Logic/Deduction

Domain lens: formal grammars, term-rewriting, and logic/deduction (RuleTaker/ProofWriter
style). Breadth survey for the whetstone-ai quick-test task selection. Up to 12 candidates,
**not ranked**. Each is evaluated against the `design/quick-test-rubric.html` criteria.

## Rubric quick-reference (what a good candidate needs)

- **C1** single LLM-call node + eval node (no encoder→decoder chain)
- **C2** exact deterministic programmatic scoring, no sandbox, no partial credit
- **C3** short inputs; output bounded to multiple-choice or a few words
- **C4** default prompts score well below a **known** ceiling; gap closes **incrementally**
- **C5** temp-0 deterministic; ≥10-point prompt effect resolvable on 10–20 tasks
- **C7** hundreds of synthetic, parameterized instances with exact labels
- **C8** contamination-resistant (not a memorizable published set)
- **C9** ceiling/floor/reference prompt known in advance
- **C10** difficulty decomposes into independent latent rules
- **C11** failures diagnostic ("expected X, got Y" reveals which rule was violated)
- **C12** few-shot demos measurably help
- **C13** format violations exercised
- **Tie-breaker** active 2025–2026 research interest

A recurring theme: the *natural-language* logic benchmarks (RuleTaker, ProofWriter, FOLIO,
PrOntoQA) are all published, so they fail C8 as-is and must be **regenerated synthetically**.
Their *generators*, on the other hand, are exactly what the rubric wants. The strongest
rubric fits are abstract-symbol rewriting families where the LLM applies invented productions
to reach a normal form or a True/False/Unknown entailment.

---

## Candidate 1 — RuleTaker (Transformers as Soft Reasoners over Language)

- **Kind:** existing benchmark + parametric generator
- **What it is:** Synthetically generated theories (facts + if-then rules in templated NL) with
  a query; label is True/False under closed-world assumption + negation-as-failure. Five
  datasets D0–D5 by reasoning depth, ~100k theories each. Generate-and-test guarantees the
  depth label. Includes the "birds"/exceptions rulebase (abnormality, ostriches can't fly).
- **Sources:** Clark, Tafjord, Richardson, "Transformers as Soft Reasoners over Language",
  IJCAI 2020 — https://arxiv.org/pdf/2002.05867 (2020). D* datasets:
  https://allenai.org/data/ruletaker
- **Adaptation:** Output is already a 3-way label → trivially bounded (C3). Regenerate from the
  open generator with fresh predicate/entity names to dodge contamination (C8). Bin by depth D0–D5
  to get the incremental gap and known ceiling (C4/C9). Latent rules = negation-as-failure,
  rule chaining depth, exception handling — each an independent prompt lever (C10). Strip the NL
  templating toward abstract symbols to further harden C8.
- **Rubric fit / weaknesses:** Very strong on C1/C2/C3/C7/C9/C10. Depth-labelling gives a natural
  incremental ladder for C4. Weakness: the templated NL is highly stylized and may be partially
  memorized (C8 risk) — must regenerate. CWA semantics can make "Unknown"/"False" ambiguous for a
  naive prompt, which actually *helps* C4 (headroom) but demands a precise reference prompt (C9).

## Candidate 2 — ProofWriter

- **Kind:** existing benchmark + generator (RuleTaker successor)
- **What it is:** Theories of NL rules+facts with queries; supports closed-world (CWA) and
  open-world (OWA, with explicit "Unknown"). Answers True/False/Unknown plus optional full proof
  chains and abductive statements. Depth-controlled like RuleTaker.
- **Sources:** Tafjord, Dalvi, Clark, "ProofWriter: Generating Implications, Proofs, and
  Abductive Statements over Natural Language", ACL Findings 2021 —
  https://aclanthology.org/2021.findings-acl.317/ ; https://allenai.org/data/proofwriter
- **Adaptation:** For the quick test, discard proof generation (that pushes toward multi-node /
  long output) and keep only the True/False/Unknown label (C1/C3). Use OWA so "Unknown" is a real
  third class → richer failure taxonomy for C11. Regenerate synthetically with novel symbols (C8).
- **Rubric fit / weaknesses:** Same strengths as RuleTaker with a cleaner OWA "Unknown" that makes
  failures more diagnostic (C11) and adds a latent rule (open- vs closed-world) for C10. Weakness:
  the richer proof/abduction outputs are a distraction — must be trimmed. Published, so C8 needs
  regeneration.

## Candidate 3 — FLD / FLD★ (Formal Logic Deduction)

- **Kind:** existing benchmark + principled synthetic corpus generator
- **What it is:** Synthetic deductive-reasoning corpus built from a *complete* set of first-order
  predicate-logic deduction rules (axiomatically grounded, so multistep composition can derive any
  rule). Task: given facts + hypothesis, decide proved / disproved / unknown (and optionally emit
  proof steps). GPT-4 solves only ~half, showing large headroom on knowledge-free logic.
- **Sources:** Morishita, Morio, Yamaguchi, Sogawa, "Learning Deductive Reasoning from Synthetic
  Corpus based on Formal Logic", ICML 2023 — https://arxiv.org/abs/2308.07336 (2023);
  code https://github.com/hitachi-nlp/FLD ; follow-up "Enhancing Reasoning Capabilities of LLMs via
  Principled Synthetic Logic Corpus", NeurIPS 2024 — https://arxiv.org/pdf/2411.12498
- **Adaptation:** Keep only the 3-way answer, drop proof-step generation for C1/C3. The generator
  is designed for controlled difficulty (rule set, depth, distractors) → directly supports C4/C7/C10.
  Regenerate with fresh symbols (C8). Reference prompt = spell out the specific deduction rules used
  (C9).
- **Rubric fit / weaknesses:** Among the best-fit: an open, principled *generator* (not just a frozen
  set), demonstrated large headroom (C4), decomposable rules (C10), active 2023–2024 research
  (tie-breaker). Weakness: the intended framing includes proof generation (long output) — must be
  restricted to the label. Some FLD instances embed multi-sentence NL that is longer than ideal (C3).

## Candidate 4 — PrOntoQA (+ PrOntoQA-OOD)

- **Kind:** existing benchmark + synthetic ontology generator
- **What it is:** Each instance samples a synthetic FOL ontology (often *fictional* predicates like
  "every wumpus is a yumpus"), builds a proof chain, and renders it to NL; the model answers a
  True/False query. Ground-truth proof structure enables diagnosing which inference step failed.
  Distractor rules guard against pattern-matching.
- **Sources:** Saparov, He, "Language Models Are Greedy Reasoners: A Systematic Formal Analysis of
  Chain-of-Thought", ICLR 2023 — https://arxiv.org/abs/2210.01240 ; PrOntoQA-OOD, NeurIPS 2023 —
  https://arxiv.org/abs/2305.15269 ; code https://github.com/asaparov/prontoqa
- **Adaptation:** Fictional-name ontologies are already strongly contamination-resistant (C8) — the
  best of the NL logic sets on that axis. Bound output to the True/False token only (C3). Vary hops,
  distractor count, and rule types (composition, negation) for the incremental ladder (C4/C10). Use
  the generator's exact proof to define which latent rule a failure violated (C11).
- **Rubric fit / weaknesses:** Very strong on C8 (fictional symbols), C10/C11 (proof-step diagnosis),
  active research. Weakness: the pure single-hop version can be near-ceiling for strong models — must
  push depth/distractors to preserve C4 headroom. NL rendering is verbose; consider a symbolic render.

## Candidate 5 — Abstract Term-Rewriting to Normal Form (synthetic family, no NL)

- **Kind:** synthetic family (design proposal grounded in TRS theory)
- **What it is:** Give the model a small **term/string rewriting system** over invented symbols
  (rules ℓ→r) and a start term; ask for the **normal form** (irreducible result) or a bounded property
  (e.g., the final symbol, length, or a specific position). Confluence/termination theory guarantees a
  unique normal form when the system is confluent + terminating, so the label is exact and program-checkable.
- **Sources:** Terese, *Term Rewriting Systems*, Cambridge 2003 (foundational); Baader & Nipkow,
  *Term Rewriting and All That*, 1998. Modern automation context: "Automated Strategy Invention for
  Confluence of Term Rewrite Systems", 2024 — https://arxiv.org/html/2411.06409v2 ; Newman's lemma
  (local confluence ⇒ confluence for terminating systems).
- **Adaptation:** Fully synthetic and generatable with a reference rewriter as oracle (C2/C7/C8 — no
  published set exists, maximal contamination resistance). Bound output to the normal-form string /
  a few tokens (C3). Latent rules = each production, plus meta-rules like innermost/outermost strategy,
  overlap handling, and termination (C10). Failures localize to a single misapplied production (C11 —
  ideal for GEPA reflection). Few-shot demos of rule application should help strongly (C12).
- **Rubric fit / weaknesses:** Arguably the purest rubric fit: abstract symbols, exact oracle, per-production
  decomposition, single call, bounded output. Weakness: the designer must **control determinism** — either
  restrict to confluent+terminating systems (unique normal form) or fix a deterministic reduction strategy,
  else the label is ambiguous. Requires building the generator/oracle in-house (no off-the-shelf dataset).

## Candidate 6 — SATBench (puzzles from SAT/CNF formulas)

- **Kind:** existing benchmark + automated generator
- **What it is:** Samples CNF formulas, then generates a natural-language logic puzzle whose
  satisfiability mirrors the formula. Answer is SAT/UNSAT (i.e., "is there a consistent assignment").
  2100 puzzles; solver-based validation. Even o4-mini ~65% on hard UNSAT (near random) → large headroom.
- **Sources:** Wei et al., "SATBench: Benchmarking LLMs' Logical Reasoning via Automated Puzzle
  Generation from SAT Formulas", EMNLP 2025 — https://arxiv.org/abs/2505.14615 ;
  https://github.com/Anjiang-Wei/SATBench
- **Adaptation:** Answer is already binary → bounded (C3). Difficulty via clause/variable count gives the
  incremental ladder (C4). Regenerate CNFs with novel story surface (or skip the story, present raw CNF)
  to control contamination (C8). The SAT solver is the exact oracle (C2).
- **Rubric fit / weaknesses:** Strong C2/C4/C7, very recent (tie-breaker), documented large headroom. Weakness:
  the LLM-generated "story background" adds a generation model into the pipeline and lengthens inputs (C3),
  and can inject noise into the label — either present formulas symbolically or lean on solver validation.
  Latent-rule decomposition (C10) is weaker than rewriting/deduction tasks (SAT is more monolithic search).

## Candidate 7 — FOLIO (first-order-logic NLI)

- **Kind:** existing benchmark (expert-written, FOL-verified)
- **What it is:** 1,430 conclusions over 487 premise sets, each labeled True/False/Unknown, with FOL
  annotations verified by an inference engine. Open-domain, human-authored, logically complex.
- **Sources:** Han et al., "FOLIO: Natural Language Reasoning with First-Order Logic", EMNLP 2024 —
  https://aclanthology.org/2024.emnlp-main.1229/ ; https://arxiv.org/abs/2209.00840 ;
  https://github.com/Yale-LILY/FOLIO
- **Adaptation:** 3-way label is well-bounded (C3). BUT it is human-written and small (~1.4k) → fails C7
  (hundreds of *generatable* instances) and C8 (published, memorizable). To use it, you would need to treat
  its FOL schemas as templates and *regenerate* with fresh predicates — at which point it becomes closer to
  Candidate 3/8 than to FOLIO proper.
- **Rubric fit / weaknesses:** Included for breadth as the canonical FOL-NLI reference. Weak fit as-is:
  not synthetically generatable at scale, real-world knowledge leaks in (breaks the "knowledge-free logic"
  headroom), and contamination risk is high. Best used as a *style reference*, not the quick test.

## Candidate 8 — LogicNLI (controlled FOL NLI)

- **Kind:** existing benchmark + templated generator
- **What it is:** NLI-style dataset diagnostically isolating FOL reasoning; up to 5 reasoning depths,
  small vocabulary (~1077 words), fewer than 50 distinct abstract syntax trees → highly parametric and
  abstract. Labels are entailment classes.
- **Sources:** Tian et al., "Diagnosing the First-Order Logical Reasoning Ability Through LogicNLI",
  EMNLP 2021 — https://aclanthology.org/2021.emnlp-main.303/
- **Adaptation:** The small AST inventory + templated surface is close to what the rubric wants: regenerate
  from the templates with novel entities/predicates (C8), bin by depth for the ladder (C4), bound to the
  entailment label (C3). Latent rules = quantifier handling, negation, multi-hop chaining (C10).
- **Rubric fit / weaknesses:** Good structural fit (abstract, depth-controlled, generatable). Weakness:
  the published surface is memorizable and the limited AST set can be reverse-engineered by a strong prompt
  engineer, threatening C4's "not one-shot guessable" — mitigate by expanding the template grammar. Less
  active recently than SATBench/FLD.

## Candidate 9 — Wason Selection Task (abstract + deontic conditionals)

- **Kind:** existing psychology paradigm, recently adapted to LLMs; synthetic-family potential
- **What it is:** Given a conditional rule "if P then Q" and four cards, pick exactly which cards must be
  turned to test the rule. Abstract versions are famously hard (humans ~10–20%); deontic framings are easier.
  Recent work builds an LLM dataset encoding deontic vs descriptive conditionals.
- **Sources:** "Evaluation of Deontic Conditional Reasoning in LLMs: The Case of Wason's Selection Task",
  EACL 2026 — https://aclanthology.org/2026.eacl-short.42.pdf / https://arxiv.org/html/2603.06416 ;
  classic: Wason 1968.
- **Adaptation:** Output is a small set of card labels → bounded multiple-choice-like answer with an exact
  match check (C2/C3). Generate synthetically over invented P/Q predicates and card values (C8). Latent
  rules = the correct modus-ponens + modus-tollens card selection, avoidance of affirming-the-consequent,
  deontic vs descriptive framing (C10/C11 — errors map cleanly to specific fallacies, great for reflection).
  Documented low baseline gives strong C4 headroom.
- **Rubric fit / weaknesses:** Excellent diagnostic structure (C10/C11) and known low floor (C4). Weakness:
  the *space of distinct instances* is small (four-card structure), so hundreds of truly independent instances
  (C7) require varying framing/domain heavily; risk that a single well-crafted prompt ("check P-true and
  Q-false cards") one-shot-solves the abstract case, threatening C4's incrementality. Best combined with
  framing/latent-rule variation rather than used bare.

## Candidate 10 — Dyck / formal-language membership & repair (well-formedness)

- **Kind:** synthetic family (formal-language theory)
- **What it is:** Present a string over a bracket/symbol alphabet and ask a bounded question: is it a
  member of Dyck-k (balanced)? / what is the minimal number of edits to balance it? / what is the next
  legal symbol? Membership in Dyck-k and CFG acceptance are exactly checkable by a parser.
- **Sources:** Dyck language / Chomsky–Schützenberger theorem (foundational); transformer–Dyck learnability
  literature, e.g. "Self-Attention Networks Can Process Bounded Hierarchical Languages", ACL 2021 —
  https://aclanthology.org/2021.acl-long.292/ ; hierarchical-recognition analysis 2024 —
  https://arxiv.org/pdf/2410.12413
- **Adaptation:** Fully synthetic, exact parser oracle (C2/C7/C8). Output bounded to yes/no or a small
  integer (C3). Latent rules = matching nesting, multiple bracket types, depth limits (C10). Few-shot demos
  of balanced/unbalanced strings should help (C12). Contamination-resistant since instances are random strings.
- **Rubric fit / weaknesses:** Clean, cheap, deterministic. Weakness: pure balanced-parenthesis membership can
  be too *easy or too monolithic* — the difficulty is one skill (counting depth), so C10 decomposition is thin
  unless enriched (multiple bracket types, forbidden-substring regular constraints, "Dyck ∩ regular" per
  Chomsky–Schützenberger). Failure diagnosis (C11) is coarse compared to rewriting/deduction.

## Candidate 11 — L-system derivation (parallel rewriting to a target string)

- **Kind:** synthetic family (Lindenmayer-system rewriting)
- **What it is:** Give an alphabet, an axiom, and a set of **parallel** production rules; ask for the string
  after *n* rewriting iterations, or a bounded feature of it (length, count of a symbol, symbol at position k).
  Deterministic L-systems (D0L) produce an exact unique derivation → exact label. Distinct from sequential
  grammars: rules apply to *all* symbols simultaneously each step.
- **Sources:** Lindenmayer 1968; Prusinkiewicz & Lindenmayer, *The Algorithmic Beauty of Plants*, 1990.
  Recent LLM/benchmark use: "Multi-Language Benchmark Generation via L-Systems", 2025 —
  https://arxiv.org/abs/2512.17616 (uses L-systems to *generate* programs; the rewriting mechanism is the
  reusable idea).
- **Adaptation:** Build a generator that samples D0L systems and asks for a bounded derived feature (C3/C7),
  scored by an exact reference expander (C2). Novel symbols → contamination-proof (C8). Latent rules = each
  production, parallel-vs-sequential application, iteration counting (C10); an error reveals which production
  or the application discipline was misunderstood (C11). Few-shot demos of one/two derivation steps should
  strongly help (C12).
- **Rubric fit / weaknesses:** Very strong synthetic fit; parallel rewriting is a genuinely distinct latent
  rule that models get wrong (good for C4 headroom). Weakness: string length blows up exponentially with
  iterations, so you must cap *n* and ask for a *bounded feature* rather than the full string to satisfy C3.
  The cited 2025 paper targets compiler benchmarks (no QA ground truth), so the dataset itself isn't reusable —
  only the mechanism.

## Candidate 12 — RuleBERT / soft-rule probabilistic deduction (weighted rules)

- **Kind:** existing benchmark + synthetic generator
- **What it is:** Deduction over **soft (weighted) logical rules** — rules hold with a probability/confidence,
  and the task is to infer a fact's truth (or its probability bucket) given weighted rules and facts. Extends
  RuleTaker-style deduction with a confidence dimension.
- **Sources:** Saeed et al., "RuleBERT: Teaching Soft Rules to Pre-Trained Language Models", EMNLP 2021 —
  https://aclanthology.org/2021.emnlp-main.110/ ; https://github.com/MhmdSaiid/RuleBert
- **Adaptation:** Bound the output to a discrete label (True/False, or a coarse probability bin) for exact
  matching (C2/C3). Regenerate weighted theories synthetically (C7/C8). Latent rules = rule chaining plus the
  weight-combination semantics (a genuinely extra latent rule beyond hard deduction → good for C10).
- **Rubric fit / weaknesses:** Adds a distinctive latent dimension (rule weights) that increases headroom (C4)
  and decomposition (C10). Weakness: probabilistic answers risk *non-exact* scoring — must discretize carefully
  to keep C2 clean and avoid partial-credit ambiguity; the exact-weight arithmetic can make failures less
  linguistically diagnostic (C11) than pure symbolic deduction. Published set → regenerate for C8.

---

## Cross-cutting notes

- **Best pure-rubric fits (abstract, generatable, decomposable, diagnostic):** Candidate 5 (term
  rewriting to normal form), Candidate 3 (FLD generator), Candidate 4 (PrOntoQA fictional ontologies),
  Candidate 11 (L-system derivation). These maximize C8/C10/C11 simultaneously.
- **Best off-the-shelf generators to bootstrap quickly:** RuleTaker (1), ProofWriter (2), FLD (3),
  PrOntoQA (4), SATBench (6) — all ship open code with exact oracles.
- **Determinism caveat (C5):** rewriting-based tasks (5, 11) must fix a reduction/derivation strategy or
  restrict to confluent+terminating systems so the ground-truth label is unique.
- **Contamination (C8) is the dominant filter:** every published NL logic set (1,2,4,6,7,8,12) must be
  regenerated with fresh/invented symbols. The two families that never had a public dataset (5 term
  rewriting, and the QA-repurposed 11 L-systems) are contamination-proof by construction.
- **Weakest fits as-is:** FOLIO (7, human-written, small, not scalable) and bare Wason (9, too few
  independent instances). Both are strong *references* but weak quick-test tasks without heavy adaptation.
- **Tie-breaker / 2025–2026 activity:** SATBench (EMNLP 2025), Wason/deontic (EACL 2026), FLD follow-up
  (NeurIPS 2024), automated confluence-strategy invention (2024), L-system benchmark generation (2025).
