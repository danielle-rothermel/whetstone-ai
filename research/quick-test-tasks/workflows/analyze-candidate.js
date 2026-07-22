export const meta = {
  name: 'analyze-candidate-related-work',
  description: 'Deep-read related-work dossiers, syntheses, draft baseline spec, and family-styled HTML docs for a quick-test candidate',
  phases: [
    { title: 'Extract', detail: 'one dossier agent per paper', model: 'sonnet' },
    { title: 'Verify', detail: 'adversarial fact-check per dossier', model: 'sonnet' },
    { title: 'Synthesize', detail: 'capability + positioning syntheses', model: 'opus' },
    { title: 'Spec', detail: 'draft baseline experiment spec', model: 'opus' },
    { title: 'Docs', detail: 'family-styled HTML + candidate-page links', model: 'sonnet' },
  ],
}

// Generalized from the c19 pilot (wf_2c2d5a48-41b). Candidate-specific content comes
// entirely from args: candidate, candidate_title, task_shape, baseline_constraints,
// repo_notes[], papers[{key,file,name,corpus_claim}], base, rubric, polish_skill, fetch_date.

// args may arrive as a JSON-encoded string depending on harness marshalling — normalize.
const A = typeof args === 'string' ? JSON.parse(args) : args

const BASE = A.base
const WORK = `${BASE}/related-work/work/${A.candidate}`
const OUT = `${BASE}/related-work`
const PAPERS = `${BASE}/papers`
const RUBRIC = A.rubric
const SKILL = A.polish_skill
const REPO_NOTES = A.repo_notes.join(', ')

const EXTRACT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['key', 'dossier_file', 'claim', 'protocol_mode', 'observation_summary', 'models_tested', 'corpus_claim_verdict', 'baseline_relevant'],
  properties: {
    key: { type: 'string' },
    dossier_file: { type: 'string' },
    claim: { type: 'string' },
    protocol_mode: { type: 'string', enum: ['single-shot', 'interactive', 'both'] },
    observation_summary: { type: 'string' },
    models_tested: { type: 'array', items: { type: 'string' } },
    corpus_claim_verdict: { type: 'string' },
    baseline_relevant: { type: 'array', items: { type: 'string' } },
  },
}
const VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['key', 'corrections', 'confidence'],
  properties: {
    key: { type: 'string' },
    corrections: { type: 'array', items: { type: 'string' } },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
}

const extractPrompt = (p) => `You are building a related-work dossier for quick-test candidate ${A.candidate} (${A.candidate_title}). Work strictly from LOCAL files — no web access needed.

Read:
1. ${PAPERS}/${p.file} — the paper's verbatim local extraction (header notes fetch route/quality; respect any noted gaps).
2. ${BASE}/candidates/${A.candidate}.md — the "What it proposes" and "Verdict" sections. Orientation: ${A.task_shape}

Then write a dossier to ${WORK}/dossier-${p.key}.json (mkdir -p ${WORK} first) as one JSON object with these fields:
- key, title, authors, venue: verified from the paper header.
- claim: the paper's central claim in 2-3 sentences.
- novelty: what the authors present as the contribution/novelty vs prior work.
- protocol: {mode: "single-shot"|"interactive"|"both", description, observation_content: EXACTLY what the model sees per call — full problem statement? partial feedback per turn? demos? Quote the paper's own prompt format where available.}
- representations: input/representation/format variants compared and their effect on accuracy.
- models: array of {name, strategies (CoT / few-shot / many-shot / algorithm-in-prompt / ReAct etc.), key_results (exact numbers with table/section references)}.
- headline_numbers: the paper's most decision-relevant quantitative findings, each with its table/section ref.
- failure_modes: what failure patterns the paper reports.
- one_shot_vs_multistep: what the paper concludes models can do in one forward pass vs across an interaction loop or with more demonstrations.
- relation_to_candidate: {position: where this paper sits on the family's key protocol axes (single-shot vs interactive/multi-turn; what is observed; output shape — single fact/label vs full sequence/structure; verifier type); overlap with ${A.candidate}; differences; what ${A.candidate} would add beyond it}.
- corpus_claim: ${JSON.stringify(p.corpus_claim)}
- corpus_claim_verdict: does the paper support that claim as written? (supported | partially-supported | unsupported | miscited), with the correction if any.
- baseline_relevant: concrete facts useful for designing our un-optimized baseline (instance sizes/difficulty dials used, prompt formats that swung accuracy, metrics, floors/ceilings observed, contamination/data-release status).
- quotes: 4-8 short verbatim quotes (with section refs) backing the load-bearing facts above.

Ground EVERY number and protocol detail in the paper text with a section/table reference. If the local extraction is missing something (noted gaps, unrendered blocks), mark that field "unverified — missing from extraction" rather than guessing. Never fabricate.

Return the compact structured summary (dossier_file = full path you wrote).`

const verifyPrompt = (p) => `Adversarial verification pass. The dossier ${WORK}/dossier-${p.key}.json was produced from ${PAPERS}/${p.file}. Your job is to REFUTE it: re-read the source paper file and check every number, model name, accuracy figure, protocol classification (single-shot vs interactive), observation-content description, and each verbatim quote (locate it in the source).

For each error found, FIX the dossier file directly (surgical edits preserving structure) and record the correction. Pay special attention to: (a) numbers copied from linearized tables — column misalignment is the classic failure; (b) protocol_mode — a paper can have both single-shot and interactive tracks, do not let one flatten the other; (c) claims about what the observation contains — these drive our design decisions. When done, add a "_verification" field to the JSON: {date: "${A.fetch_date}", corrections: [...], confidence: "high"|"medium"|"low"}.

Return the corrections list and your confidence. An empty corrections list is a legitimate result if the dossier survives genuine checking — but check for real.`

const synthAPrompt = () => `Read all ${A.papers.length} verified dossiers (${WORK}/dossier-*.json). Write a capability synthesis to ${WORK}/synthesis-capability.md for candidate ${A.candidate} (${A.candidate_title}).

Questions to answer, citing dossier keys + the papers' section refs:
1. What do these papers collectively conclude models CAN and CANNOT do in this task family in a SINGLE forward pass — at what instance sizes / difficulty levels does one-shot accuracy hold or collapse?
2. Same for MULTISTEP / interactive / many-shot settings — what breaks, and does interaction or more demonstrations help or hurt vs one-shot?
3. Break down by model tier: frontier reasoning models vs the cheap tier our quick test runs at (Gemini-2.5-Flash / GPT-5-mini / Claude Haiku 4.5 / DeepSeek-Chat class) — the cheap tier's numbers matter most.
4. Which prompting strategies and input representations moved accuracy, and by how much — with exact conditions.
5. Implications for ${A.candidate}'s reseed-only baseline (${A.baseline_constraints}): what naive-prompt floor and ceiling-prompt score does this evidence predict, at the cheap tier, temp 0? State ranges with explicit uncertainty and reasoning — this feeds a go/no-go decision on whether the task needs added complexity.

Separate what the evidence SHOWS from what you INFER. No fabrication; if the dossiers conflict, surface the conflict. End the file with a 10-line executive summary. Return that executive summary as your final message.`

const synthBPrompt = () => `Read: all dossiers in ${WORK}/ (dossier-*.json), ${BASE}/candidates/${A.candidate}.md (full), ${BASE}/trends-2025-2026.md (especially area 9, prompt-optimization benchmarking), and ${BASE}/papers/manifest.json (citations fields: cited_by, by_year, since_2025 — note in your write-up that these are OpenAlex counts, fetched ${A.fetch_date}, which undercount vs Google Scholar and are near-zero-by-construction for 2026 papers).

Write a positioning synthesis to ${WORK}/synthesis-positioning.md for ${A.candidate} (${A.candidate_title}):
1. A positioning map: choose the 2-3 protocol/design axes that best organize this task family (e.g., verification type, constraint/rule provenance, single-shot vs multi-turn, output shape) and place all ${A.papers.length} papers plus ${A.candidate} on them. Identify crowded and empty regions.
2. Per paper: what ${A.candidate}-as-designed adds beyond it (one paragraph each, honest about overlap).
3. Publishability analysis: the project will first run an UN-OPTIMIZED baseline (${A.baseline_constraints}), then potentially prompt-optimize (COPRO/MIPROv2/GEPA). IF optimization produces large verified gains on this task: (a) what is the workshop-paper claim, stated precisely; (b) which publication line does it belong to — this family's own evaluation line, or prompt-optimizer benchmarking (GEPA is an ICLR 2026 oral; MAS-PromptBench 2026 exists — see trends doc); (c) what baselines, ablations, and comparisons would reviewers demand; (d) realistic venues/workshops and why; (e) what result would NOT be publishable (so we can recognize it early).
4. Research-currency verdict from the citation data, with the stated caveats.

Separate evidence from judgment throughout. End with a 10-line executive summary and return that summary as your final message.`

const specPrompt = () => `Read: ${RUBRIC} (the quick-test rubric — criteria 1-14), ${BASE}/candidates/${A.candidate}.md, the run-verified repo notes (${REPO_NOTES}), ${WORK}/synthesis-capability.md, ${WORK}/synthesis-positioning.md.

Write a DRAFT baseline experiment spec to ${WORK}/baseline-spec.md for ${A.candidate} (${A.candidate_title}). The experiment: measure headroom on the EXISTING task shape before building any added complexity. Constraints fixed by prior decisions: ${A.baseline_constraints} Fresh seeds only (never published instances — contamination, rubric criterion 8); exact 0/1 scoring vs an independent oracle; temperature 0; repeats = 3.

Spec contents:
1. Strata design: choose candidate-appropriate stratum axes from the candidate doc and repo notes, N per stratum and total N, justified by rubric criterion 5's resolvability requirement (>=10-point effects on 10-20 task internal evals).
2. The two probe prompts, DRAFTED VERBATIM: (a) a deliberately naive prompt; (b) a best-effort ceiling prompt stating all the standard/default conventions of the existing task. These operationalize rubric criteria 4 and 9.
3. The three-outcome decision rule with proposed numeric thresholds justified from synthesis-capability evidence: (a) naive ~= ceiling ~= high -> no headroom -> proceed to the candidate's added-complexity design; (b) naive ~= ceiling ~= mid/low -> headroom exists but is NOT prompt-closable (raw capability deficit) -> added-complexity design AND keep base difficulty shallow; (c) naive << ceiling -> prompt-closable headroom -> direct prompt optimization is viable on the existing shape.
4. Model selection: an OPEN DECISION. Build a table of models tested by these papers with their results (from the dossiers), propose a candidate set for our baseline, state the tradeoffs — do NOT finalize; the project owner decides.
5. Cost & wall-clock estimate per full baseline run (tokens per instance x N x models x repeats).
6. A rubric-mapping table: every design choice -> which criteria (esp. 4, 5, 8, 9) it serves.
7. An "Open decisions" section listing everything awaiting the owner.

Mark the whole document DRAFT. Ground floor/ceiling expectations in the synthesis; where evidence is thin, say so. Return a 10-line summary of the spec's key numbers and open decisions.`

const doc1Prompt = () => `Produce the polished related-work HTML doc for candidate ${A.candidate}.

First read the doc-polish guidance at ${SKILL}, then study the local family style: ${BASE}/candidates/${A.candidate}.html and ${OUT}/c19-related-work.html (the established template for this doc type — same structure, same ../doc.css and ../flow.css hrefs, .cite/.key/.unverified/.fix conventions, byline format "whetstone-ai · date"). The family style OVERRIDES the skill's bundled kit — join the family.

Content sources (read all): the ${A.papers.length} dossier JSONs in ${WORK}/, ${WORK}/synthesis-capability.md, ${WORK}/synthesis-positioning.md.

Write ${OUT}/${A.candidate}-related-work.html:
- Title: "${A.candidate} Related Work: ${A.candidate_title}". Byline "whetstone-ai · ${A.fetch_date}".
- Per-paper dossier cards: claim, novelty, protocol + EXACT observation content, models/strategies with headline numbers, failure modes, corpus-claim verdict (surface corrections prominently), citation counts (OpenAlex from ${PAPERS}/manifest.json, dated, with the undercount caveat).
- The positioning map from the synthesis, rendered as a simple styled grid/table.
- The capability synthesis (one-shot vs multistep conclusions, cheap-tier focus, predicted floor/ceiling for the baseline).
- The positioning + publishability synthesis.
- Anything unverified or extraction-limited carries a visible flag, matching the family's unverified convention.
Titles as contracts (skill section 4); scan-layer .key anchors (section 7); no process residue. Link back to ../candidates/${A.candidate}.html in a nav row.
Verify all relative hrefs resolve from ${OUT}/. Return the file path + a 5-line content summary.`

const doc2Prompt = () => `Produce the polished baseline-spec HTML doc for candidate ${A.candidate}. Read the guidance at ${SKILL} and match the established family style (see ${OUT}/c19-baseline-spec.html — the template for this doc type — plus ${OUT}/${A.candidate}-related-work.html just created; stylesheet hrefs ../doc.css ../flow.css).

Source: ${WORK}/baseline-spec.md (a DRAFT spec with explicit open decisions).

Write ${OUT}/${A.candidate}-baseline-spec.html:
- Title: "${A.candidate} Baseline Spec: Headroom Before Complexity". Byline "whetstone-ai · ${A.fetch_date}". A visible DRAFT badge/callout at top: this spec awaits owner decisions (model set, thresholds).
- Preserve ALL spec content faithfully: fixed-constraints block, strata design, both verbatim probe prompts (as code blocks), the three-outcome decision rule (three distinct outcome cards), model-evidence table, cost estimate, rubric-mapping table, open-decisions list (visually prominent).
- Recommendations must stay distinguishable from fixed constraints (skill section 3).
Link back to ../candidates/${A.candidate}.html and to ${A.candidate}-related-work.html. Verify hrefs. Return the file path + a 5-line summary.`

const linkPrompt = () => `Surgically link the two new docs from the candidate pages. Edit BOTH ${BASE}/candidates/${A.candidate}.html AND ${BASE}/candidates/${A.candidate}.md, keeping them in sync:
1. In the "Related active research" section: append a short pointer line/item linking to ../related-work/${A.candidate}-related-work.html ("deep-read dossiers + capability/positioning synthesis, ${A.fetch_date}") and ../related-work/${A.candidate}-baseline-spec.html ("draft baseline experiment spec — headroom before complexity").
2. In the footer nav (html: the "Sources" navgroup; md: the Navigation/Sources section): add both links.
Match the existing markup patterns exactly (same classes, same list style — the c19 pages show the established pattern for these links). Change NOTHING else. Return a diff-style summary of the edits made.`

log(`analyzing ${A.papers.length} papers for ${A.candidate}`)

const dossiers = await pipeline(
  A.papers,
  (p) => agent(extractPrompt(p), { label: `extract:${p.name}`, phase: 'Extract', schema: EXTRACT_SCHEMA, model: 'sonnet' }),
  (prev, p) => prev && agent(verifyPrompt(p), { label: `verify:${p.name}`, phase: 'Verify', schema: VERIFY_SCHEMA, model: 'sonnet' }).then(v => ({ extract: prev, verify: v }))
)
const ok = dossiers.filter(Boolean)
log(`dossiers verified: ${ok.length}/${A.papers.length}${ok.length < A.papers.length ? ' — SOME FAILED, synthesis will note the gap' : ''}`)

const [capability, positioning] = await parallel([
  () => agent(synthAPrompt(), { label: 'synthesis:capability', phase: 'Synthesize', model: 'opus' }),
  () => agent(synthBPrompt(), { label: 'synthesis:positioning', phase: 'Synthesize', model: 'opus' }),
])

const [spec, relatedDoc] = await parallel([
  () => agent(specPrompt(), { label: 'spec:baseline', phase: 'Spec', model: 'opus' }),
  () => agent(doc1Prompt(), { label: 'doc:related-work', phase: 'Docs', model: 'sonnet' }),
])

const specDoc = await agent(doc2Prompt(), { label: 'doc:baseline-spec', phase: 'Docs', model: 'sonnet' })
const links = await agent(linkPrompt(), { label: 'doc:link-candidate', phase: 'Docs', model: 'sonnet' })

return {
  dossiers: ok.map(d => ({ key: d.extract.key, verdict: d.extract.corpus_claim_verdict, protocol: d.extract.protocol_mode, corrections: d.verify ? d.verify.corrections : ['VERIFY MISSING'], confidence: d.verify ? d.verify.confidence : 'low' })),
  capability_summary: capability,
  positioning_summary: positioning,
  spec_summary: spec,
  related_doc: relatedDoc,
  spec_doc: specDoc,
  link_edits: links,
}
