# Candidate deep-dive builder notes

Terse instructions for filling `_template.html` into one `c{NN}.html` per candidate
(24 pages). Copy the template, replace every `{{PLACEHOLDER}}`, delete the badge/
verdict variants you do not use. The pages sit beside `recommendations*.html` in
look and feel — keep them lean and let content decide density.

## Paths — the `../` rule
Pages live in `candidates/`. Everything else (css, favicon, source docs, `repos/`)
is one level up. So: `../doc.css`, `../flow.css`, `../favicon.svg`,
`../recommendations.html`, `../recommendations-round2.html`,
`../candidates-merged.html`, `../brainstorm.html`, `../repos/{slug}.md`.
Sibling candidate pages are bare: `c07.html`. Never link a css/favicon/source doc
without `../`.

## Section order (fixed — do not reorder)
1. Header — `.cand-head` (id + name + status badge) then `.anchor-sub`
2. Verdict — `.verdict` box + `.chips`
3. What it proposes — `.grid-2` of `.card` (task, example instance)
4. Academic anchor & lineage — `.grid-2` (literature / brainstorm parents) + full-width dead-branches card
5. Related active research (2025-2026) — full-width `.card` list
6. Perspectives — `.grid-3` of five `.lens` blocks
7. Implementation: steps & risks — `.steps` + `.grid-2` (risks, repo/run)
8. Footer nav — `nav.cand-nav`

## CSS classes to use (defined in doc.css/flow.css or the page's `<style>`)
- Layout: `main`, `section-head` > `h2`, `grid-2`/`grid-3`, `card`, `card full`, `space-top`.
- Header: `cand-head`, `cand-id`, `badge` + one state, `anchor-sub`.
- Verdict: `verdict` (`.good` adopt/backup · `.bad` gated/below-cut · plain for ranked); chips: `chips`, `chip`, `chip total`, inner `.k`/`.v`.
- Lineage: `cite`, `unverified`, `dead-branch` + `.why`.
- Perspectives: `lens`, `rowlabel`, `ul.pros`, `ul.cons`.
- Steps/risks: `steps` > `step`, `card` lists, `status` + `.pass`/`.partial`/`.fail`/`.open`.
- Nav: `cand-nav`, `navgroup`, `navlabel`, `spacer`.
- Scan layer: `key` for 1-4 semantic anchors per paragraph/bullet; `muted`, `small` for secondary text.
Add no new component CSS — reuse the family palette variables if you must.

## Status badge — pick exactly one, keep class + label in sync
`adopt` ADOPT · `backup` BACKUP · `ranked` RANKED #n · `gated` GATED · `below-cut` BELOW CUT.
Colors: adopt=green, backup=accent-blue, ranked=neutral, gated=amber, below-cut=red.
Each badge carries its own text label, so it stays accessible without relying on
color alone, in both light and dark. Match `.verdict` tone to the badge
(`good` for adopt/backup, `bad` for gated/below-cut, plain otherwise).

## Tone
- Reasons-first prose, complete sentences. Lead with the conclusion, then the why.
- Titles are contracts: name the subject and its behavior/consequence, not a bare category.
- Preserve qualifications; keep proposed behavior distinct from current fact.
- Cons in a lens are real cons — do not soften them into disguised pros.

## Numbers only as chips
No digits in prose. Every score (R1 rubric, I1-I6, impl total) lives in a `.chip`
in the verdict box and nowhere else. Refer to standing in words ("the higher of the
two run-verified generators", "mid-pack after the rescore"), not "ranked 3rd with 13".
Ranks in the footer nav and the badge are the only other place a number may appear.

## Citations
Format in prose: **Name (Authors, Venue Year)** — e.g. PrOntoQA (Saparov & He, ICLR 2023).
Wrap in `<span class="cite">`; if there is a link, put the `<a>` on the Name.
Keep author lists short (first author + "et al." past two). Same format in both the
lineage section and the related-research section.

## Marking unverified links
Only cite a link as solid if you opened it and confirmed it resolves to the claimed
work. Anything you could not confirm (dead link, guessed venue/year, second-hand
citation) gets `class="unverified"` on the `<li>` (or the `<a>`), which appends a
dashed "unverified" tag. Never present an unconfirmed link as verified. Prefer one
verified source over several shaky ones.

## Footer nav wiring
- Table: `../recommendations-round2.html#table`.
- By rank: prev/next are the adjacent candidates in the round-2 rescore order (bare
  `cNN.html`); include their rank number.
- Related: 1-3 sibling candidates that share a lineage or compete for the same slot.
- Sources: Round 1 `../recommendations.html#cNN`, Round 2
  `../recommendations-round2.html#table`, Candidates merged `../candidates-merged.html#cNN`.
  Confirm the `#cNN` anchors exist in those docs; drop the fragment if they do not.

## Before shipping each page
Open it: css/favicon resolve through `../`, no digits leaked into prose, exactly one
badge, five lenses each with a pro and a con, every unverified link tagged, all
footer links land. Stop at a clean 80% pass.
