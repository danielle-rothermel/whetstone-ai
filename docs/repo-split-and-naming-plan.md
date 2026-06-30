# Repo split and naming plan

**Status:** decision note
**Date:** 2026-06-30
**Scope:** extracting this package out of the nested DSPy fork into a
standalone personal-org repo, and naming it

This note captures the decision to stop developing inside a nested directory of
a `stanfordnlp/dspy` fork and instead move the eval platform into its own
repository in a personal organization, along with the name chosen for it.

## Summary

The dr-dspy eval platform currently lives at `dr-dspy/` inside a fork of
`stanfordnlp/dspy`. The fork only ever made sense as a path to contributing
changes back upstream. There is no intention to upstream anything to DSPy, so
the fork relationship is pure overhead.

The plan is to:

1. Land the current `graph-workflow` branch work first (one risky thing at a
   time).
2. Extract the nested package into a fresh standalone repo, preserving git
   history.
3. Make DSPy a pinned dependency instead of a vendored substrate.
4. Host it in a personal organization where org-reserved dev tooling (e.g.
   Depot) and stronger CI can be set up against a clean repo root.
5. Rename the package from `dr_dspy` to `whetstone` as a separate, isolated
   commit after the move settles.

The repo identity is intentionally **broad** (a natural-language optimization
research platform), while the current code stays **narrow** (just HumanEval,
just DSPy-based generation) to honor the current research scope constraint.

## Why split it out

### The fork only pays off if you upstream

A fork's entire reason to exist is the pull-request-back relationship with the
upstream repository. With no intention to contribute to `stanfordnlp/dspy`, the
fork carries the whole DSPy source tree, its CI config, its `pyproject.toml`,
its issue templates, and its history as ambient noise to filter past. None of it
serves this project.

### The nested directory fights the tooling

This is the concrete blocker for the stated goal of better CI. Depot,
branch-based CI, coverage gates, and release automation all assume *the repo
root is the project root*. Right now the project root is `dr-dspy/`, with
with `pyproject.toml`, `src/`, tests, and Alembic config nested inside a
directory whose parent is an unrelated library. Every CI tool wired up against
this layout needs path gymnastics, and some will not cooperate. The "improve CI"
effort would be spent fighting the layout instead of improving CI.

### DSPy should be a dependency, not a substrate

The clean relationship to DSPy is a pinned PyPI dependency in the lockfile:
`uv add dspy`, pinned and upgraded deliberately. That is strictly better than
vendoring the whole tree — reproducibility, a clear version boundary, and no
conflation of "my code" with "their code." The platform design already steps
*away* from DSPy internals (the `PlainPromptAdapter` work exists precisely to
stop depending on DSPy's `ChatAdapter` hidden formatting). The repo structure
should reflect that boundary.

## Aspiration: broad name, small current slice

The research scope is currently constrained to *just HumanEval* and *just
DSPy-based* approaches. That is a constraint on **what gets built and
published**, not on **how the infrastructure is organized** — those are
separable, as long as we are disciplined about not prematurely generalizing.

The eval platform architecture already built here is method-agnostic. The
append-only-facts substrate, the mutable projection, and DBOS-owns-execution
split answer one question — "given a system and an optimization loop, what
happened, and what should analysis use?" — that is identical whether the
optimizer is COPRO, an RL policy, a knowledge-graph iterator (e.g. a Cognee-style
loop), or a coding agent in a sandbox with a `test_prompt` tool and a bounded
run budget.

So the intended shape is:

- **One repo, broad identity, narrow current content.** The repo's *identity* is
  the umbrella (natural-language optimization research). The *current code* stays
  scoped to exactly the approved HumanEval/DSPy slice.
- **Reserve namespace, do not fill it.** A package namespace with room for future
  siblings signals where work *would* go without committing to building it now.
- **Future approaches are siblings, not current obligations.** Knowledge-graph,
  RL, and agent-sandbox ideas live as README "future directions," not
  scaffolding. This is the discipline that keeps the umbrella honest while the
  research scope stays intact.

```
whetstone/                 # repo + top-level package
├── humaneval/             # approved scope: just HumanEval
├── graph/                 # the current DSPy-based generation path
├── platform/              # eval substrate: append-only facts, DBOS, projections
└── ...                    # future siblings: kg/, rl/, agent/  ← reserved, not built
```

## Name: whetstone

The chosen name is **whetstone** — a stone for sharpening tools. The sharpening
metaphor fits the work better than forging: the platform *refines* an existing
system's performance rather than creating from raw material. The name carries
personal affiliation and is broad enough to house prompt optimization, RL,
knowledge-graph, and agent-based approaches without being literal about any one
method.

Other candidates considered and dropped:

- **anneal** — evocative and method-adjacent (simulated annealing), but reads
  old-school, and the bare PyPI name is occupied by a *real* annealing-
  optimization package, which would actively confuse. The name being claimed by
  the thing it evokes confirmed dropping it.
- **crucible** — strong "refined under pressure" metaphor, but saddled with the
  Atlassian Crucible code-review brand in the dev-tools space we are about to set
  up CI in. Noisier than whetstone for this context.

### Whetstone name-variant availability

Names checked on 2026-06-30. PyPI normalizes `-` and `_` to the same project, so
`whetstone-ai` and `whetstone_ai` are the same distribution there. GitHub
org/user handles **cannot contain underscores**.

| Surface | Name | Status | Notes |
|---|---|---|---|
| PyPI | `whetstone` | **TAKEN** | Stale placeholder: v0.6.0, no summary, no real homepage. Looks abandoned. |
| PyPI | `whetstone-ai` | **AVAILABLE** | (also `whetstone_ai`, `whetstoneai` — same project) |
| PyPI | `whetstone-opt` | **AVAILABLE** | alternative if `-ai` ever undesirable |
| PyPI | `pywhetstone` | **AVAILABLE** | alternative |
| GitHub org | `whetstone-ai` (`Whetstone-AI`) | **TAKEN** | Empty org created ~2026-04-30, 0 public repos, no name/bio. A handle squat / not-yet-launched; not an active product to collide with. |
| GitHub org | `whetstoneai` | **FREE** | usable as a dedicated org handle if ever wanted |
| GitHub repo | `<personal-org>/whetstone-ai` | **AVAILABLE** | unaffected by the empty `Whetstone-AI` org — a repo inside an existing org does not need the top-level handle |

Other existing unrelated GitHub repos named "whetstone" (e.g. `iliaal/whetstone`,
`johniwasz/whetstone.chatgpt`, `thoutz/Whetstone`) are in separate namespaces and
do not block a repo inside a personal org.

### Chosen identity

```
GitHub repo:     github.com/<personal-org>/whetstone-ai
PyPI dist name:  whetstone-ai          # whetstone_ai / whetstoneai also reserved-able
import name:     whetstone             # keep code clean; whetstone_ai if literal preferred
```

The `-ai` suffix resolves the only real weakness of bare `whetstone` (the dead
PyPI placeholder plus generic-word ambiguity) by scoping it to the domain, while
keeping the sharpening metaphor. The PyPI distribution name matches the repo
name, and the import name stays the clean `whetstone`.

Note: a PyPI distribution name and a Python import name can diverge cleanly.
Even if the distribution name is `whetstone-ai`, the wheel can ship the
`whetstone` import package:

```toml
[project]
name = "whetstone-ai"          # distribution name (only matters if published)

[tool.hatch.build.targets.wheel]
packages = ["src/whetstone"]    # import name stays the clean word
```

For a research platform that may never publish to public PyPI, the distribution
name only matters the day it is published; the import name `whetstone` works
locally and in CI regardless.

## Name reservations across git hosts

GitHub remains the **actual home** for this project for now: better CI, Depot,
and the real extraction all land there. The reservations below are defensive
name-holds on services that *might* become the home later — cheap insurance
against namespace contention if `whetstone-ai` takes off — not a migration off
GitHub today.

Context: there is a documented 2026 "GitHub exodus" narrative (reliability
incidents, GitHub folding into Microsoft's CoreAI division in Aug 2025, a
Copilot train-by-default change), with high-profile projects (Zig → Codeberg,
Ghostty/Mitchell Hashimoto) leaving. It is still a vocal minority — surveys show
GitHub at ~59% developer preference — but the new entrants are real.

> Note: this landscape was researched on 2026-06-30 via live web search and
> moves fast. Confirm reservation mechanics on each service's own page before
> relying on them.

**Reserve `whetstone-ai` now (priority order):**

1. **Cursor Origin** — `cursor.com/origin`. The headline "modern GitHub,"
   launched June 16–17, 2026 at Cursor's "Compile" conference (this is the
   recent launch that prompted this question). Agent-first git forge from
   Cursor/Anysphere, built by the acquired Graphite team — stacked PRs, AI
   merge-conflict resolution, storage built for massive parallel agent commits.
   **Waitlist-only; GA targeted fall 2026.** Highest namespace-contention risk
   and most likely to matter. Join the waitlist and grab the name if offered.
2. **Codeberg** — `codeberg.org`. Runs Forgejo, the open-source momentum leader
   and where principled GitHub-leavers are landing. Free, instant handle.
3. **Tangled** — `tangled.sh`. European, decentralized, built on the AT Protocol
   (Bluesky's stack). Raised €3.8M in March 2026; angels include the ex-GitHub
   CEO and Tailscale's CEO. Trivial to grab a handle.

**Skip preemptive reservation:**

- **GitLab**, **Bitbucket**, **Sourcehut** — low namespace pressure; reserve
  only if one becomes the actual home.
- **Radicle** — peer-to-peer, identity is key-based; there is no central
  namespace to claim.

## Extraction runbook (after `graph-workflow` lands)

History-preserving, low-risk, mechanical. Run against a fresh clone.

1. **Land `graph-workflow` → `main`.** Includes dr-dspy-scoped CI
   (`dr_dspy_tests.yml`), portable `scripts/ci/` entrypoints, and pinned
   `dspy==3.3.0b1`. One risky thing at a time.
2. **Extract with history:**
   `git filter-repo --subdirectory-filter dr-dspy` against a fresh clone
   → a new repo where this package is the root, every commit preserved, the
   vendored DSPy tree and upstream history dropped.
3. **Cut the DSPy cord:** remove `tool.uv.sources` workspace override and
   `scripts/ci/ensure_pypi_dspy.sh`; keep `dspy==3.3.0b1` pinned from PyPI.
4. **Create the repo in the personal org**, push, then wire **Depot + CI**
   against the now-clean root (drop `working-directory: dr-dspy` from workflows).
5. **Rename the package** `dr_dspy` → `whetstone` as a *separate, dedicated
   commit* after extraction — a large mechanical diff, isolated from the
   structural move for a clean review.

Keep the Cognee / RL / agent-sandbox ideas as README "future directions" rather
than scaffolding, so the umbrella stays honest while the current research scope
stays intact.
