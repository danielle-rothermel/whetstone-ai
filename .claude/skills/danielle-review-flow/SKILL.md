---
name: danielle-review-flow
description: Use when reviewing code and producing review findings, or when preparing massive stacked changes for review. Applies Danielle's standard findings-first review flow and contains a WIP workflow for reshaping large branch stacks into reviewer-friendly chunks.
---

# Danielle Review Flow

Use the relevant flow:

- **Standard code review**: use for normal code reviews, PR reviews, review comments, or review summaries.
- **Massive stacked changes review flow**: use when a branch stack or large change set needs to be reorganized for human review.

## Standard Code Review

### Findings first

When reviewing code, lead with concrete findings ordered by impact. Do not lead with praise, broad summaries, style commentary, or optional refactoring ideas.

When reviewing, use this priority order:

1. **Bugs and behavioral regressions**: changed semantics, broken edge cases, state corruption, race conditions, lifecycle or ordering bugs, data loss, incorrect persistence, security vulnerabilities, user-visible failures, and any place where the system writes or reads the wrong data.
2. **Structural risks**: wrong ownership boundaries, duplicated sources of truth, incompatible data model changes, hidden coupling, circular dependencies, misplaced responsibilities, and abstractions that make future changes harder.
3. **Correctness risks**: incomplete validation, missing error handling, invalid assumptions, time or order sensitivity, nondeterminism, partial failure behavior, and mismatches between types, schemas, docs, and runtime behavior.
4. **Contract clarity**: unclear public APIs, ambiguous naming for domain concepts, implicit data shapes, unversioned breaking changes, surprising side effects, and call sites that obscure what is required.
5. **Test and verification gaps**: missing regression coverage, untested edge cases, weak assertions, fixture drift, and tests that pass without proving the behavior under review.
6. **Maintainability problems**: unnecessary duplication, over-broad functions, speculative abstractions, dead paths, avoidable complexity, and difficult-to-debug control flow.
7. **Local readability and style**: naming, comments, formatting, succinctness, code smells, and small simplifications.

When readability or style issues do not obscure correctness, contracts, or future maintenance, treat them as non-blocking. Put summaries and optional suggestions after findings. Use severity labels only when they clarify priority.

### Tool-owned style

Leave mechanical style to project tooling.

- Do not leave review comments about formatting, import order, lint, or mechanical style issues that project tooling can catch or fix.
- When tool-owned style issues are substantial, mention them once at the end and recommend the relevant formatter, linter, type checker, or repo check.
- Discuss style manually when it affects correctness, public API clarity, domain meaning, debugging, or maintainability.

## Massive Stacked Changes Review Flow

This workflow is WIP. The initial setup below is the current usable portion; the later review-chunking procedure still needs to be written.

### Motivation

Use this when a branch stack was organized for efficient implementation but now needs to be reorganized for efficient review.

Implementation order often follows dependency discovery, WIP checkpoints, or agent workflow. Review order should help a human understand the final change set: foundations first, then data flow, then integrations, then tests and docs.

The goal is to preserve the final tree from the most mature branch while reshaping commit history into logical review chunks.

### Initial setup

1. Inspect the current repository state, including the active branch, local changes, local branches, remote branches, and the default remote base branch.
2. Identify the most mature source branch: the branch that contains the complete final tree to review, not necessarily the branch with the cleanest implementation history.
3. If the source branch is unclear, compare candidate branches against the default base by changed files, line delta, and commits ahead.
4. Before switching branches, resolve or preserve unrelated local changes in the current worktree. If local instruction-file changes exist, make sure their substance is already present on the mature source branch before removing them locally.
5. Create a new review-prep branch from the default base branch in the requested worktree.
6. Mirror the mature source branch onto the review-prep branch as one squash-applied change set.
7. Verify that the review-prep branch's working tree matches the mature source branch exactly.
8. Create one giant commit that captures the full mirrored change set.
