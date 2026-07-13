# whetstone-ai

Graph-based evaluation platform: generation and scoring run as graph-shaped
specs with explicit LM/prompt boundaries and append-only terminal outcomes,
durable via DBOS through the dr-platform kernel.

## Language

**Experiment**:
A named set of Prediction Specs evaluated together. At the platform
boundary, exactly one Generation Operation/Manifest fixes membership at first
accepted registration, while scoring may use several linked dr-platform
Operations. Membership growth creates a new Experiment identity/version; Experiment
outcome comes from Whetstone's Generation Run and Score Attempt records, not
from platform execution success alone. Acceptance is strictly complete by
default; a partial result requires an explicit persisted, stratified,
operator-confirmed policy.
_Avoid_: batch, sweep, run (unqualified)

**Experiment Acceptance**:
The caller-owned predicate that determines whether every expected Prediction
and required scoring profile has an accepted domain result. It is separate
from DBOS and platform execution terminality: terminal success of each selected
execution is necessary but not sufficient. Each evaluation is append-only
and pins its exact domain and platform source cut; the Experiment points to at
most one current evaluation, and any relevant later outcome makes the prior
evaluation historical until reevaluation succeeds. When a Prediction has more
than one successful Generation Run at the evaluation cut, the accepted run is
the one with the highest dr-platform Attempt ordinal within the single accepted
Generation Operation/Item lineage; earlier successes remain
superseded provenance and do not create required scoring cells.
`PARTIAL` Generation Runs remain eligible for scoring selection only when
persisted terminal output contains a non-POSIX-whitespace character; empty and
whitespace-only rows are excluded before Manifest identity. This populated-
only set intentionally replaces the legacy status-only intermediate set, but
these runs do not satisfy strict Generation acceptance without a separate explicit
persisted policy. Before scoring exists, an empty accepted-Scoring set produces
a durable `PARTIAL` evaluation with explicit `MISSING_SCORE` cells; later
scoring appends a new evaluation.
_Avoid_: treating Operation success or one global completion percentage as
proof that an Experiment is complete, mutating an earlier acceptance decision,
choosing "latest" independently in each reader, requiring scores for a
superseded successful Generation Run

**Prediction**:
The unit of generation work — one graph execution against one task,
defined by a Prediction Spec. At the platform boundary, a Prediction Spec
is the dr-platform Item. Prediction identity remains a domain identity;
execution equality additionally requires dr-platform's complete versioned
Execution Recipe digest.
_Avoid_: item, task, sample

**Generation Run**:
One durable execution of a Prediction's graph, with a stable
content-addressed identity per platform-owned Attempt. Whetstone determines
whether a Domain Outcome is eligible for another Attempt but does not allocate
the ordinal. Multiple ordinals may succeed; Experiment Acceptance selects the
successful run with the highest ordinal at its pinned evaluation cut.
_Avoid_: job, workflow (unqualified)

**Node Attempt**:
One execution of a single graph node within a Generation Run, recorded
append-only.

**Score Attempt**:
One scoring pass over a completed Generation Run, recorded append-only and
identified by the platform-owned Attempt ordinal. Whetstone determines domain
eligibility for another scoring pass but does not allocate the ordinal.
For an overlapping logical cell, acceptance selects from the newest accepted
Scoring relationship containing a success for the cell's pinned accepted
Generation Run, then its highest successful Attempt in that Item lineage.
Other-run candidates remain immutable superseded-generation provenance and
cannot satisfy the cell. A selected Score Attempt is acceptance-compatible only
after its exact DBOS/platform execution is terminal successful.
_Avoid_: eval, grading

**Generation Operation**:
The platform-facing submission that manages Prediction generation through the
shared dr-platform lifecycle. An Experiment accepts exactly one Generation
Operation/Manifest; unequal replacement is rejected and growth requires a new
Experiment identity/version.

**Scoring Operation**:
The platform-facing submission that manages Score Attempts through the same
dr-platform lifecycle after Whetstone selects eligible Generation Runs. Each
frozen selection is a distinct Scoring Operation; one Experiment may combine
results from several selections during acceptance.
_Avoid_: a separate scoring batch/recovery subsystem

### Boundary

**Platform boundary**:
The seam where whetstone's domain nouns become dr-platform's generic
Operation/Item language. Domain nouns stay on the whetstone side; only
platform-facing plumbing speaks Operation/Item.
_Avoid_: leaking prediction/experiment vocabulary into dr-platform, leaking
batch/Operation/Item vocabulary into whetstone domain code
