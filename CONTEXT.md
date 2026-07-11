# whetstone-ai

Graph-based evaluation platform: generation and scoring run as graph-shaped
specs with explicit LM/prompt boundaries and append-only terminal outcomes,
durable via DBOS through the dr-platform kernel.

## Language

**Experiment**:
A named set of Prediction Specs evaluated together. At the platform
boundary, generation and scoring are linked dr-platform Operations; Experiment
outcome comes from Whetstone's Generation Run and Score Attempt records, not
from platform execution success alone.
_Avoid_: batch, sweep, run (unqualified)

**Prediction**:
The unit of generation work — one graph execution against one task,
defined by a Prediction Spec. At the platform boundary, a Prediction Spec
is the dr-platform Item.
_Avoid_: item, task, sample

**Generation Run**:
One durable execution of a Prediction's graph, with a stable
content-addressed identity per attempt.
_Avoid_: job, workflow (unqualified)

**Node Attempt**:
One execution of a single graph node within a Generation Run, recorded
append-only.

**Score Attempt**:
One scoring pass over a completed Generation Run, recorded append-only.
_Avoid_: eval, grading

**Generation Operation**:
The platform-facing submission that manages Prediction generation through the
shared dr-platform lifecycle.

**Scoring Operation**:
The platform-facing submission that manages Score Attempts through the same
dr-platform lifecycle after Whetstone selects eligible Generation Runs.
_Avoid_: a separate scoring batch/recovery subsystem

### Boundary

**Platform boundary**:
The seam where whetstone's domain nouns become dr-platform's generic
Operation/Item language. Domain nouns stay on the whetstone side; only
platform-facing plumbing speaks Operation/Item.
_Avoid_: leaking prediction/experiment vocabulary into dr-platform, leaking
batch/Operation/Item vocabulary into whetstone domain code
