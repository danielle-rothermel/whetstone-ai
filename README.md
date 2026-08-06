# Whetstone

[![CI](https://github.com/danielle-rothermel/whetstone-ai/actions/workflows/whetstone_tests.yml/badge.svg)](https://github.com/danielle-rothermel/whetstone-ai/actions/workflows/whetstone_tests.yml)

| [Repo Definitions](https://danielle-rothermel.github.io/whetstone-ai/) | [Terms](https://github.com/danielle-rothermel/whetstone-ai/blob/main/.defs/terms.toml) | [Contracts](https://github.com/danielle-rothermel/whetstone-ai/blob/main/.defs/contracts.toml) | [Changelog](https://github.com/danielle-rothermel/whetstone-ai/blob/main/CHANGELOG.md) | [dr-code](https://github.com/danielle-rothermel/dr-code) | [dr-exec](https://github.com/danielle-rothermel/dr-exec) | [dr-graph](https://github.com/danielle-rothermel/dr-graph) | [dr-providers](https://github.com/danielle-rothermel/dr-providers) | [dr-serialize](https://github.com/danielle-rothermel/dr-serialize) | [dr-store](https://github.com/danielle-rothermel/dr-store) | whetstone-envs (private) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

**Whetstone evaluates and optimizes prompt candidates through typed,
reproducible experiment contracts.** Its functionality is organized into
these areas:

- **[Core identity and effects](src/whetstone/core/)** provide validated IDs,
  content-addressed references, evaluation roles, and replay-safe effect
  primitives shared by every functional area.
- **[Experiment modeling](src/whetstone/experiment/)** binds candidates,
  computation graphs, objectives, rewards, and realized evaluation settings
  into immutable configurations.
- **[Environments and sampling](src/whetstone/envs/)** assemble task pools,
  internal and official splits, rollout definitions, and reward policies for
  code-generation and encoder-decoder experiments.
- **[Provider interaction](src/whetstone/provider/)** classifies transport
  outcomes, applies bounded semantic retry policy, and retains the exact
  evidence for every completed attempt.
- **[Execution and recovery](src/whetstone/execution/)** fan out process work,
  preserve partial progress, reuse prompt results, and resume completed work
  without changing its identity.
- **[Evaluation and scoring](src/whetstone/evaluation/)** own evaluation
  definitions and configs, task-and-repeat plans, measurements, compression,
  aggregation, graph execution, and complete reward evidence.
- **[Optimization](src/whetstone/optimization/)** provides the shared harness,
  proposal and tool contracts, native COPRO, MIPROv2, and GEPA flows, and a
  Codex MCP adapter with a typed output artifact.
- **[Coordination and authority](src/whetstone/coordination/)** arbitrate
  durable proposal and evaluation work, official selection, ownership claims,
  and terminal result binding across recovery.
- **[Validation runner](src/whetstone/runner/)** drives
  optimizer-environment cells, enforces budget guards, and publishes resumable
  ledgers and viewer projections from durable evidence.

The repository boundaries follow the same shape:

| Boundary | Package | Responsibility |
| --- | --- | --- |
| Core | `whetstone.core` | Shared identity, roles, and effect primitives |
| Experiment | `whetstone.experiment` | Candidates, bindings, graph identity, objectives, and rewards |
| Environments | `whetstone.envs` | Task pools, sampling, rollout definitions, and environment-specific policy |
| Provider | `whetstone.provider` | Provider requests, attempt evidence, classification, and retry policy |
| Execution | `whetstone.execution` | Process fanout, partial progress, prompt caching, and resume behavior |
| Evaluation | `whetstone.evaluation` | Evaluation configs, plans, drivers, traces, measurements, compression, scoring, evidence, and aggregates |
| Optimization | `whetstone.optimization` | Shared optimization contracts plus COPRO, MIPROv2, GEPA, Codex, and tool use |
| Coordination | `whetstone.coordination` | Durable claims, official authority, and proposal/evaluation services |
| Validation runner | `whetstone.runner` | Resumable validation cells, budgets, ledgers, and viewer projections |

The excerpts below show stable contract shapes; validation and implementation
details are omitted so this overview can remain useful as internals evolve.

## [Core identity and effects](src/whetstone/core/)

Core owns the small identity vocabulary shared across persistence, subprocess,
and optimization boundaries, plus the closed evaluation roles.

```python
@verify(UNIQUE)
class EvaluationRole(StrEnum):
    INTERNAL = "internal"
    OFFICIAL = "official"
```

```python
class TypedRef(BaseModel):
    schema_name: NonEmptyId
    content_hash: ContentHash

class TerminalFailure(BaseModel):
    code: NonEmptyId
    message: NonEmptyId
    details: ImmutableJsonObject
```

## [Experiment modeling](src/whetstone/experiment/)

Experiment models bind the candidate being changed to the exact policy,
environment, objective, and provenance under which it is evaluated.

```python
class Candidate(BaseModel):
    candidate_id: NonEmptyId
    base_ref: TypedRef
    payload: ImmutableJsonObject

class EvaluationBinding(BaseModel):
    eval_config: EvalConfigRef
    role: EvaluationRole
    authority_principal: NonEmptyId | None
    campaign: NonEmptyId
    environment_fingerprint: ExecutionEnvironmentFingerprint
```

```python
@verify(UNIQUE)
class Direction(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"

@dataclass(frozen=True, slots=True)
class Objective:
    name: str
    value: float
    direction: Direction
    derivation: ObjectiveDerivation
```

## [Environments and sampling](src/whetstone/envs/)

Environment packages compose reusable evaluation inputs without coupling the
evaluation engine to a particular task family or graph implementation.

```python
class RolloutDefinitionLike(Protocol):
    @property
    def graph_hash(self) -> str: ...
    @property
    def provider_call_config(self) -> ProviderCallConfig: ...
    @property
    def procedure_config_hash(self) -> str: ...

@dataclass(frozen=True, slots=True)
class EnvExperiment:
    env_name: str
    rollout_definition: RolloutDefinitionLike
    initial_candidate: Candidate
    ceiling_candidate: Candidate
    eval_configs: EnvEvalConfigs
    reward_policy: RewardPolicy
```

```python
class Completeness(StrEnum):
    PROPAGATE = "propagate"
    SKIP = "skip"
```

## [Provider interaction](src/whetstone/provider/)

Provider contracts separate raw invocation evidence from Whetstone's semantic
classification and keep the complete ordered retry history.

```python
class ProviderCallAttempt(BaseModel):
    logical_call_id: StrictStr
    attempt_number: StrictInt
    execution_policy_hash: StrictStr
    evidence: ProviderInvocationEvidence
    generation: Generation | None
    semantic_failure: ProviderSemanticFailure | None

class ProviderCallResult(BaseModel):
    logical_call_id: StrictStr
    request_identity: dict[str, Any]
    execution_policy_hash: StrictStr
    attempts: tuple[ProviderCallAttempt, ...]
    generation: Generation | None
    semantic_failure: ProviderSemanticFailure | None
```

```python
class ProviderExecutionPolicy(BaseModel):
    transport_policy: ProviderTransportPolicy
    max_attempts: StrictInt
    retry_eligibility: dict[SemanticFailureClass, bool]
    backoff: BackoffSchedule
```

## [Execution and recovery](src/whetstone/execution/)

Execution owns bounded subprocess fanout and the durable local state used to
resume provider work without treating elapsed time as evidence of progress.

```python
class ProcessJob(BaseModel):
    entrypoint: StrictStr
    payload: JsonValue

@dataclass(frozen=True, slots=True)
class CallSpec[K, R]:
    key: K
    job: ProcessJob
    decode: Callable[[JsonValue], R]
    deadline_seconds: float
    commit: Callable[[R], None] | None
    cancellation_barrier: Callable[[], None] | None
```

```python
@dataclass(slots=True)
class PartialLog:
    path: Path
    def append(self, record: PartialCallRecord) -> None: ...
    def load(self) -> list[PartialCallRecord]: ...

class PromptResultCache:
    def get_result(
        self, key: str
    ) -> tuple[ProviderCallResult, CacheProvenance] | None: ...
    def put(
        self,
        key: str,
        *,
        request_identity: dict[str, Any],
        execution_policy_hash: str,
        repeat_index: int,
        drive_ordinal: int,
        result: ProviderCallResult,
        phase: str,
        unit: str,
        logical_call_id: str,
    ) -> CacheProvenance: ...
```

## [Evaluation and scoring](src/whetstone/evaluation/)

The canonical engine turns one immutable request into a durable evidence graph
whose rows, traces, aggregates, and optional reward all address exact records.

```python
@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    candidate: Candidate
    evaluation_binding: EvaluationBinding
    purpose: str

class EvaluationEngine:
    def evaluate(self, request: EvaluationRequest) -> EngineEvaluation: ...
```

```python
class EvaluationEvidence(BaseModel):
    candidate: CandidateRef
    evaluation_binding: EvaluationBinding
    graph_hash: StrictStr
    task_identities: tuple[str, ...]
    row_accounting: RowAccounting
    component_traces_ref: TypedRef
    outputs_ref: TypedRef
    aggregate_ref: TypedRef
    reward_ref: RewardRef | None
```

## [Optimization](src/whetstone/optimization/)

Optimization keeps one harness contract while algorithm-specific code lives in
[COPRO](src/whetstone/optimization/copro/),
[MIPROv2](src/whetstone/optimization/miprov2/),
[GEPA](src/whetstone/optimization/gepa/), and
[Codex](src/whetstone/optimization/codex/) subpackages.

```python
class StepMode(StrEnum):
    PURE = "pure"
    PROPOSAL_ONLY = "proposal_only"
    TOOL_USING = "tool_using"

class OptimizationStepRequest(BaseModel):
    run: OptimizationRunRef
    step_id: NonEmptyId
    step_index: NonNegativeInt
    candidates: tuple[Candidate, ...]
    budget: BudgetState
    step_output_contract: OutputContract
```

```python
class OptimizerAdapter(Protocol):
    def invoke(
        self,
        request: OptimizationStepRequest,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput: ...

class CodexRunner(Protocol):
    def run(
        self,
        request: OptimizationStepRequest,
        handle: RuntimeToolHandle,
    ) -> CodexRunResult: ...

class CodexOutputArtifact(BaseModel):
    run_id: StrictStr
    proposals: tuple[Candidate, ...]
    conversation_evidence: dict[str, Any]
    control_cost: dict[str, Any]
```

The subprocess Codex runner uses the pinned official MCP SDK to validate the
protocol and tool-input boundary. It fails closed without macOS `sandbox-exec`
and limits filesystem access to staged runtime and declared state paths. It
accepts the configured Codex executable without enforcing a CLI version and
does not claim network, credential, or descendant-process containment.
Proposal acceptance validates the typed artifact and mutation contract; it
does not require matching MCP evidence for every proposal. A configured
partial log grants write access to its parent directory, so that directory
must not contain unrelated state.

## [Coordination and authority](src/whetstone/coordination/)

Coordination owns process- and restart-safe claims and binds terminal
evaluation or proposal results to the exact durable work request.

```python
class EvaluationIntentClaim(BaseModel):
    intent_ref: TypedRef
    owner_id: StrictStr
    event_ordinal: StrictInt
    generation: StrictInt
    heartbeat_ordinal: StrictInt
    expires_at: StrictFloat
    result_attestation_ref: TypedRef | None

class EvaluationResultAttestation(BaseModel):
    graph_hash: IdentityHash
    resolution: IntentResolution
```

```python
class EngineEvaluationService:
    @property
    def replay_policy(self) -> ReplayPolicy: ...
    def resolve_evaluation_intent(
        self, intent: EvaluationIntent
    ) -> IntentResolution: ...
    def validate_resolution_graph(
        self, resolution: IntentResolution
    ) -> None: ...
```

The authoritative repository vocabulary and standing rules are
[`terms.toml`](https://github.com/danielle-rothermel/whetstone-ai/blob/main/.defs/terms.toml)
and
[`contracts.toml`](https://github.com/danielle-rothermel/whetstone-ai/blob/main/.defs/contracts.toml).
The [published definitions page](https://danielle-rothermel.github.io/whetstone-ai/)
renders those files directly rather than maintaining a generated copy. To view
it locally, serve `.defs/` over HTTP.

## Known limitation: concurrent tool-call identity conflicts

Within one durable optimization step, each `call_id` must identify one exact
`ToolCall`. Retries and concurrent recovery may reuse that same call, but
separate issuers must not submit different calls under one ID.

The current storage boundary publishes the call-ID claim and ordinal slot as
separate immutable bindings. If incompatible same-ID calls race, a losing slot
can enter the durable prefix before the claim conflict is recognized. Later
recovery then raises `IssuedToolCallConflictError` while reading that prefix,
and the losing slot continues to occupy tool-call capacity.

Removing this caller restriction requires an all-or-nothing multi-binding
transaction in `dr-store`: the call-ID claim and ordinal slot must publish
together, ordinal conflicts alone may retry, and an incompatible call-ID loser
must publish no slot. Binding the claim first is not a safe intermediate fix,
because a crash before slot publication would omit the attempted call from the
replay prefix.

## Testing

Run the complete local gate before committing or pushing:

```bash
./scripts/pre-check.sh
```

Install the same gate for both Git hooks with:

```bash
uv run pre-commit install
```

The authoritative unit lane is serial:

```bash
uv run pytest tests/ -q \
  -m "not process_integration and not postgres_integration and not sqlite_time_integration and not sqlite_contention"
```

For a faster local iteration loop, the same selection can use a fixed four
workers with load balancing:

```bash
uv run pytest tests/ -q \
  -m "not process_integration and not postgres_integration and not sqlite_time_integration and not sqlite_contention" \
  -n 4 --dist=load
```

The parallel command is a local convenience, not the CI default. Keep the
worker count bounded; do not replace it with `-n auto`. The isolated serial
integration entrypoints are `scripts/ci/process-integration.sh`,
`scripts/ci/sqlite-time-integration.sh`,
`scripts/ci/sqlite-contention.sh`, and
`scripts/ci/postgres-integration.sh`. The PostgreSQL entrypoint requires
`WHETSTONE_TEST_POSTGRES_DSN`. For a local PostgreSQL 17 service that listens
only on loopback, run `scripts/local/postgres-integration.sh`. It provisions a
unique least-privilege role and database, removes both after successful
validated cleanup, and otherwise leaves them for inspection. It never uses the
durable runtime database as a test fallback. Run the complete serial suite with
`uv run pytest -q`; CI also exercises installed-wheel and Python 3.14
compatibility contracts.

Process-integration cleanup is watchdog-bounded and best effort. POSIX process
and process-group identifiers can be reused, and macOS does not provide an
atomic pidfd-equivalent signaling handle, so abrupt-failure cleanup cannot
guarantee that a late signal still identifies the original process. Run these
tests in an isolated local or CI environment; their process-group assertions
exercise observed behavior, not a strict containment guarantee.

## The validation runner

`whetstone.runner` drives validation runs and records their durable evidence.

A **cell** is one attempt at one `(optimizer, environment)` pair. Its identity
is `optimizer:env:aN`, and the ledger keys resumability on it: a completed cell
is skipped on resume, an interrupted one is re-driven.

**Durable, runner-owned.** `cells.jsonl` and `spend.jsonl` are append-only
JSONL under the run root. Each cell line records the official arm scores, the
paired confidence intervals its status is read off, the accounting, and
references to the artifacts it published. Spend snapshots bracket each cell so
cumulative spend is auditable and the budget guards key off persisted numbers.
Terminal artifacts -- the immutable per-cell viewer directory and the
official-anchor projection -- are fsynced and committed atomically before the
cell line that cites them becomes durable.

**Durable, harness-owned.** The optimization run binding, the ordered step
results, the terminal optimization result, candidates, and evaluation and tool
evidence all live content-addressed in the ObjectStore. The ledger references
them; it never restates them.

**Derived.** Confidence intervals, status strings, and the human-readable
trace are recomputable from the durable evidence and are never authoritative.
`whetstone.runner.refinalize` recomputes a recorded status from a line's own
persisted evidence and appends a corrected line, preserving the original.

**Budget guards.** A canonical cell refuses to start below the reserve, and a
cell whose spend crosses its per-cell stop-loss halts.

### Run lifecycle

The runner owns the DBOS workflow context. The public CLI first invokes its
zero-argument factory, which returns a fully assembled `RunnerLaunch` whose
controllers and GEPA factories are already constructed. After the completed
cell preflight, `register_runtime` registers the proposer transport, mints and
returns a `DurableProposalExecutor`, and registers those preconstructed
capabilities before `DBOS.launch()`. The CLI does not feed the returned executor
back into their construction. This is a current limitation: CLI startup does
not establish that the factory-built controllers and factories use the
executor minted by `register_runtime`.

Each harness-driven run then executes through
`whetstone.coordination.run_workflow` inside exactly one parent workflow, keyed
by the run request's identity hash, so a recovered process resumes that run
rather than starting a second one.

Two consequences are load-bearing. The proposal executor refuses to run outside
a workflow body, so driving steps from the parent satisfies it by construction
and the shared optimization harness stays DBOS-unaware. GEPA's algorithm-owned
runtime is the separate DBOS path. The harness requires its configured replay
policy to equal each adapter's exactly, so the launch factory builds one
harness, and one controller, per optimizer.

`whetstone.runner.startup.register_runtime` is the single registration site.
It binds the proposer transport, returns the durable proposal executor it
mints, and registers the run controllers and GEPA adapter factories supplied by
`RunnerLaunch`. Registration happens strictly before `DBOS.launch()`, because
recovery begins at launch and resolves its dependencies by identity.

### Optimizers the runner drives

Harness-driven optimizers -- MIPROv2, Codex, and identity -- run through
`HarnessRunController` under the shared parent run workflow. GEPA runs through
its own `DbosGepaRunner` parent workflow, which replays a frozen engine run
from ordinal 0; the runner registers GEPA's factories at the same startup site
so both paths share one registration invariant.

MIPROv2 persists the exact optimization run, optimizer configuration,
proposal-executor policy, and proposer-transport durability identity in its
runtime state. Its durable effect budget accounts for task rows alongside
rollouts, proposal calls, and evaluations; the adapter verifies the persisted
ceiling and preflights the next row batch before resolving an Eval Config,
publishing an Evaluation Intent, or invoking a proposal effect. Candidate
assembly uses the run's render contract and rejects literal-replacement input
that would make instruction and demonstration text indistinguishable.

COPRO is not among them. `CoproAdapter.invoke` reports `CONTINUE` on every
successful round and `terminalize` refuses a continuing tail, so a COPRO run
reaches a terminal Optimization Result only once a controller folds each
round's resolved evaluation intents back into the adapter's `attempt_history`
pool. That folding capability has no implementation outside the optimization
layer's own tests, so the runner does not drive COPRO.

### CLI

`whetstone-validate` has three commands. `cell --factory module:callable`
resolves a typed `RunnerLaunch`, registers capabilities, owns the DBOS
lifecycle, and prints the resulting cell line; a completed cell short-circuits
before any runtime is constructed. `status --root <dir>` prints the validated
ledger lines. `refinalize --root <dir> --optimizer … --env … --attempt …`
appends an evidence-only corrected line.

The DBOS system database defaults to a per-cell SQLite file under
`<ledger>/dbos/`, and is overridable by `--dbos-system-database-url` or
`$WHETSTONE_DBOS_SYSTEM_DATABASE_URL`; `--dbos-application-database-url` and
`--dbos-application-version` have matching `WHETSTONE_DBOS_*` variables.

The DBOS application version and the importable model paths referenced by its
checkpoints are part of recovery compatibility. An incompatible package
cutover uses a distinct application version and a fresh system database; the
previous database is archived rather than reused or destroyed.
