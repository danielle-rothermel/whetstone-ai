# Platform graph workflow implementation notes

The v1 platform graph workflow currently runs one already-created
`PredictionSpecRecord` through DBOS and persists append-only generation and
node-attempt outcomes.

## Running the narrow path

Run one existing prediction spec:

```bash
uv run python -m dr_dspy.platform.worker run-one \
  --database-url "$DATABASE_URL" \
  --prediction-id "<prediction-id>"
```

`run-one` requires a `PredictionSpecRecord` row to exist before it starts. This
phase wires `insert_prediction_spec` in the database layer, but it does not add
a spec-creation CLI or end-to-end fixture command. Create specs through tests,
migration/backfill setup, or ad-hoc insertion before using the documented
runner command.

Start the minimal platform DBOS runtime shell:

```bash
uv run python -m dr_dspy.platform.worker worker \
  --database-url "$DATABASE_URL"
```

The `worker` command launches DBOS with no listened queues. It is a runtime
shell for the direct `run-one` stage, not a production queue-consuming worker
path. Batch submission, fairness, queue consumption, throttle-aware backoff,
scoring, projections, and migration/backfill are deferred.

The CLI currently reuses the legacy `dr_dspy.harness.dbos` bootstrap helpers to
avoid introducing a second DBOS configuration path during this narrow phase.
Before the platform worker grows queue ownership or batch submission, DBOS
runtime setup should move into a shared, non-v0 runtime module.

## Clock steps

Generation start and generation completion use distinct DBOS step names. This
avoids depending on DBOS memoization details for repeated calls to a single
clock step. Node-attempt timestamps are captured inside the node execution step,
where the provider call happens. If DBOS exhausts retries before the node step
returns, the workflow converts the step exception into a terminal node error in
a separate DBOS step.

## Node attempt indexes

Node-attempt persistence records one terminal outcome for each invoked node in a
generation run. DBOS retries happen inside the node execution step and do not
create separate node-attempt rows. Until explicit node reattempt workflows are
added, each invoked node is persisted with `attempt_index=0`.

## Provider config scope

The runtime provider config is reconstructed from the fields currently stored in
`ProviderConfigRef`: provider kind, endpoint kind, model, throttle key, and
request parameters. Custom provider runtime fields such as `base_url`,
`api_key_env`, and capability flags are not spec-owned yet; adding those belongs
in a later provider-config contract change.

## Follow-up notes

- Replace prompt metadata keys such as `user_prompt_template`, `system_prompt`,
  and `provider_config_id` with typed graph/spec fields once the graph contract
  is ready for another breaking change.
- Move database engine/pool ownership into the platform worker runtime instead
  of creating short-lived SQLAlchemy engines inside each DBOS step.
- Move DBOS bootstrap ownership out of `dr_dspy.harness.dbos` and into a shared
  runtime module before platform queue workers are added.
- Add a supported spec-creation path for v1 runs, either as a CLI helper or a
  standard integration-test fixture.
- Extend the persisted provider config contract before allowing experiments to
  vary provider runtime details such as `base_url`, `api_key_env`, or capability
  flags from specs.

## Integration-test status

The default test suite covers the pure graph orchestration, node execution,
record conversion, idempotent persistence statement shape, and worker import.
It does not require a live DBOS system database. A narrow live DBOS/Postgres
workflow test should be added once the project has a standard integration-test
fixture for DBOS.
