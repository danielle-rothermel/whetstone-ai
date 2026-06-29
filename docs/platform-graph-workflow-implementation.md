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

Start the minimal platform worker process:

```bash
uv run python -m dr_dspy.platform.worker worker \
  --database-url "$DATABASE_URL"
```

This entrypoint intentionally does not add batch submission, fairness,
throttle-aware backoff, scoring, projections, or migration/backfill.

## Clock steps

Generation start, generation completion, node-attempt fallback start, and
node-attempt fallback completion each use distinct DBOS step names. This avoids
depending on DBOS memoization details for repeated calls to a single clock step.

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
- Extend the persisted provider config contract before allowing experiments to
  vary provider runtime details such as `base_url`, `api_key_env`, or capability
  flags from specs.
