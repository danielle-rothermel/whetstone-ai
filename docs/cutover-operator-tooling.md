# Cutover operator tooling

`whetstone-cutover` is dry-run by default. It never fetches provider prices or
dispatches work. Mutating store commands require both `--execute` and an exact
`--confirm RUN_ID`.

## Estimates

Create a reviewed version-1 price book with one entry per locked model. Each
entry explicitly records input/output USD per million tokens and the versioned
assumed input/output token envelope. Then:

```json
{
  "schema_version": 1,
  "effective_at": "2026-07-13T00:00:00Z",
  "currency": "USD",
  "assumptions_version": "legacy-token-envelope-v1",
  "source": "operator-reviewed provider price snapshot",
  "review_id": "acceptance-price-review-171",
  "source_document_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "token_evidence_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "models": {
    "provider/model": {
      "input_usd_per_million": "0.10",
      "output_usd_per_million": "0.20",
      "assumed_input_tokens": 100,
      "assumed_output_tokens": 200
    }
  }
}
```

Prices are JSON strings deliberately: floating-point JSON numbers are rejected
at the validation boundary. Every price and token assumption must be positive,
the reviewed token/source evidence digests are embedded in the artifact, and
the complete estimate must remain within the locked `$1.54`–`$4.62` envelope.

```sh
whetstone-cutover estimates generate /private/tmp/platform-v6-live-sweep-161 \
  --price-book /absolute/operator/prices-v1.json \
  --output /absolute/operator/cell-estimates.json
whetstone-cutover estimates generate /private/tmp/platform-v6-live-sweep-161 \
  --price-book /absolute/operator/prices-v1.json \
  --output /absolute/operator/cell-estimates.json --execute
whetstone-cutover estimates validate /private/tmp/platform-v6-live-sweep-161 \
  --artifact /absolute/operator/cell-estimates.json
```

## Fresh stores and bound commands

Configure base admin URLs only in `DATABASE_URL`, `MOTHERDUCK_DATABASE_URL`,
and `NEON_DATABASE_URL`. The descriptor contains names and ownership facts,
never URLs or credentials.

```sh
whetstone-cutover stores prepare --run-id acceptance_171 \
  --descriptor /absolute/operator/stores.json
whetstone-cutover stores prepare --run-id acceptance_171 \
  --descriptor /absolute/operator/stores.json \
  --execute --confirm acceptance_171
whetstone-cutover stores verify --descriptor /absolute/operator/stores.json
whetstone-cutover stores run --descriptor /absolute/operator/stores.json -- \
  uv run whetstone-live-sweep submit-canary \
  /private/tmp/platform-v6-live-sweep-161 --execute \
  --ledger /absolute/operator/live-sweep.sqlite3 \
  --estimates /absolute/operator/cell-estimates.json
```

`submit-remaining` reserves one bounded page at a time. It reconciles durable
provider cost and requires a stable terminal lifecycle for every cell in that
page before reserving the next; unknown or unobserved cost stops dispatch.

Cleanup is fail-closed and ownership-checked:

```sh
whetstone-cutover stores cleanup --descriptor /absolute/operator/stores.json
whetstone-cutover stores cleanup --descriptor /absolute/operator/stores.json \
  --execute --confirm acceptance_171
whetstone-cutover stores verify-cleanup \
  --descriptor /absolute/operator/stores.json
```

If preparation crashes before the descriptor is committed, the journal already
contains the complete recovery descriptor. Cleanup and verification can use it
directly:

```sh
whetstone-cutover stores cleanup \
  --journal /absolute/operator/stores.json.journal.json \
  --execute --confirm acceptance_171
whetstone-cutover stores verify-cleanup \
  --journal /absolute/operator/stores.json.journal.json
```

Each schema and the DBOS SQLite database carries an immutable run ID and
descriptor-digest marker. Binding and cleanup verify those markers; a missing or
replacement marker stops the operation before any resource is deleted.

Hosted parity is the release gate. Record the returned run ID; do not select a
run by recency when concurrent dispatch is possible:

```sh
gh workflow run release-parity.yml --repo drothermel/unitbench --ref main
gh run watch RUN_ID --repo drothermel/unitbench --exit-status
```

Before Vercel production promotion, record the project ID, current production
deployment ID/URL, new reviewed deployment ID/URL, and fingerprints of both
`ANALYSIS_DATABASE_URL` and `DATABASE_URL`. Promotion and rollback remain owner
operations: rollback must restore both old store values and the old deployment
together; mixed Analysis/Detail generations are forbidden.
