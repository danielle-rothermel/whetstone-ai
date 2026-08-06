# Changelog

All notable changes to Whetstone are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add code-grounded repository terms and binding contracts under `.defs/`,
  with a client-side page that renders both authoritative TOML files.
- Add isolated PostgreSQL/DBOS, process, SQLite-time, and SQLite-contention CI
  lanes alongside installed-wheel smoke and Python 3.14 contract checks.
- Expand deterministic coverage for recovery, conflict, cache-accounting,
  optimization-adapter, and real-environment behavior.

### Changed

- Reorganize production and test packages around the canonical core,
  experiment, environment, provider, execution, evaluation, optimization, and
  coordination boundaries, with a hard cutover to the new imports.
- Rewrite the README and package metadata around the current system,
  repository boundaries, and complete local test entrypoints.
- Preserve structured prompt identity and immutability, use canonical UTC
  partial timestamps, and apply one character-budget rounding rule.

### Fixed

- Make immutable JSON values safe for DBOS checkpoint serialization and keep
  DBOS tests independent of registry teardown order.
- Bind proposal transport atomically and strengthen durable recovery,
  contention, partial-log, and prompt-cache accounting behavior.

### Removed

- Remove retired audit, design, planning, and research artifacts together with
  stale package paths and nonessential module/package prose.
