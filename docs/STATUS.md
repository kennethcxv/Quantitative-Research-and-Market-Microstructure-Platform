# Status

Last updated: 2026-11-02 (M2 revised).

## Completed

* M1 architecture and data model (`docs/architecture.md`)
* M2 canonical schema with nullable timestamps and per-stream sequencing; integrity checks
  (`src/qrp/schemas.py`, `src/qrp/validate.py`); interface spec (`docs/event_interface.md`)

## Test results

`uv run pytest`: 16 passed. ruff, ruff format, mypy --strict: clean.

## Remaining (in order)

1. Synthetic L2 feed generator (openly redistributable fixture) and versioned Parquet catalog
2. Book reconstruction with gap invalidation, snapshot recovery, crossed/stale-quote checks
3. Availability clock, causal features, labels, truncation-invariance tests
4. Purged walk-forward splits, baselines, rank IC / calibration / block bootstrap, regimes
5. Config-driven experiment runner with run manifests; CLI
6. Event-driven backtester: latency, fees, spread, slippage, limits, accounting invariants
7. Negative controls and leakage canaries
8. Tardis adapter (real data), qrp-v1 adapter + Project 1 fixture + integration test
9. Benchmarks vs pandas (CPU, memory, latency, sizes)
10. README, research report, fresh-clone verification, final audit

## Known issues

* The CI run for commit dfd67c3 failed: pytest collected no tests. Fixed by the next commit.
