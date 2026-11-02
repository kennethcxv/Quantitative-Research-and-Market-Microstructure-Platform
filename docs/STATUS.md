# Status

Last updated: 2026-11-02.

## Completed

* M1 architecture and data model (`docs/architecture.md`)
* M2 canonical schema (nullable timestamps, per-stream sequencing, snapshots); 18 integrity
  checks; interface spec (`docs/event_interface.md`)
* M3 synthetic L2 generator; versioned, partitioned Parquet catalog with DuckDB/Polars access
* M4 book reconstruction with causal gap invalidation and snapshot recovery
* M5 availability clock, causal features, labels; Tardis.dev adapter (real data)
* M6 purged walk-forward, baselines, rank IC / OOS R² / calibration / block bootstrap / nulls
* M7 event-driven backtester with latency, fees, spread, slippage, limits, accounting checks
* M8 experiment runner, CLI, JSON logging, mechanical report
* M9 negative controls; Project 1 interface fixture; end-to-end and determinism tests

## Test results

`uv run pytest`: 69 passed. ruff, ruff format, mypy --strict: clean. CI green on main.

## Remaining

1. Real-data experiment E2 (pre-registered in docs/experiment_log.md), report under reports/
2. Benchmarks vs pandas (`qrp bench`): ingestion, partition-pruned query, as-of join, windows
3. README, docs/features.md, docs/data.md, docs/benchmarks.md; journal update
4. Fresh-clone verification; final audit

## Known issues

* None open. Fixed: nondeterministic calibration aggregation (abffd8d).
