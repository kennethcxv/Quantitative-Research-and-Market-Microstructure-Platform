# Status

Last updated: 2026-11-02.

## Completed

* M1 architecture and data model · M2 schema, per-stream sequencing, integrity checks (19) ·
  M3 synthetic feed and versioned catalog · M4 book reconstruction · M5 availability clock,
  features, Tardis adapter · M6 splits, baselines, statistics · M7 backtester · M8 runner, CLI,
  report · M9 negative controls, interface fixture, end-to-end tests · M10 real-data experiment
  E2 · M11 benchmarks and profiling · M12 independent review and corrections · documentation

## Verified

* `uv run pytest`: 83 passed; ruff, ruff format, mypy --strict (src): clean.
* CI (GitHub Actions, Python 3.11 and 3.12): tests plus the synthetic end-to-end demo with a
  check that every negative control passed.
* Fresh clone: install from the lockfile, tests, demo, and real-data reproduction of E2 (same
  dataset version, identical report) — see docs/experiment_log.md.

## Known limitations

See README "Limitations": no passive fills or persistent impact; one real test day; vendor
capture times; 100 ms depth batching (stale-depth fill bias not measured); Python book
reconstruction (~2.2 M rows/s).

## Next steps if work continues

1. A second instrument and more test days (first-of-month samples) to measure day-to-day
   variation of the signal.
2. Measure the stale-depth fill bias: compare each simulated fill with trades printed between
   the depth batch and the order's arrival.
3. A compiled book reconstructor (or consume Project 1's C++ engine output directly).
