# Engineering journal

What was actually done at each milestone, the test results at the time, and the decisions made
(with rejected alternatives). Code was written with AI assistance (Claude Code) and reviewed by
the repository owner at each milestone checkpoint.

## M1 — Requirements and architecture (2026-11-02)

* Wrote `docs/architecture.md`: requirements R1–R7, the three clocks, the `qrp.events.v1` schema,
  storage layout, temporal-leakage rules.
* Projects 1 and 2 (`Low-Latency-Market-Data-Engine-and-Trading-Simulator`,
  `Fault-Tolerant-Distributed-Market-Event-Log`) contain no code yet, so this repository defines
  the event interface they should emit rather than adapting to an existing one.
* Tests: none yet. The CI run for this commit fails because pytest collects no tests (exit 5).

Decisions:

* **Research clock = receive time, not exchange time.** Exchange time is when something happened;
  receive time is when we could have acted on it. Using exchange time for features would leak
  the feed latency. Exchange time is kept for the simulated exchange-side book.
* **One long event table instead of separate book/trade tables.** A single seq-ordered stream
  is what an event-driven consumer reads, and message atomicity (`rows sharing seq`) is explicit.
  Rejected: a wide row with nullable bid/ask/trade columns (ambiguous semantics per row kind).
* **Depth-1 sources are a one-level book, not a separate `quote` kind.** Keeps one code path and
  makes "this dataset has no depth" a manifest capability that features check.

## M2 — Event schema and integrity checks (2026-11-02)

First draft (reviewed, not pushed) assumed every row had a receive time, that `seq` was one
contiguous space per symbol, and planned to fill missing receive times with
`event_time + assumed latency` at import. Review caught that all three are wrong for real feeds:

* Public archives often have no capture time. Writing an assumed one into the data makes it
  indistinguishable from a measured one. **Fix:** `recv_time_ns` and `event_time_ns` are nullable;
  the latency assumption moves to the experiment config (required when receive times are absent)
  and is recorded in the run manifest.
* Venues number channels independently (Binance trade ids vs depth update ids) and some ids skip
  legitimately. **Fix:** a `stream` column; `seq` is scoped to `(symbol, stream)` and each stream
  declares `contiguous`, `monotonic` or `derived` sequencing. Only contiguous streams report gaps.
* Incremental books need an explicit recovery point. **Fix:** a `snapshot` kind whose message is
  the complete book.

Dataset survey (all checked by downloading or probing, 2026-11-02):

* LOBSTER samples: the site is now a single-page app; no direct sample URLs found. Not used.
* Binance `data.binance.vision` futures `bookTicker`: exists for 2024-03-01 (top of book only,
  no capture timestamp). Superseded by the option below.
* **Chosen:** Tardis.dev free first-of-month samples, Binance spot LINKUSDT 2024-03-01,
  `incremental_book_L2` (35.4 MB gz, 4,764,503 rows) and `trades` (2.6 MB gz, 223,569 rows).
  Full-depth L2 with snapshot resets and a genuine collector arrival timestamp
  (`local_timestamp`). Observed: 3 snapshot runs (initial + 2 reconnects), contiguous trade ids,
  snapshot rows whose exchange timestamp equals the arrival timestamp (Tardis's documented
  fallback when the venue sends none — the adapter maps those to null).
  Licence: research/personal use allowed; raw or reformatted data may not be redistributed;
  derived statistics may be published in research reports; models trained on the data may not be
  published. So the repo ships a download script, never the data or fitted coefficients.

Implementation:

* `schemas.py`: canonical columns, `conform()` (strict casts, nothing filled in), manifest models
  with per-stream sequencing and timestamp capabilities.
* `validate.py`: `deduplicate()` drops verbatim re-deliveries (count goes to the manifest);
  `validate_events()` runs 17 checks. Errors: nulls, no timestamp at all, bad codes, undeclared
  stream, non-positive values, remaining duplicates, conflicting level updates, mixed
  snapshot/incremental message, seq or receive time going backwards, `recv < event`,
  out-of-range timestamps. Warnings: seq gaps on contiguous streams, exchange-time regressions,
  latency outliers, partial receive times, feed silence > 60 s.
* Tests: 16 passed, including a Hypothesis property test that the gap count equals the number of
  missing runs for arbitrary deletions from a contiguous stream.

## M3 — Synthetic feed and versioned catalog (2026-11-02)

* `synth.py`: zero-intelligence book with regime-switching arrival rates, multi-level sweeps,
  one packet-loss episode per session plus recovery snapshot. 425k messages/s generated in ~1 s.
* `catalog.py`: import = conform → dedupe → validate (errors block) → partitioned zstd Parquet
  written to a temp directory and renamed into place → manifest → append to `versions.jsonl`.
* Tests: 25 passed (round trip preserves feed order, idempotent re-import, revisions with
  parent + knowledge-time resolution, partition pruning, DuckDB view equals Polars, tampered
  file detected, failed validation leaves nothing behind).

Decisions: the version id hashes the *inputs* plus adapter parameters, not the output bytes,
because Parquet files embed writer metadata that changes with library versions. Rejected:
mutable "latest" partitions with overwrite-on-correction — they make "what did we know on
date D" unanswerable.

## M4 — Book reconstruction (2026-11-02)

* Per-message top-N ladder with `synced`/`crossed`; contiguous-stream gaps invalidate until
  the next snapshot. Tests: 34 passed, including a Hypothesis comparison with a naive
  reference book over random message sequences.
* Design bug caught before commit: the first version dropped trade rows before scanning
  sequence numbers, so in a unified feed every trade-only message would have looked like a gap.
  Trade rows on the book stream now stay in the sequence scan.
* Observation: all 11 crossed quotes in the synthetic feed occur while unsynced (updates applied
  to a book that missed messages). That is the evidence for excluding unsynced rows downstream
  rather than "repairing" them.

## M5 — Availability clock, features, Tardis adapter (2026-11-02)

* `features.py`: availability clock (measured receive time, or exchange time + an explicit
  assumption), 18 features, forward labels; `adapters.read_tardis`.
* Real-data findings: snapshot rows carry Tardis's timestamp fallback (mapped to null); one
  snapshot level has size 0, which the validator wrongly called an error (now a warning);
  reconstruction of 8.9 M rows takes ~5 s.
* Bugs found by tests: `rv` was NaN on flat stretches because incremental rolling sums leave
  tiny negative values (clipped at 0); the truncation-invariance test showed the grid's last
  point depends on when the session's last quote arrives, so the invariant was made precise
  (truncated grid is a prefix with identical values).
* Process error and fix: while checking feature sanity I computed full-day correlations on
  2024-03-01, the day I had intended to split into development and test. That day became
  development-only and a new day (2024-04-01) was downloaded unopened as the test set. Recorded
  in the experiment log.
* Tests: 42 passed, including hand-computed values for every feature family on a ten-row feed.

## M6 — Splits, baselines, statistics (2026-11-02)

* Purged/embargoed walk-forward; zero/mean/ridge models; rank IC, OOS R² vs zero, hit rate,
  calibration, circular block bootstrap, circular-shift nulls. Tests: 51 passed.
* Ridge first used `np.linalg.solve`, which fails on a constant feature; it now solves the
  augmented least-squares problem with `lstsq` (minimum-norm, zero coefficient).
* A property test failed on an off-by-one in its own expectation: a label ending exactly at
  the test start is known at that instant and is correctly admissible.
* Decision: circular-shift nulls instead of permutation nulls, because permutations destroy the
  autocorrelation that makes overlapping-label ICs look significant.

## M7 — Event-driven backtester (2026-11-02)

* Signals, arrivals and acks on one priority queue; book timelines on the availability and
  exchange clocks; IOC fills over visible depth with consumed-liquidity tracking; ledger with
  touch and limit checks and a full recomputation at the end. Tests: 64 passed.
* Bug: an empty equity frame had a null dtype and crashed the summary (typed schema now).
  Three test failures were test-construction mistakes (the last signal of a session is the
  flatten point), not engine bugs.
* Decision: no passive fills and one working order at a time. Both make results worse than a
  real system could achieve, which is the safe direction for research.

## M8 — Runner, CLI, report (2026-11-02)

* `experiment.py`, `cli.py`, `log.py`, `report.py`. First synthetic run: the cost-aware
  strategy never traded (largest forecast 0.7 bp vs 5 bp threshold). Added a diagnostic
  cost-ignoring strategy so the report shows edge, spread, latency and fees separately, and
  moved threshold settings out of the engine config (the engine never used them).

## M9 — Negative controls and end-to-end tests (2026-11-02)

* Seven controls with fixed pass criteria; the Project 1 interface fixture; pipeline tests.
  Tests: 69 passed.
* The shuffled-data controls first passed trivially because selection correctly picked the zero
  forecast (IC exactly 0). They now also refit the run's own model on the destroyed data and
  require its test-IC interval to contain 0.
* The run-determinism test found a real bug: two identical runs wrote calibration means
  differing in the last digits, because Polars' multithreaded group-by mean does not fix the
  summation order. Replaced with `numpy.bincount`; the test then passed in 5/5 repetitions.

## M10 — Real-data experiment E2 (2026-11-02)

* Pre-registered config and decision rules committed (208f906) before the test day was
  evaluated; one run on a clean tree. Result: signal and usability rules pass, tradeability
  fails. Details in `docs/experiment_log.md` and `reports/linkusdt_imbalance.md`.

## M11 — Benchmarks and profiling (2026-11-02)

* `qrp bench`: four cases, each implementation in its own process, results cross-checked.
* Profiling the real pipeline: book reconstruction is ~95% of `prepare`. Removing per-message
  numpy calls gave −23% on synthetic but only −7% on real data; cProfile had inflated the
  per-call overhead, and real messages carry ~7 rows each. Remaining cost is per-row Python
  work (~0.58 µs/row). A compiled implementation was considered and not done (see
  docs/benchmarks.md).

## M12 — Independent review and corrections (2026-11-02)

A separate reviewer agent audited the repository with probe scripts. Confirmed issues and what
was done:

* **Look-ahead without receive times (fixed).** With exchange-time-only data, a message whose
  exchange time was earlier than its predecessor's made the cumulative book state visible
  before a message it already contained. Book availability is now the running maximum in
  message order, and a message carries its last row's timestamp. My causality test could not
  catch this: it used measured receive times, which are monotone by validation.
* **One-sided books counted as valid (fixed).** Empty ladder levels are NaN, and the validity
  check tested for null; a NaN OFI term then wiped out a whole session. Neither committed
  experiment had a one-sided book (feature tables byte-identical before and after both fixes).
* **Holdout discipline not enforced (fixed).** Stage re-runs could re-score the test period
  under an edited config; `qrp features` wrote test labels; the evaluation counter reset with
  a new runs folder. Now: config hash check, dev-only feature export by default, and a
  per-dataset test-access ledger in the catalog (backfilled with the two earlier E2 runs,
  marked as such).
* **Backtest signals depended on labels (fixed).** Signals now cover every complete row.
* **Unsafe symbols, stale versions on adapter change, version-log crash window (fixed).**
* **Truncation checks used raw time and one cut (fixed):** availability time, several cuts.
* **Report rules differed from the pre-registered ones (fixed)**, and docs overstated a few
  guarantees (corrected: README, features.md, data.md, event_interface.md).

Correction to M11: the "−7%" reconstruction speed-up was measured while the reviewer agent was
loading the machine. Interleaved fresh-process measurements showed that version within 2% of
the original and ~600 MB heavier. Replacing its list-of-lists ladder with a flat buffer gave
the real gain: 5.91 s → 3.97 s on the real data, same memory, identical output. The method is
now a committed script (`benchmarks/compare_book_revisions.py`).

E2 was re-run once on the corrected code with the unchanged config (third evaluation of the
test day, recorded in the ledger and the experiment log). All statistical results were
identical; the diagnostic backtest changed by 2 orders.

Tests: 83 passed.
