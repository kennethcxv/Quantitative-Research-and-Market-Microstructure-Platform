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
