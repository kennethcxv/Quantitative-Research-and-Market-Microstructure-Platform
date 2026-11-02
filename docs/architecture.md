# Architecture and research requirements

This document is the design contract for the platform. Code that violates it is a bug.

## 1. What the system is for

Answer narrow questions of the form *"does quantity X, observable at time t, predict the
mid-price change over (t, t+h], and does that survive realistic execution costs?"* — and make
it hard to fool ourselves while doing so. Profitability is not a goal; correctly separating
evidence from noise is.

Requirements that drive the design:

| # | Requirement | Where it is enforced |
|---|-------------|----------------------|
| R1 | A strategy may only use data that would have been observable at decision time. | `recv_time_ns` clock, as-of joins, `MarketView` clock guard, causality tests |
| R2 | Every result is traceable to an immutable dataset version, config and code revision. | content-addressed dataset versions, run manifests |
| R3 | Model selection never touches the holdout test period. | `experiment.py`: selection receives only development folds |
| R4 | Overlapping labels never leak across train/test boundaries. | purged + embargoed walk-forward (`splits.py`) |
| R5 | Simulated fills are conservative and visibly separate from observed market data. | `backtest/execution.py`, `Fill.simulated` |
| R6 | Features are only computed when the dataset supports them. | dataset capabilities (`book_depth`, `has_trades`) checked in `features.py` |
| R7 | The evaluation can demonstrably detect leakage and noise. | negative controls (`controls.py`) with pass/fail criteria |

## 2. Data model

### 2.1 Three clocks

| Clock | Column / field | Meaning | Used for |
|-------|----------------|---------|----------|
| Event time | `event_time_ns` | Exchange timestamp of the matching-engine event | exchange-side book for simulated fills |
| Receive time | `recv_time_ns` | When our gateway observed the message | **all research**: features, labels, decisions |
| Knowledge time | manifest `created_at_ns` | When this version of the dataset entered the catalog | revisions; "which data did we have on date D" |

`recv_time_ns >= event_time_ns` is a validation rule. When a source has no capture
timestamp (e.g. public exchange archives) the adapter sets `recv_time_ns = event_time_ns +
assumed_latency` and records `recv_time_measured = false` in the manifest; reports surface that.

### 2.2 Canonical event schema `qrp.events.v1`

One row per price-level update or trade print. Rows that share `(symbol, seq)` are one feed
message and are applied atomically (a market order that sweeps three levels is one message with
trade rows and book rows).

| Column | Type | Semantics |
|--------|------|-----------|
| `symbol` | string | instrument id |
| `seq` | int64 | feed message sequence number, contiguous per symbol |
| `event_time_ns` | int64 | exchange time, UTC epoch ns |
| `recv_time_ns` | int64 | capture time, UTC epoch ns |
| `kind` | string | `book` or `trade` |
| `side` | int8 | book: `+1` bid, `-1` ask. trade: aggressor `+1` buy, `-1` sell, `0` unknown |
| `price` | float64 | price level / trade price |
| `size` | float64 | book: new **total** size at that level (`0` deletes it). trade: traded quantity |

Dataset-level facts live in the manifest, not in rows: source and license, `synthetic` flag,
`book_depth` (number of levels the feed exposes, `null` = full depth), `recv_time_measured`,
`has_trades`. A depth-1 (top-of-book) source is represented exactly as a one-level book; depth
features are refused for it. We never fabricate depth from OHLCV or quotes.

This schema is the versioned interface for events produced by Project 1 (market-data engine)
and stored by Project 2 (event log); see [`event_interface.md`](event_interface.md).

### 2.3 Storage layout

```
data/catalog/<dataset>/
  versions.jsonl                      append-only log: version, created_at_ns, parent, note
  <version>/manifest.json             source, license, capabilities, file hashes, validation summary
  <version>/events/symbol=S/date=D/   hive-partitioned Parquet (zstd), sorted by seq
```

`version` = sha256 of (raw input bytes, adapter name, adapter parameters, schema version).
Re-importing identical input yields the same version; a corrected input yields a new version
that records `parent_version` and a revision note. Old versions are never modified.

## 3. Temporal-leakage rules

1. **One research clock.** Decisions happen at sample times `t` on the receive clock. A feature
   at `t` may use only rows with `recv_time_ns <= t` (backward as-of joins; window sums over
   `(t-W, t]`).
2. **Labels are future by construction and carry their end time.** `y_t = log mid(t+h) - log
   mid(t)` with `label_end_ns = t + h`. A label is "known" only at `label_end_ns`.
3. **Purging.** A training sample is admissible for a test block starting at `s` only if
   `label_end_ns + embargo <= s`. Walk-forward folds only train on the past.
4. **Holdout discipline.** Data are split into a development period and a final test period.
   Walk-forward folds inside development select the model and its hyper-parameters; the frozen
   choice is evaluated on the test period exactly once.
5. **Causality tests.** Feature code and the backtester are tested by *truncation invariance*:
   outputs at times `<= T` must be identical whether or not data after `T` exists.
6. **Execution latency.** An order decided at `t` reaches the exchange at
   `t + decision_latency + order_latency` and fills against the book reconstructed from events with
   `event_time_ns` at or before that arrival — never against the book the strategy saw.

## 4. Components

```
adapters (qrp-v1 parquet/csv/ndjson, binance, synthetic generator)
    -> validate.py      integrity checks, severity-ranked report
    -> catalog.py       versioned, partitioned Parquet + manifest
    -> book.py          L2 reconstruction -> per-message top-of-book / top-N ladder
    -> features.py      causal microstructure features on a clock-time grid
    -> labels.py        forward mid returns with label_end_ns
    -> splits.py        dev/test split, purged + embargoed walk-forward
    -> models.py        zero / mean / OLS / ridge baselines
    -> stats.py         rank IC, OOS R^2, calibration, block bootstrap, circular-shift nulls
    -> backtest/        event loop, conservative execution, accounting invariants
    -> controls.py      negative controls and leakage canaries
    -> report.py        markdown research report
cli.py ties these into reproducible commands; experiment.py writes run manifests.
```

## 5. Deliberate non-goals (v1)

* **Passive (limit) order fills.** Queue position is unknowable from L2 data; simulating maker
  fills would overstate performance. Only marketable orders are simulated.
* **Market impact on later prices.** Our simulated orders do not alter the historical book;
  impact is approximated by a slippage term. Results are therefore optimistic for large size.
* **Cross-asset portfolios, ML beyond linear baselines.** Added only if a baseline result
  justifies it.
