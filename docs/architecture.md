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
| R1 | A strategy may only use data that would have been observable at decision time. | availability clock, as-of joins, clock-guarded market view, causality tests |
| R2 | Every result is traceable to an immutable dataset version, config and code revision. | content-addressed dataset versions, run manifests |
| R3 | Model selection never touches the holdout test period. | `experiment.py`: selection receives only development folds |
| R4 | Overlapping labels never leak across train/test boundaries. | purged + embargoed walk-forward (`splits.py`) |
| R5 | Simulated fills are conservative and visibly separate from observed market data. | `backtest/execution.py`, `Fill.simulated` |
| R6 | Features are only computed when the dataset supports them. | dataset capabilities (`book_depth`, `has_trades`) checked in `features.py` |
| R7 | The evaluation can demonstrably detect leakage and noise. | negative controls (`controls.py`) with pass/fail criteria |

## 2. Data model

### 2.1 Timestamps

| Time | Where | Meaning |
|------|-------|---------|
| Exchange time | `event_time_ns` (nullable) | Timestamp the venue put on the message. Null when the venue gives none. |
| Receive time | `recv_time_ns` (nullable) | When a capture process actually observed the message. Null when nobody captured it. |
| Knowledge time | manifest `created_at_ns` | When this dataset version entered the catalog; governs revisions. |
| Availability time | computed per experiment, never stored | When the research strategy is allowed to see a row. |

Availability time is the only clock used for features, labels and decisions:

* if the dataset has measured receive times: `available = recv_time_ns + extra_latency`
* if it does not: `available = event_time_ns + assumed_feed_latency`, and the experiment config
  **must** set `assumed_feed_latency_ms` explicitly — there is no default.

Stored data therefore never contains an invented timestamp; every assumption lives in a config
file that is copied into the run manifest and printed in the report. Receive times measured by
a data vendor's collector describe *that* collector's network position, not ours, so no result
in this repository is a claim about achievable live latency.

### 2.2 Canonical event schema `qrp.events.v1`

One row per price-level update, snapshot level or trade print, specified in
[`event_interface.md`](event_interface.md). The points that matter for correctness:

* **Streams and sequencing.** Each `(symbol, stream)` has its own `seq` space, and the manifest
  declares whether that space is `contiguous` (a jump means lost messages), `monotonic` (jumps are
  normal, e.g. ids shared with other channels) or `derived` (the adapter numbered messages; loss
  is undetectable from ids). Validation only reports missing data where the protocol can prove it.
* **Messages are atomic.** Rows that share `(symbol, stream, seq)` are one message and are applied
  together; the book is only observed between messages.
* **Snapshots reset state.** A `snapshot` message is the complete book. Incremental `book` rows
  after a detected gap are not trusted until the next snapshot (see `book.py`).
* **Capabilities.** `book_depth` (levels per side; `null` = full depth), `has_trades`, and the
  presence of receive times are dataset facts that features check. A top-of-book source is a
  depth-1 book of snapshot messages; depth features are refused for it.

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

1. **One research clock.** Decisions happen at sample times `t` on the availability clock. A
   feature at `t` may use only rows with `available_time_ns <= t` (backward as-of joins; window
   sums over `(t-W, t]`).
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
   `t + decision_latency + order_latency` and fills against the book state that existed on the
   exchange clock at that arrival time — never against the book the strategy saw when it
   decided. The strategy learns of the fill `ack_latency` later and has at most one working
   order. All latencies are configured assumptions.

## 4. Components

```
adapters.py         qrp-v1 Parquet/CSV/NDJSON + sidecar, Tardis.dev CSV; synth.py generator
    -> validate.py      18 integrity checks, severity-ranked report, deduplication
    -> catalog.py       immutable content-addressed versions, partitioned Parquet + manifest
    -> book.py          L2 reconstruction -> per-message top-N ladder, synced/crossed flags
    -> features.py      availability clock, causal features, forward-return labels
    -> splits.py        purged + embargoed walk-forward
    -> models.py        zero / mean / OLS / ridge baselines
    -> stats.py         rank IC, OOS R^2, calibration, block bootstrap, circular-shift nulls
    -> experiment.py    config-driven run: select on dev, test once, evaluate, backtest
    -> backtest/        event loop, conservative execution, accounting invariants
    -> controls.py      negative controls and leakage canaries
    -> report.py        markdown research report with mechanical conclusions
cli.py exposes each stage as a command; log.py emits one JSON object per log line.
```

## 5. Deliberate non-goals (v1)

* **Passive (limit) order fills.** Queue position is unknowable from L2 data; simulating maker
  fills would overstate performance. Only marketable orders are simulated.
* **Market impact on later prices.** Our simulated orders do not alter the historical book;
  impact is approximated by a slippage term. Results are therefore optimistic for large size.
* **Cross-asset portfolios, ML beyond linear baselines.** Added only if a baseline result
  justifies it.
