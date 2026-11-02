# Data sources

| Dataset | What it is | Licence | In repo? |
|---|---|---|---|
| Synthetic feed (`qrp fixture`) | Zero-intelligence limit order book, full depth, unified contiguous feed with sweeps, packet loss and recovery snapshots | CC0 (generated) | Generated on demand; a 641-row sample is `tests/fixtures/project1_feed.ndjson` |
| Tardis.dev free samples, Binance spot LINKUSDT, 2024-03-01 and 2024-04-01 | Full-depth L2 diff updates (`incremental_book_L2`) and trades | Tardis.dev Licence Agreement (below) | **No.** Download with `qrp fetch-tardis` |

## Tardis.dev samples

Tardis.dev publishes the first day of every month without an API key
(<https://docs.tardis.dev/downloadable-csv-files>). Download and verify:

```
uv run qrp fetch-tardis --exchange binance --symbol LINKUSDT --date 2024-03-01 2024-04-01
uv run qrp import --name tardis_binance_linkusdt --tardis 2024-03-01 2024-04-01
```

`fetch-tardis` checks each file against the SHA-256 recorded in
`qrp.adapters.TARDIS_REFERENCE_SHA256`. If Tardis ever regenerates a file the check fails
rather than silently producing a different dataset version.

| File | Size | Rows |
|---|---|---|
| `binance_incremental_book_L2_2024-03-01_LINKUSDT.csv.gz` | 35.4 MB | 4,764,503 |
| `binance_trades_2024-03-01_LINKUSDT.csv.gz` | 2.6 MB | 223,569 |
| `binance_incremental_book_L2_2024-04-01_LINKUSDT.csv.gz` | 30.1 MB | (test day) |
| `binance_trades_2024-04-01_LINKUSDT.csv.gz` | 2.4 MB | (test day) |

Imported together: 9,287,482 canonical rows, dataset version `34bcccee27a08362`.

**Licence terms that shape this repository** (Tardis.dev Licence Agreement, read 2026-11-02):
research and personal use is permitted (clause 1, "Permitted Use"); the data and anything
merely reformatted from it may not be redistributed (9.2), so no data files, Parquet copies or
event-level extracts are committed; derived statistics may appear in research reports (9.3),
which is what `reports/linkusdt_imbalance.md` contains; models trained on the data may not be
published (9.5), so fitted coefficients stay in the local, git-ignored `runs/` directory. Coinbase
data carries extra restrictions (clause 23) and is deliberately not used.

### Mapping to `qrp.events.v1`

| Tardis column | Canonical | Notes |
|---|---|---|
| `local_timestamp` (µs) | `recv_time_ns` | Arrival time at Tardis's collector. A genuine capture time, but of *their* network position. |
| `timestamp` (µs) | `event_time_ns` | Venue timestamp; **null** where it equals `local_timestamp` to the microsecond, Tardis's documented fallback when the venue sends none (here: REST snapshots). |
| `is_snapshot` | `kind = snapshot` | A run of snapshot rows is one message, so the book is replaced atomically. |
| `side` bid/ask, buy/sell | `side` ±1 | Trades: aggressor side as reported by Binance. |
| `amount` | `size` | Absolute level size (`0` deletes), as in the canonical schema. |
| — | `stream`, `seq` | `depth`: message counter assigned by the adapter (`derived`: the CSV drops Binance's update ids, so loss is undetectable from ids). `trades`: Binance trade id (`contiguous`). |

### What validation and reconstruction found

* 3 snapshot runs on 2024-03-01 and 2 on 2024-04-01: the initial snapshots plus reconnect
  resets. Each reset discards the book, as Tardis documents.
* 8 snapshot levels with size 0 (`empty_snapshot_level` warning). The first version of the
  validator classified these as errors; real data showed they are harmless, so they are now a
  warning and are ignored when the snapshot is applied.
* 12,430 deletes of price levels not in the book. Binance documents this as normal for its diff
  depth stream; it is counted, not treated as loss.
* One `seq_gap` on the trade stream: the month between the two sample days.
* One `stale_feed_gap`: no trades for more than 60 s around 23:42 UTC on 2024-04-01.
* No crossed books and no unsynced messages after reconstruction.
* The depth channel is batched (100 ms on Binance spot) while trades are real-time, so a trade
  can print inside a quote the book has not yet updated: 9,490 of 425,489 prints (2.2%)
  (`book.trades_inside_stale_quote`, reported in every run). The backtester fills against the
  latest exchange-clock depth message at order arrival, not the book the strategy saw when it
  decided, but that message can itself lag the trades by up to one batch, and liquidity those
  trades already took is not removed. That bias is optimistic for a taker and is not measured.

## Synthetic feed

`qrp.synth` simulates messages as a Poisson process whose rate switches between a calm and an
active regime. Each message is one of: a limit order at a geometric distance behind the
opposite best, a cancellation at one of the top levels, or a market order sweeping the
opposite side. One packet-loss episode per session drops 40 messages and is followed by a
recovery snapshot 2 s later. Receive time = event time + log-normal latency (median 400 µs),
kept non-decreasing. Everything derived from it is labelled SYNTHETIC in manifests and reports.
