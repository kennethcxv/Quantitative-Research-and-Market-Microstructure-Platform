# Feature definitions

All features are evaluated on a clock-time grid `t = k·Δ` (Δ = `interval_ms`, default 1 s)
inside each session (UTC date), using only rows whose availability time is `<= t`
(see [architecture.md §3](architecture.md#3-temporal-leakage-rules)). Implementation:
`src/qrp/features.py`; hand-computed checks: `tests/test_features.py::test_hand_computed_values`.

Notation, for the last book message available at `t`: best bid `b`, best ask `a`, sizes
`q_b`, `q_a`, mid `m = (a + b) / 2`, and `D_b^N`, `D_a^N` the total size in the top `N` levels.
A primed symbol refers to the previous book message.

## When a sample is valid

A grid row is `valid` only if the book it reads is synced (a snapshot has been applied since the
last detected gap), not crossed, two-sided (empty ladder levels are NaN and count as missing),
and its last message is at most `max_book_age_s` old. On invalid rows `m` is null, and:

* clock- and event-time returns and the label need **both endpoints** valid; the path between
  them may cross an invalid stretch (the endpoints are trustworthy prices);
* `rv_{W}s` needs **every** step in its window valid;
* window sums (OFI, trade counts and volumes) include whatever arrived in the window; OFI terms
  are zero across unsynced or crossed messages;
* rows that are invalid or miss any feature never reach a model or the backtester.

A book row becomes available only when every message it reflects is available (running
maximum of availability in message order), so out-of-order exchange times on feeds without
receive times cannot leak state early (`test_out_of_order_exchange_times_cannot_leak_book_state`).

## Book state

| Feature | Definition | Notes |
|---------|------------|-------|
| `spread_bps` | `1e4 · (a − b) / m` | On a one-tick market (LINKUSDT) this is nearly constant; it moves mostly with price level. |
| `imbalance_l1` | `(q_b − q_a) / (q_b + q_a)` ∈ [−1, 1] | The classic large-tick predictor: a thin side is more likely to be depleted first. |
| `imbalance_lN` | `(D_b^N − D_a^N) / (D_b^N + D_a^N)` | Requires `book_depth` ≥ N (refused otherwise). Levels beyond the venue's snapshot depth may be stale. |
| `microprice_dev_bps` | `1e4 · (μ − m) / m`, `μ = (a·q_b + b·q_a) / (q_b + q_a)` | Identity: `μ − m = (a − b)·imbalance_l1 / 2`. It adds information beyond imbalance only through the spread. This is the size-weighted mid, not Stoikov's (2018) iterated micro-price. |
| `log_depth_lN` | `log(D_b^N + D_a^N)` | Liquidity proxy; logged because depth is heavy-tailed. |

## Order flow

| Feature | Definition | Notes |
|---------|------------|-------|
| `ofi_{W}s` | `Σ e_n` over book messages with availability in `(t − W, t]`, where `e_n = 1[b ≥ b′]·q_b − 1[b ≤ b′]·q_b′ − 1[a ≤ a′]·q_a + 1[a ≥ a′]·q_a′` | Cont, Kukanov & Stoikov (2014), at the message granularity of the feed. `e_n = 0` when either message is unsynced or crossed. On Binance spot the depth channel is batched every 100 ms, so each `e_n` nets everything that happened within a batch: this is a coarser OFI than one computed from every order event. |
| `trade_count_{W}s` | number of prints in `(t − W, t]` | Trade intensity. A print is one venue trade report. |
| `log_volume_{W}s` | `log(1 + Σ size)` over prints in the window | |
| `trade_imbalance_{W}s` | `Σ side·size / Σ size` over prints in the window; 0 if none | Uses the venue's aggressor flag; prints with unknown side (0) count in volume only. |

Window sums are computed as `C(t) − C(t − W)`, where `C` is a per-session cumulative sum read
with a backward as-of join. That is exact for half-open windows and costs two joins per window
regardless of `W`.

## Returns and volatility

| Feature | Definition | Notes |
|---------|------------|-------|
| `ret_{W}s` | `log m(t) − log m(t − W)` | Clock-time return; null if either end is invalid. |
| `ret_event_K` | `log m_j − log m_{j−K}` where `j` indexes book messages that changed the top of book | Event-time return: K top-of-book changes regardless of how long they took. |
| `rv_{W}s` | `sqrt(Σ r_i²)` over the `W/Δ` grid returns `r_i = log m(t_i) − log m(t_{i−1})` in the window | Realised volatility from Δ-sampled mids. On a one-tick market mid moves are discrete (almost always one tick), so this effectively counts tick moves in the window rather than measuring a continuous volatility. |

## Label

`fwd_ret = log m(t + h) − log m(t)` with `h = horizon_s`, null if either end is invalid or
`t + h` is past the session's last book message. `label_end_ns = t + h` is used for purging
(`splits.py`). With `h = 10 s` and `Δ = 1 s`, consecutive labels share 9 s of their windows; the
bootstrap block length (default `2 · max(h, longest window) / Δ` = 60 samples) and the embargo
(30 s) are set with that overlap in mind.

## What the data cannot support

* No order-level (L3) features: queue position, order age and cancellation hazards need order
  ids, which neither source has.
* Top-of-book-only sources (`book_depth = 1`) get no depth features; the code refuses them.
* Nothing is inferred from OHLCV.
