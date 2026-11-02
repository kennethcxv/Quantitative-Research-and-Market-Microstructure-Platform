# linkusdt_imbalance

Run `20261102T130546.156Z-a0aca6e8` · dataset `tardis_binance_linkusdt@34bcccee27a08362` · commit `208f906c1b`

## Data

* Source: Tardis.dev samples: binance LINKUSDT 2024-03-01, 2024-04-01
* Licence: Tardis.dev Licence Agreement. Research/personal use. Raw or reformatted data may not be redistributed (clause 9.2); derived statistics may appear in research reports (9.3); models trained on the data may not be published (9.5). Not included in this repository.
* Notes: recv_time = Tardis collector arrival time (their network position, not ours). Depth messages are numbered by the adapter: the CSV omits venue update ids, so loss is only visible through snapshot resets. The venue's REST snapshot caps initial depth (~1000 levels/side); levels far from the touch may be incomplete.
* Book depth: full; trades: yes; receive times: measured; exchange times: partial
* Stream sequencing: `depth` derived, `trades` contiguous
* Validation warnings: empty_snapshot_level=8, seq_gap=1, stale_feed_gap=1
* Book reconstruction: {'messages': 1292245, 'snapshots': 5, 'gaps': 0, 'unsynced_messages': 0, 'crossed_messages': 0, 'deletes_of_unknown_level': 12430}
* Rows: {'grid': 172674, 'valid': 172674, 'dev': 86297, 'test': 86297}

## Features

| feature | definition |
|---|---|
| `spread_bps` | 1e4 * (a - b) / m |
| `imbalance_l1` | (qb - qa) / (qb + qa); in [-1, 1] |
| `imbalance_l5` | same as imbalance_l1 with sizes summed over the top 5 levels |
| `microprice_dev_bps` | 1e4 * (micro - m) / m, micro = (a*qb + b*qa) / (qb + qa). Note micro - m = spread * imbalance_l1 / 2, so this is imbalance scaled by the spread |
| `log_depth_l5` | log of total size in the top 5 levels, both sides |
| `ret_event_20` | log m now - log m 20 top-of-book changes ago (event clock) |
| `ofi_5s` | order-flow imbalance (Cont, Kukanov & Stoikov 2014) summed over messages in (t-5s, t]: e = 1[b>=b'] qb - 1[b<=b'] qb' - 1[a<=a'] qa + 1[a>=a'] qa', primes = previous message; 0 across unsynced or crossed books |
| `trade_count_5s` | number of prints in (t-5s, t] |
| `log_volume_5s` | log(1 + traded quantity in (t-5s, t]) |
| `trade_imbalance_5s` | (buy volume - sell volume) / volume in (t-5s, t], aggressor side; 0 if none |
| `ret_5s` | log m(t) - log m(t-5s) (clock time) |
| `rv_5s` | sqrt(sum of squared 1000ms log-mid returns over the last 5s) |
| `ofi_30s` | order-flow imbalance (Cont, Kukanov & Stoikov 2014) summed over messages in (t-30s, t]: e = 1[b>=b'] qb - 1[b<=b'] qb' - 1[a<=a'] qa + 1[a>=a'] qa', primes = previous message; 0 across unsynced or crossed books |
| `trade_count_30s` | number of prints in (t-30s, t] |
| `log_volume_30s` | log(1 + traded quantity in (t-30s, t]) |
| `trade_imbalance_30s` | (buy volume - sell volume) / volume in (t-30s, t], aggressor side; 0 if none |
| `ret_30s` | log m(t) - log m(t-30s) (clock time) |
| `rv_30s` | sqrt(sum of squared 1000ms log-mid returns over the last 30s) |

## Model selection (walk-forward on the development period only)

| model | folds | mean OOS R² vs zero | mean rank IC | folds beating zero |
|---|---|---|---|---|
| zero | 5 | 0.00000 | 0.0000 | 0 |
| mean | 5 | 0.00011 | 0.0000 | 4 |
| ols_imbalance_l1 | 5 | 0.05955 | 0.2731 | 5 |
| ridge_all[alpha=0.001] | 5 | 0.05742 | 0.2674 | 5 |
| ridge_all[alpha=0.01] | 5 | 0.05785 | 0.2678 | 5 |
| ridge_all[alpha=0.1] | 5 | 0.06035 | 0.2700 | 5 |
| ridge_all[alpha=1] | 5 | 0.06105 | 0.2731 | 5 |

Selected by lowest mean fold MSE: **ridge_all[alpha=1]**. Test-period evaluations of this experiment before this run: 0.

## Out-of-sample results

Intervals are 95% circular block bootstrap intervals (block = 60 samples); p-values are against 200 circular-shift nulls, so the smallest attainable p is 0.005.

| split | n | rank IC [95% CI] | OOS R² vs zero [95% CI] | hit rate | calib. slope | null p |
|---|---|---|---|---|---|---|
| dev walk-forward | 71914 | 0.2699 [0.2565, 0.2823] | 0.05782 [0.05229, 0.06357] | 0.617 | 1.248 | 0.005 |
| test (once) | 86297 | 0.2544 [0.2413, 0.2692] | 0.02768 [0.02183, 0.03439] | 0.619 | 1.139 | 0.005 |

Test results by `rv_30s` tercile (cut points from development data):

| regime | n | mean |ret| | rank IC [95% CI] | OOS R² [95% CI] |
|---|---|---|---|---|
| low | 33007 | 1.53e-04 | 0.3265 [0.3049, 0.3507] | 0.07601 [0.06468, 0.08756] |
| mid | 22995 | 2.13e-04 | 0.3014 [0.2755, 0.3258] | 0.06812 [0.05688, 0.07938] |
| high | 30295 | 3.83e-04 | 0.1742 [0.1535, 0.1933] | 0.01232 [0.00691, 0.01906] |

Test calibration (forecast deciles):

| decile | n | mean forecast | mean outcome |
|---|---|---|---|
| 0 | 8629 | -9.89e-05 | -1.30e-04 |
| 1 | 8630 | -7.65e-05 | -9.13e-05 |
| 2 | 8630 | -5.81e-05 | -5.97e-05 |
| 3 | 8630 | -3.69e-05 | -3.78e-05 |
| 4 | 8629 | -1.36e-05 | -2.24e-06 |
| 5 | 8630 | 9.48e-06 | 1.56e-07 |
| 6 | 8630 | 3.25e-05 | 3.09e-05 |
| 7 | 8630 | 5.35e-05 | 4.24e-05 |
| 8 | 8630 | 7.27e-05 | 7.96e-05 |
| 9 | 8629 | 9.83e-05 | 1.18e-04 |

## Simulated execution

Simulated marketable orders against the reconstructed book at arrival time; no passive fills, no persistent market impact. Fills below are simulations, not historical trades.

Strategies (thresholds on the forecast, set from development data): `cost_rule` = 2.050e-03, `diagnostic_top20pct` = 8.567e-05. `cost_rule` trades only when the forecast exceeds the estimated round-trip cost; diagnostic strategies ignore costs on purpose, to show where the edge goes.

| split | strategy | orders | fills | notional | edge at decision | latency cost | spread+slippage | fees | net P&L | break-even fee (bps) | P&L/min [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dev_oos | cost_rule | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | - | - |
| dev_oos | diagnostic_top20pct | 11609 | 25386 | 11419583 | 664.39 | 5.94 | 831.70 | 11419.58 | -11592.83 | -0.15 | -9.668 [-10.512, -8.844] |
| test | cost_rule | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | - | - |
| test | diagnostic_top20pct | 12167 | 27362 | 11066297 | 609.21 | 7.05 | 963.69 | 11066.30 | -11427.83 | -0.33 | -7.946 [-8.819, -7.064] |

Costs: {'taker_fee_bps': 10.0, 'slippage_bps': 0.0, 'impact_bps_per_l1': 0.0}; latencies (ms): decision 1.0, order 5.0, ack 5.0 (assumptions, not measurements); position size 50.0, limit 50.0.

## Negative controls

| control | expectation | observed | passed |
|---|---|---|---|
| shuffled_labels | selection returns no signal; forced model's test IC CI contains 0 | selected mean (IC 0.0000); ridge_all[alpha=1] IC -0.0036 [-0.0102, 0.0033] | yes |
| shuffled_features | selection returns no signal; forced model's test IC CI contains 0 | selected zero (IC 0.0000); ridge_all[alpha=1] IC 0.0028 [-0.0040, 0.0085] | yes |
| time_rotated_labels | selection returns no signal; forced model's test IC CI contains 0 | selected zero (IC 0.0000); ridge_all[alpha=1] IC 0.0023 [-0.0151, 0.0227] | yes |
| leakage_canary | truncation test flags only the canary | flagged real=[], canary run=['leak_canary']; canary test IC 0.996 | yes |
| label_overlap | shuffled K-fold IC >> purged walk-forward IC | nearest-in-time model: shuffled 5-fold IC 0.921, purged walk-forward IC 0.008 | yes |
| random_signal_backtest | net P&L < 0 (pays costs, no edge) | edge -5.25, costs 28212.24, net -28217.49 | yes |
| null_strategy | no fills, P&L exactly 0 | fills 0, P&L 0.0 | yes |

## Conclusions (mechanical)

* Test rank IC distinguishable from zero (CI excludes 0): yes
* Beats the zero forecast on test squared error (R² CI above 0): yes
* Cost-rule strategy did not trade on test: the largest forecast (7.99e-04) never exceeded the cost threshold (2.05e-03)
* `diagnostic_top20pct` on test: edge before costs 609.21, net -11427.83 after 970.74 spread/latency and 11066.30 fees

## Reproduce

```
qrp run --config <this run's config.toml>   # dataset version 34bcccee27a08362
```

Provenance: `{"git_commit": "208f906c1b99e54831d4cbd9ffcbe5a28492149d", "git_dirty": false, "source_sha256": "d236b9fa6218761d9063c4ad5f6769771eeca52a0cf62b7479df5852299975f3", "python": "3.12.14", "platform": "macOS-27.0.1-arm64-arm-64bit", "packages": {"qrp": "0.1.0", "numpy": "2.5.4", "polars": "2.1.0", "pyarrow": "26.0.0", "duckdb": "2.0.0", "pydantic": "2.14.2"}}`

<details><summary>config.toml</summary>

```toml
# Real data: Binance spot LINKUSDT, Tardis.dev free samples (download with `qrp fetch-tardis`;
# not redistributable, see docs/data.md). Pre-registered in docs/experiment_log.md (E2) before
# the test day was evaluated.
name = "linkusdt_imbalance"
seed = 20240301
description = """
Does order-book state (imbalance, OFI, trade flow, recent returns) observable at t predict the
10 s log mid return of LINKUSDT on a held-out day, and does any of it survive taker execution?
"""

[data]
catalog = "data/catalog"
dataset = "tardis_binance_linkusdt"
version = "latest"
symbol = "LINKUSDT"
dev_dates = ["2024-03-01", "2024-03-01"]
test_dates = ["2024-04-01", "2024-04-01"]

[timing]
extra_latency_ms = 0.0  # research clock = Tardis collector arrival time

[features]
interval_ms = 1000
windows_s = [5, 30]
depth_levels = 5
horizon_s = 10
warmup_s = 60
max_book_age_s = 60.0

[evaluation]
n_folds = 5
embargo_s = 30
n_boot = 500
n_null = 200
regime_feature = "rv_30s"

[[models]]
name = "zero"
kind = "zero"

[[models]]
name = "mean"
kind = "mean"

[[models]]
name = "ols_imbalance_l1"
kind = "ridge"
features = ["imbalance_l1"]

[[models]]
name = "ridge_all"
kind = "ridge"
features = "all"
alphas = [0.001, 0.01, 0.1, 1.0]

[strategy]
threshold_cost_multiple = 1.0
diagnostic_quantile = 0.8

[backtest]
position_size = 50   # LINK, roughly 1,000 USDT
max_position = 50
decision_latency_ms = 1.0
order_latency_ms = 5.0
ack_latency_ms = 5.0
max_book_age_s = 5.0

[backtest.costs]
taker_fee_bps = 10.0   # Binance spot regular-tier taker fee (0.1%)
```

</details>
