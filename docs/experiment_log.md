# Experiment log

Every experiment that evaluated a test period is listed here, including failed and null
results. Entries are written before the test evaluation (plan) and completed after (result).

## E1 — Synthetic order-book imbalance (pipeline validation)

* Config: `configs/synthetic_demo.toml`; data: `qrp fixture` (SYNTHETIC).
* Purpose: check that the pipeline recovers a signal the generator is known to contain and that
  negative controls behave. Not evidence about markets.
* Test-period use: the synthetic test day was evaluated many times while the runner and
  report were being debugged (runs were deleted between attempts, so the report's counter of
  earlier evaluations understates this). That is acceptable only because the data is synthetic
  and the generator's signal is known by construction.
* Result (`reports/synthetic_demo.md`): ols_imbalance_l1 selected; test rank IC 0.26 [0.24,
  0.29], OOS R² 0.042 [0.033, 0.053], calibration slope 1.09. Cost-rule strategy (2 bps fee) never
  trades: largest forecast 0.7 bp vs 5 bp round-trip cost. The diagnostic top-20% strategy has
  positive edge before costs (+56) that spread (-324) and fees (-470) turn into -737. All seven
  negative controls pass.

## E2 — LINKUSDT order-book state vs 10 s mid return (real data)

**Plan (committed before the test day was evaluated).**

* Data: Tardis.dev free samples, Binance spot LINKUSDT. Development: 2024-03-01. Test:
  2024-04-01. SHA-256 of all four files pinned in `qrp.adapters.TARDIS_REFERENCE_SHA256`.
* Disclosure: while building the feature code I computed full-day Spearman correlations of
  several features with the 10 s forward return on 2024-03-01 (imbalance_l1 0.26, ofi_5s 0.17,
  ret_5s 0.10, spread_bps 0.03). That day was therefore reassigned to development only and a
  separate, unopened day (2024-04-01) was downloaded as the test set; it was not inspected
  before this entry was committed.
* Hypothesis: top-of-book imbalance and related order-book features have positive rank IC with
  the 10 s forward mid return on the test day; the effect is too small to pay for a taker
  round trip (spread + 10 bps fees each way).
* Protocol: `configs/linkusdt_tardis.toml` unchanged. Walk-forward (5 folds, 30 s embargo) on
  the development day selects among zero, mean, OLS on imbalance_l1 and ridge on all features
  (4 penalties) by mean fold MSE; the selected model is refit on the full development day and
  evaluated once on the test day. Strategy thresholds come from development data only.
* Decision rules: the signal "exists" if the test rank IC 95% block-bootstrap CI excludes 0 and
  the circular-shift p-value < 0.01; it is "usable" if test OOS R² vs zero has a CI above 0; it
  is "tradeable" if the cost-rule strategy's net P&L per minute has a CI above 0.

**Result** (run `20261102T130546.156Z-a0aca6e8`, commit 208f906, clean tree, first and only
test evaluation; full report `reports/linkusdt_imbalance.md`):

* Selected: ridge on all 18 features, alpha = 1 (dev walk-forward R² 0.061 vs 0.060 for OLS on
  imbalance_l1 alone — a negligible difference; the extra features buy almost nothing).
* Signal exists: test rank IC 0.254 [0.241, 0.269]; circular-shift p = 0.005 (the floor for 200
  nulls). **Pass.**
* Usable: test OOS R² vs zero 0.028 [0.022, 0.034]. **Pass**, but half the development value
  (0.058): the fitted scale does not transfer fully across days (test calibration slope 1.14).
* Regimes: rank IC 0.33 / 0.30 / 0.17 in the low / mid / high `rv_30s` terciles; in the
  high-volatility third the R² is 0.012 [0.007, 0.019].
* Tradeable: **fail.** The cost-rule strategy never traded (largest forecast 8.0 bp vs a 20.5 bp
  round-trip cost at 10 bps taker fees). The diagnostic top-20% strategy earned 0.55 bp of edge
  per unit notional before costs, lost 0.88 bp to spread and latency and 10 bp to fees: net
  -11,428 USDT on 11.1 M notional. Break-even taker fee: -0.33 bp, i.e. it would lose even with
  zero fees, because crossing the one-tick spread costs more than the predictable move.
* Negative controls: all seven pass (shuffled/rotated-label ICs within ±0.004 with CIs
  covering 0; canary flagged by truncation; nearest-in-time model IC 0.92 under shuffled K-fold
  vs 0.008 under purged walk-forward).
* Conclusion: top-of-book imbalance is a strong, stable *predictor* of the next 10 s of LINKUSDT
  mid movement and is worthless to a taker. Using it would require earning the spread (passive
  execution), which this simulator deliberately does not model.
