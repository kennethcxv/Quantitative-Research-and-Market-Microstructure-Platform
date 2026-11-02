# Experiment log

Every experiment that evaluated a test period is listed here, including failed and null
results. Entries are written before the test evaluation (plan) and completed after (result).

## E1 — Synthetic order-book imbalance (pipeline validation)

* Config: `configs/synthetic_demo.toml`; data: `qrp fixture` (SYNTHETIC).
* Purpose: check that the pipeline recovers a signal the generator is known to contain and that
  negative controls behave. Not evidence about markets.
* Result: see `reports/synthetic_demo.md`.

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

**Result:** (filled in after the run)
